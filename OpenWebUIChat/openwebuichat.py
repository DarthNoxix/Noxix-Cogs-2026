import asyncio
import contextlib
import logging
import io
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
import numpy as np
import discord
from discord.ext import tasks, commands
import httpx
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.i18n import Translator, cog_i18n
from redbot.core.utils.chat_formatting import pagify, box, humanize_number
from redbot.core.utils.views import SimpleMenu
from rank_bm25 import BM25Okapi
from multiprocessing.pool import Pool
from time import perf_counter

log = logging.getLogger("red.OpenWebUIChat")
_ = Translator("OpenWebUIChat", __file__)

MAX_MSG = 1900
FALLBACK = "I'm here to help! How can I assist you today?"
SIM_THRESHOLD = 0.7  # Cosine similarity gate (0-1)
TOP_K = 5  # Max memories sent to the LLM
MAX_CONVERSATION_LENGTH = 20  # Max messages in conversation history

class ChatView(discord.ui.View):
    """Interactive view for chat responses with buttons"""
    
    def __init__(self, cog, ctx, message_id: str):
        super().__init__(timeout=300)
        self.cog = cog
        self.ctx = ctx
        self.message_id = message_id
        
    @discord.ui.button(label="🔄 Regenerate", style=discord.ButtonStyle.secondary)
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user != self.ctx.author:
            return await interaction.response.send_message("Only the original user can regenerate responses.", ephemeral=True)
        
        await interaction.response.defer()
        # Get the original message from conversation history
        conversation = await self.cog.config.guild(self.ctx.guild).conversations.get_raw(self.message_id, default=[])
        if conversation:
            last_user_msg = None
            for msg in reversed(conversation):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break
            
            if last_user_msg:
                # Remove the last AI response
                conversation = [msg for msg in conversation if msg["role"] != "assistant" or msg != conversation[-1]]
                await self.cog.config.guild(self.ctx.guild).conversations.set_raw(self.message_id, conversation)
                
                # Generate new response
                await self.cog._handle_chat(self.ctx, last_user_msg, interaction.followup)
            else:
                await interaction.followup.send("Could not find the original message to regenerate.", ephemeral=True)
        else:
            await interaction.followup.send("No conversation history found.", ephemeral=True)
    
    @discord.ui.button(label="💾 Save Memory", style=discord.ButtonStyle.primary)
    async def save_memory(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not interaction.user.guild_permissions.manage_messages:
            return await interaction.response.send_message("You need Manage Messages permission to save memories.", ephemeral=True)
        
        modal = MemoryModal(self.cog, self.ctx)
        await interaction.response.send_modal(modal)
    
    @discord.ui.button(label="🗑️ Clear Chat", style=discord.ButtonStyle.danger)
    async def clear_chat(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user != self.ctx.author:
            return await interaction.response.send_message("Only the original user can clear the chat.", ephemeral=True)
        
        await self.cog.config.guild(self.ctx.guild).conversations.set_raw(self.message_id, [])
        await interaction.response.send_message("✅ Chat history cleared!", ephemeral=True)

class MemoryModal(discord.ui.Modal):
    """Modal for saving memories"""
    
    def __init__(self, cog, ctx):
        super().__init__(title="Save Memory")
        self.cog = cog
        self.ctx = ctx
        
        self.name_input = discord.ui.TextInput(
            label="Memory Name",
            placeholder="Enter a name for this memory...",
            max_length=100,
            required=True
        )
        self.add_item(self.name_input)
        
        self.content_input = discord.ui.TextInput(
            label="Memory Content",
            placeholder="Enter the content to remember...",
            style=discord.TextStyle.paragraph,
            max_length=1000,
            required=True
        )
        self.add_item(self.content_input)
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            await self.cog._add_memory(self.name_input.value, self.content_input.value)
            await interaction.response.send_message(f"✅ Memory '{self.name_input.value}' saved successfully!", ephemeral=True)
        except ValueError as e:
            await interaction.response.send_message(f"❌ Error: {str(e)}", ephemeral=True)
        except Exception as e:
            await interaction.response.send_message(f"❌ Failed to save memory: {str(e)}", ephemeral=True)

class SetupModal(discord.ui.Modal):
    """Modal for quick setup"""
    
    def __init__(self, cog):
        super().__init__(title="OpenWebUI Setup")
        self.cog = cog
        
        self.url_input = discord.ui.TextInput(
            label="OpenWebUI URL",
            placeholder="http://localhost:8080",
            default="http://localhost:8080",
            required=True
        )
        self.add_item(self.url_input)
        
        self.key_input = discord.ui.TextInput(
            label="API Key (optional)",
            placeholder="Leave empty if no key required",
            required=False
        )
        self.add_item(self.key_input)
        
        self.chat_model_input = discord.ui.TextInput(
            label="Chat Model",
            placeholder="deepseek-r1:8b",
            default="deepseek-r1:8b",
            required=True
        )
        self.add_item(self.chat_model_input)
        
        self.embed_model_input = discord.ui.TextInput(
            label="Embedding Model",
            placeholder="bge-large-en-v1.5",
            default="bge-large-en-v1.5",
            required=True
        )
        self.add_item(self.embed_model_input)

@cog_i18n(_)
class OpenWebUIMemoryBot(commands.Cog):
    """An advanced AI assistant with OpenWebUI integration, featuring modern UI, slash commands, and auto-responses."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.q: "asyncio.Queue[Tuple[commands.Context, str]]" = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self.config = Config.get_conf(self, 0xBADA55, force_registration=True)
        
        # Global settings
        self.config.register_global(
            api_base="",
            api_key="",
            chat_model="deepseek-r1:8b",
            embed_model="bge-large-en-v1.5",
            memories={},  # {name: {"text": str, "vec": List[float]}}
            default_system_prompt="You are a helpful AI assistant. Be friendly, informative, and helpful. If you don't know something, say so honestly and offer to help in other ways.",
        )
        
        # Guild settings
        self.config.register_guild(
            auto_channels=[],  # Channel IDs for auto-response
            conversations={},  # {message_id: [{"role": str, "content": str, "timestamp": float}]}
            system_prompt="",  # Custom system prompt per guild
            rate_limit=5,  # Messages per minute per user
            max_conversation_length=20,
            enabled=True,
            user_limits={},  # {user_id: {"count": int, "reset_time": float}}
        )
        
        # User settings
        self.config.register_user(
            conversation_preferences={},
            custom_prompts={},
        )

    # ───────────────── lifecycle ─────────────────
    async def cog_load(self):
        self.worker = asyncio.create_task(self._worker())
        log.info("OpenWebUIMemoryBot is ready and ready to assist.")
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_task())

    async def cog_unload(self):
        if self.worker:
            self.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task

    # ───────────────── cleanup task ──────────────
    async def _cleanup_task(self):
        """Clean up old conversations and rate limits"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in cleanup task: {e}")

    async def _cleanup_old_data(self):
        """Clean up old conversations and expired rate limits"""
        for guild in self.bot.guilds:
            guild_config = self.config.guild(guild)
            conversations = await guild_config.conversations()
            user_limits = await guild_config.user_limits()
            
            # Clean old conversations (older than 24 hours)
            current_time = time.time()
            for msg_id, conv in list(conversations.items()):
                if conv and conv[0].get("timestamp", 0) < current_time - 86400:
                    del conversations[msg_id]
            
            # Clean expired rate limits
            for user_id, limit_data in list(user_limits.items()):
                if limit_data.get("reset_time", 0) < current_time:
                    del user_limits[user_id]
            
            await guild_config.conversations.set(conversations)
            await guild_config.user_limits.set(user_limits)

    # ───────────────── rate limiting ─────────────
    async def _check_rate_limit(self, user_id: int, guild_id: int) -> bool:
        """Check if user is within rate limits"""
        guild_config = self.config.guild_from_id(guild_id)
        rate_limit = await guild_config.rate_limit()
        user_limits = await guild_config.user_limits()
        
        current_time = time.time()
        user_data = user_limits.get(str(user_id), {"count": 0, "reset_time": current_time + 60})
        
        # Reset if time window expired
        if user_data["reset_time"] < current_time:
            user_data = {"count": 0, "reset_time": current_time + 60}
        
        # Check if within limit
        if user_data["count"] >= rate_limit:
            return False
        
        # Increment count
        user_data["count"] += 1
        user_limits[str(user_id)] = user_data
        await guild_config.user_limits.set(user_limits)
        
        return True

    # ───────────────── response formatting ───────
    def _clean_response(self, response: str) -> str:
        """Clean AI response by removing unwanted elements"""
        # Remove <think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove roleplay elements like "(We have just started...)"
        response = re.sub(r'\([^)]*We have just started[^)]*\)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\([^)]*After all[^)]*\)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\([^)]*Alright then[^)]*\)', '', response, flags=re.IGNORECASE)
        
        # Remove excessive roleplay descriptions
        response = re.sub(r'Standing tall and regal, raising chin slightly\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'A pleasure to meet you in these digital halls\.\s*', '', response, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and newlines
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = re.sub(r' +', ' ', response)
        
        return response.strip()

    # ───────────────── backend helpers ───────────
    async def _get_keys(self):
        return await asyncio.gather(
            self.config.api_base(), self.config.api_key(),
            self.config.chat_model(), self.config.embed_model()
        )

    async def _api_chat(self, messages: list, **kwargs) -> str:
        base, key, chat_model, _ = await self._get_keys()
        if not base or not key:
            raise RuntimeError("OpenWebUI URL or key not set.")
        
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": chat_model, 
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{base.rstrip('/')}/chat/completions",
                             headers=headers,
                             json=payload)
            r.raise_for_status()
            response = r.json()
            return response["choices"][0]["message"]["content"]

    async def _api_generate_image(self, prompt: str, model: str = "dall-e-3") -> str:
        """Generate an image using OpenWebUI's image generation API"""
        base, key, _, _ = await self._get_keys()
        if not base or not key:
            raise RuntimeError("OpenWebUI URL or key not set.")
        
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "standard"
        }
        
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{base.rstrip('/')}/images/generations",
                             headers=headers,
                             json=payload)
            r.raise_for_status()
            response = r.json()
            return response["data"][0]["url"]

    async def _web_search(self, query: str) -> List[Dict]:
        """Perform web search (placeholder for future implementation)"""
        # This is a placeholder for web search functionality
        # In a real implementation, you would integrate with a search API
        return [{"title": "Search Result", "snippet": f"Search results for: {query}", "url": "https://example.com"}]

    async def _api_embed(self, text: str) -> List[float]:
        base, key, _, embed_model = await self._get_keys()
        if not base or not key:
            raise RuntimeError("OpenWebUI URL or key not set.")
        
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        ollama_base = base.replace('/api', '/ollama')
        
        async with httpx.AsyncClient(timeout=60) as c:
            try:
                r = await c.post(f"{ollama_base.rstrip('/')}/api/embed",
                                headers=headers,
                                json={"model": embed_model, "input": [self._normalize_text(text)]})
                r.raise_for_status()
                return r.json()["embeddings"][0]
            except httpx.HTTPStatusError:
                r = await c.post(f"{base.rstrip('/')}/api/embeddings",
                                headers=headers,
                                json={"model": embed_model, "input": [self._normalize_text(text)]})
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\bdo\s+i\b|\bcan\s+i\b|\bhow\s+to\b|\bhow\s+do\s+i\b', 'can i', text)
        text = re.sub(r'\bi\s+download\b|\bi\s+get\b', 'i download', text)
        text = re.sub(r'\bwher\b|\bwhere\b', 'where', text)
        text = re.sub(r'\bdownlod\b|\bdl\b', 'download', text)
        return text.strip()

    # ───────────────── memory utils ──────────────
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def _best_memories(self, prompt_vec: np.ndarray, question: str, mems: Dict[str, Dict]) -> List[str]:
        if not mems:
            return []
            
        # Dense retrieval (cosine similarity)
        dense_scored = []
        for name, data in mems.items():
            vec = np.array(data["vec"])
            sim = self._cos(prompt_vec, vec)
            if sim >= SIM_THRESHOLD:
                dense_scored.append((sim, data["text"]))
        dense_scored.sort(reverse=True)
        
        # Sparse retrieval (BM25)
        texts = [data["text"] for data in mems.values()]
        bm25 = BM25Okapi([self._normalize_text(t).split() for t in texts])
        query_tokens = self._normalize_text(question).split()
        bm25_scores = bm25.get_scores(query_tokens)
        sparse_scored = [(score, text) for score, text in zip(bm25_scores, texts) if score > 0]
        sparse_scored.sort(reverse=True)
        
        # Combine dense and sparse (hybrid retrieval)
        scored = {}
        for sim, text in dense_scored:
            scored[text] = scored.get(text, 0) + sim * 0.7
        for score, text in sparse_scored:
            scored[text] = scored.get(text, 0) + score * 0.3
        sorted_scored = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        
        # Select top_k or fallback to best match
        relevant = [text for text, score in sorted_scored[:TOP_K]]
        if not relevant and sorted_scored:
            best_text = sorted_scored[0][0]
            relevant.append(best_text)
        
        return relevant

    async def _add_memory(self, name: str, text: str):
        mems = await self.config.memories()
        if name in mems:
            raise ValueError("A memory with that name already exists.")
        vec = await self._api_embed(text)
        mems[name] = {"text": text, "vec": vec}
        await self.config.memories.set(mems)
        log.info(f"Added memory '{name}' with embedding length: {len(vec)}")

    # ───────────────── conversation management ───
    async def _get_conversation_history(self, guild_id: int, message_id: str) -> List[Dict]:
        """Get conversation history for a specific message thread"""
        guild_config = self.config.guild_from_id(guild_id)
        conversations = await guild_config.conversations()
        return conversations.get(message_id, [])

    async def _add_to_conversation(self, guild_id: int, message_id: str, role: str, content: str):
        """Add a message to conversation history"""
        guild_config = self.config.guild_from_id(guild_id)
        conversations = await guild_config.conversations()
        
        if message_id not in conversations:
            conversations[message_id] = []
        
        conversations[message_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Limit conversation length
        max_length = await guild_config.max_conversation_length()
        if len(conversations[message_id]) > max_length:
            conversations[message_id] = conversations[message_id][-max_length:]
        
        await guild_config.conversations.set(conversations)

    # ───────────────── worker loop ───────────────
    async def _worker(self):
        while True:
            ctx, question = await self.q.get()
            try:
                await self._handle_chat(ctx, question)
            except Exception:
                log.exception("Error while processing the user's query")
            finally:
                self.q.task_done()

    async def _handle_chat(self, ctx: commands.Context, question: str, followup: Optional[discord.Webhook] = None):
        """Handle chat requests with modern UI and conversation management"""
        # Check rate limits
        if not await self._check_rate_limit(ctx.author.id, ctx.guild.id):
            embed = discord.Embed(
                title="⏰ Rate Limited",
                description="You're sending messages too quickly. Please wait a moment before trying again.",
                color=discord.Color.orange()
            )
            if followup:
                await followup.send(embed=embed, ephemeral=True)
            else:
                await ctx.send(embed=embed, ephemeral=True)
            return

        # Check if cog is enabled
        guild_config = self.config.guild(ctx.guild)
        if not await guild_config.enabled():
            return

        # Show typing indicator
        if not followup:
            await ctx.typing()

        # Get conversation history
        message_id = str(ctx.message.id)
        conversation = await self._get_conversation_history(ctx.guild.id, message_id)
        
        # Add user message to history
        await self._add_to_conversation(ctx.guild.id, message_id, "user", question)

        # Get system prompt
        system_prompt = await guild_config.system_prompt()
        if not system_prompt:
            system_prompt = await self.config.default_system_prompt()

        # Get memories if available
        mems = await self.config.memories()
        if mems:
            try:
                prompt_vec = np.array(await self._api_embed(question))
                relevant = await self._best_memories(prompt_vec, question, mems)
                
                if relevant:
                    system_prompt += (
                        "\n\nYou also have access to some specific knowledge that might be relevant:\n"
                        + "\n".join(f"- {t}" for t in relevant)
                    )
            except Exception as e:
                log.warning(f"Failed to retrieve memories: {e}")

        # Build messages for API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation[-10:]:  # Last 10 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            # Get AI response
            reply = await self._api_chat(messages)
            reply = self._clean_response(reply)
            
            # Add AI response to history
            await self._add_to_conversation(ctx.guild.id, message_id, "assistant", reply)

            # Create embed
            embed = discord.Embed(
                title="🤖 AI Assistant",
                description=reply[:4096],  # Discord embed limit
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            embed.set_footer(text=f"Requested by {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)

            # Create view with interactive buttons
            view = ChatView(self, ctx, message_id)

            # Send response
            if followup:
                await followup.send(embed=embed, view=view)
            else:
                await ctx.send(embed=embed, view=view)

        except Exception as e:
            log.error(f"Error in chat handling: {e}")
            error_embed = discord.Embed(
                title="❌ Error",
                description="Sorry, I encountered an error while processing your request. Please try again later.",
                color=discord.Color.red()
            )
            if followup:
                await followup.send(embed=error_embed, ephemeral=True)
            else:
                await ctx.send(embed=error_embed, ephemeral=True)

    # ───────────────── auto response ─────────────
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle auto-responses in configured channels"""
        if message.author.bot or not message.guild:
            return
        
        guild_config = self.config.guild(message.guild)
        if not await guild_config.enabled():
            return
        
        auto_channels = await guild_config.auto_channels()
        if message.channel.id not in auto_channels:
            return
        
        # Check if message mentions the bot or is a direct question
        if (self.bot.user in message.mentions or 
            message.content.lower().startswith(('hey', 'hi', 'hello', 'what', 'how', 'why', 'when', 'where', 'who'))):
            
            # Create a fake context for the message
            ctx = await self.bot.get_context(message)
            if ctx:
                await self.q.put((ctx, message.content))

    # ───────────────── slash commands ────────────
    @commands.hybrid_command(name="chat", description="Chat with the AI assistant")
    async def chat_command(self, ctx: commands.Context, *, message: str):
        """Chat with the AI assistant using slash command or prefix."""
        if ctx.interaction:
            await ctx.interaction.response.defer()
        await self.q.put((ctx, message))

    @commands.hybrid_command(name="ask", description="Ask the AI a question")
    async def ask_command(self, ctx: commands.Context, *, question: str):
        """Ask the AI a question using slash command or prefix."""
        if ctx.interaction:
            await ctx.interaction.response.defer()
        await self.q.put((ctx, question))

    # ───────────────── setup commands ────────────
    @commands.group(name="ai")
    @commands.is_owner()
    async def ai_group(self, ctx):
        """Configure the AI assistant."""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="🤖 AI Assistant Configuration",
                description="Configure your AI assistant with these commands:",
                color=discord.Color.blue()
            )
            embed.add_field(
                name="Quick Setup",
                value="`/ai setup` - Interactive setup modal",
                inline=False
            )
            embed.add_field(
                name="Configuration",
                value="`/ai url <url>` - Set OpenWebUI URL\n"
                      "`/ai key <key>` - Set API key\n"
                      "`/ai models <chat> <embed>` - Set models",
                inline=False
            )
            embed.add_field(
                name="Channel Management",
                value="`/ai auto <channel>` - Add auto-response channel\n"
                      "`/ai remove <channel>` - Remove auto-response channel",
                inline=False
            )
            await ctx.send(embed=embed)

    @ai_group.command(name="setup")
    async def setup_command(self, ctx):
        """Interactive setup with modal."""
        modal = SetupModal(self)
        await ctx.send_modal(modal)

    @ai_group.command(name="url")
    async def set_url(self, ctx, url: str):
        """Set the OpenWebUI API endpoint."""
        await self.config.api_base.set(url)
        embed = discord.Embed(
            title="✅ URL Set",
            description=f"OpenWebUI URL set to: `{url}`",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @ai_group.command(name="key")
    async def set_key(self, ctx, key: str):
        """Set the API key."""
        await self.config.api_key.set(key)
        embed = discord.Embed(
            title="✅ API Key Set",
            description="API key has been configured.",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @ai_group.command(name="models")
    async def set_models(self, ctx, chat_model: str, embed_model: str):
        """Set the chat and embedding models."""
        await self.config.chat_model.set(chat_model)
        await self.config.embed_model.set(embed_model)
        embed = discord.Embed(
            title="✅ Models Set",
            description=f"**Chat Model:** `{chat_model}`\n**Embedding Model:** `{embed_model}`",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @ai_group.command(name="auto")
    async def add_auto_channel(self, ctx, channel: discord.TextChannel):
        """Add a channel for auto-responses."""
        guild_config = self.config.guild(ctx.guild)
        auto_channels = await guild_config.auto_channels()
        
        if channel.id in auto_channels:
            embed = discord.Embed(
                title="⚠️ Already Added",
                description=f"{channel.mention} is already configured for auto-responses.",
                color=discord.Color.orange()
            )
        else:
            auto_channels.append(channel.id)
            await guild_config.auto_channels.set(auto_channels)
            embed = discord.Embed(
                title="✅ Auto-Response Added",
                description=f"{channel.mention} will now receive auto-responses from the AI.",
                color=discord.Color.green()
            )
        
        await ctx.send(embed=embed)

    @ai_group.command(name="remove")
    async def remove_auto_channel(self, ctx, channel: discord.TextChannel):
        """Remove a channel from auto-responses."""
        guild_config = self.config.guild(ctx.guild)
        auto_channels = await guild_config.auto_channels()
        
        if channel.id not in auto_channels:
            embed = discord.Embed(
                title="⚠️ Not Found",
                description=f"{channel.mention} is not configured for auto-responses.",
                color=discord.Color.orange()
            )
        else:
            auto_channels.remove(channel.id)
            await guild_config.auto_channels.set(auto_channels)
            embed = discord.Embed(
                title="✅ Auto-Response Removed",
                description=f"{channel.mention} will no longer receive auto-responses.",
                color=discord.Color.green()
            )
        
        await ctx.send(embed=embed)

    @ai_group.command(name="status")
    async def status_command(self, ctx):
        """Show AI assistant status and configuration."""
        base, key, chat_model, embed_model = await self._get_keys()
        guild_config = self.config.guild(ctx.guild)
        
        auto_channels = await guild_config.auto_channels()
        enabled = await guild_config.enabled()
        rate_limit = await guild_config.rate_limit()
        
        embed = discord.Embed(
            title="🤖 AI Assistant Status",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Configuration",
            value=f"**URL:** `{base or 'Not set'}`\n"
                  f"**API Key:** {'✅ Set' if key else '❌ Not set'}\n"
                  f"**Chat Model:** `{chat_model}`\n"
                  f"**Embed Model:** `{embed_model}`",
            inline=False
        )
        
        embed.add_field(
            name="Guild Settings",
            value=f"**Enabled:** {'✅ Yes' if enabled else '❌ No'}\n"
                  f"**Rate Limit:** {rate_limit} messages/minute\n"
                  f"**Auto Channels:** {len(auto_channels)} configured",
            inline=False
        )
        
        if auto_channels:
            channel_mentions = []
            for channel_id in auto_channels:
                channel = ctx.guild.get_channel(channel_id)
                if channel:
                    channel_mentions.append(channel.mention)
            embed.add_field(
                name="Auto-Response Channels",
                value="\n".join(channel_mentions) if channel_mentions else "None",
                inline=False
            )
        
        await ctx.send(embed=embed)

    # ───────────────── memory management ─────────
    @commands.group(name="memory")
    @commands.is_owner()
    async def memory_group(self, ctx):
        """Manage the AI's knowledge base."""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="🧠 Memory Management",
                description="Manage the AI's knowledge base:",
                color=discord.Color.purple()
            )
            embed.add_field(
                name="Commands",
                value="`/memory add <name> <content>` - Add memory\n"
                      "`/memory list` - List all memories\n"
                      "`/memory delete <name>` - Delete memory\n"
                      "`/memory search <query>` - Search memories",
                inline=False
            )
            await ctx.send(embed=embed)

    @memory_group.command(name="add")
    async def add_memory(self, ctx, name: str, *, content: str):
        """Add a memory to the knowledge base."""
        try:
            await self._add_memory(name, content)
            embed = discord.Embed(
                title="✅ Memory Added",
                description=f"Memory '{name}' has been added to the knowledge base.",
                color=discord.Color.green()
            )
        except ValueError as e:
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red()
            )
        except Exception as e:
            embed = discord.Embed(
                title="❌ Error",
                description=f"Failed to add memory: {str(e)}",
                color=discord.Color.red()
            )
        
        await ctx.send(embed=embed)

    @memory_group.command(name="list")
    async def list_memories(self, ctx):
        """List all memories in the knowledge base."""
        mems = await self.config.memories()
        if not mems:
            embed = discord.Embed(
                title="🧠 Knowledge Base",
                description="The knowledge base is empty.",
                color=discord.Color.blue()
            )
        else:
            embed = discord.Embed(
                title="🧠 Knowledge Base",
                description=f"Found {len(mems)} memories:",
                color=discord.Color.blue()
            )
            
            for name, data in list(mems.items())[:10]:  # Show first 10
                content = data["text"][:100] + "..." if len(data["text"]) > 100 else data["text"]
                embed.add_field(
                    name=name,
                    value=content,
                    inline=False
                )
            
            if len(mems) > 10:
                embed.set_footer(text=f"... and {len(mems) - 10} more memories")
        
        await ctx.send(embed=embed)

    @memory_group.command(name="delete")
    async def delete_memory(self, ctx, name: str):
        """Delete a memory from the knowledge base."""
        mems = await self.config.memories()
        if name not in mems:
            embed = discord.Embed(
                title="❌ Not Found",
                description=f"No memory named '{name}' exists in the knowledge base.",
                color=discord.Color.red()
            )
        else:
            del mems[name]
            await self.config.memories.set(mems)
            embed = discord.Embed(
                title="✅ Memory Deleted",
                description=f"Memory '{name}' has been removed from the knowledge base.",
            color=discord.Color.green()
        )
        
        await ctx.send(embed=embed)

    @memory_group.command(name="search")
    async def search_memories(self, ctx, *, query: str):
        """Search memories in the knowledge base."""
        mems = await self.config.memories()
        if not mems:
            embed = discord.Embed(
                title="🧠 Search Results",
                description="The knowledge base is empty.",
                color=discord.Color.blue()
            )
        else:
            try:
                query_vec = np.array(await self._api_embed(query))
                relevant = await self._best_memories(query_vec, query, mems)
                
                if relevant:
                    embed = discord.Embed(
                        title="🧠 Search Results",
                        description=f"Found {len(relevant)} relevant memories:",
                        color=discord.Color.blue()
                    )
                    
                    for i, memory in enumerate(relevant[:5], 1):  # Show top 5
                        # Find the memory name
                        memory_name = None
                        for name, data in mems.items():
                            if data["text"] == memory:
                                memory_name = name
                                break
                        
                        content = memory[:200] + "..." if len(memory) > 200 else memory
                        embed.add_field(
                            name=f"{i}. {memory_name or 'Unknown'}",
                            value=content,
                            inline=False
                        )
                else:
                    embed = discord.Embed(
                        title="🧠 Search Results",
                        description="No relevant memories found for your query.",
                        color=discord.Color.orange()
                    )
            except Exception as e:
                embed = discord.Embed(
                    title="❌ Search Error",
                    description=f"Failed to search memories: {str(e)}",
                    color=discord.Color.red()
                )
        
        await ctx.send(embed=embed)

    # ───────────────── admin commands ────────────
    @commands.group(name="aiadmin")
    @commands.has_permissions(manage_guild=True)
    async def ai_admin_group(self, ctx):
        """Admin commands for the AI assistant."""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(
                title="⚙️ AI Admin Commands",
                description="Admin commands for managing the AI assistant:",
                color=discord.Color.orange()
            )
            embed.add_field(
                name="Commands",
                value="`/aiadmin enable` - Enable AI in this guild\n"
                      "`/aiadmin disable` - Disable AI in this guild\n"
                      "`/aiadmin ratelimit <number>` - Set rate limit\n"
                      "`/aiadmin prompt <text>` - Set custom system prompt\n"
                      "`/aiadmin clear` - Clear all conversations",
                inline=False
            )
            await ctx.send(embed=embed)

    @ai_admin_group.command(name="enable")
    async def enable_ai(self, ctx):
        """Enable AI assistant in this guild."""
        guild_config = self.config.guild(ctx.guild)
        await guild_config.enabled.set(True)
        embed = discord.Embed(
            title="✅ AI Enabled",
            description="AI assistant has been enabled in this guild.",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @ai_admin_group.command(name="disable")
    async def disable_ai(self, ctx):
        """Disable AI assistant in this guild."""
        guild_config = self.config.guild(ctx.guild)
        await guild_config.enabled.set(False)
        embed = discord.Embed(
            title="❌ AI Disabled",
            description="AI assistant has been disabled in this guild.",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed)

    @ai_admin_group.command(name="ratelimit")
    async def set_rate_limit(self, ctx, limit: int):
        """Set the rate limit for AI responses."""
        if limit < 1 or limit > 60:
            embed = discord.Embed(
                title="❌ Invalid Limit",
                description="Rate limit must be between 1 and 60 messages per minute.",
                color=discord.Color.red()
            )
        else:
            guild_config = self.config.guild(ctx.guild)
            await guild_config.rate_limit.set(limit)
            embed = discord.Embed(
                title="✅ Rate Limit Set",
                description=f"Rate limit set to {limit} messages per minute.",
                color=discord.Color.green()
            )
        await ctx.send(embed=embed)

    @ai_admin_group.command(name="prompt")
    async def set_system_prompt(self, ctx, *, prompt: str):
        """Set a custom system prompt for this guild."""
        guild_config = self.config.guild(ctx.guild)
        await guild_config.system_prompt.set(prompt)
        embed = discord.Embed(
            title="✅ System Prompt Set",
            description="Custom system prompt has been configured for this guild.",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    @ai_admin_group.command(name="clear")
    async def clear_conversations(self, ctx):
        """Clear all conversation history in this guild."""
        guild_config = self.config.guild(ctx.guild)
        await guild_config.conversations.set({})
        embed = discord.Embed(
            title="✅ Conversations Cleared",
            description="All conversation history has been cleared.",
            color=discord.Color.green()
        )
        await ctx.send(embed=embed)

    # ───────────────── advanced features ─────────
    @commands.hybrid_command(name="generate", description="Generate an image using AI")
    async def generate_image(self, ctx, *, prompt: str):
        """Generate an image using AI."""
        if ctx.interaction:
            await ctx.interaction.response.defer()
        
        try:
            await ctx.typing()
            image_url = await self._api_generate_image(prompt)
            
            embed = discord.Embed(
                title="🎨 Generated Image",
                description=f"**Prompt:** {prompt}",
                color=discord.Color.purple()
            )
            embed.set_image(url=image_url)
            embed.set_footer(text=f"Generated for {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)
            
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed)
            else:
                await ctx.send(embed=embed)
                
        except Exception as e:
            log.error(f"Error generating image: {e}")
            embed = discord.Embed(
                title="❌ Image Generation Failed",
                description="Sorry, I couldn't generate an image. Please try again later.",
                color=discord.Color.red()
            )
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, ephemeral=True)
            else:
                await ctx.send(embed=embed)

    @commands.hybrid_command(name="search", description="Search the web for information")
    async def web_search(self, ctx, *, query: str):
        """Search the web for information."""
        if ctx.interaction:
            await ctx.interaction.response.defer()
        
        try:
            await ctx.typing()
            results = await self._web_search(query)
            
            embed = discord.Embed(
                title="🔍 Search Results",
                description=f"**Query:** {query}",
                color=discord.Color.blue()
            )
            
            for i, result in enumerate(results[:3], 1):  # Show top 3 results
                embed.add_field(
                    name=f"{i}. {result['title']}",
                    value=f"{result['snippet']}\n[Read more]({result['url']})",
                    inline=False
                )
            
            embed.set_footer(text=f"Searched for {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)
            
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed)
            else:
                await ctx.send(embed=embed)
                
        except Exception as e:
            log.error(f"Error in web search: {e}")
            embed = discord.Embed(
                title="❌ Search Failed",
                description="Sorry, I couldn't search the web. Please try again later.",
                color=discord.Color.red()
            )
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, ephemeral=True)
            else:
                await ctx.send(embed=embed)

    @commands.hybrid_command(name="code", description="Execute code or get code help")
    async def code_help(self, ctx, *, code_or_question: str):
        """Get help with code or execute simple code snippets."""
        if ctx.interaction:
            await ctx.interaction.response.defer()
        
        # Enhanced system prompt for code assistance
        code_system_prompt = (
            "You are an expert programming assistant. Help with code questions, debugging, "
            "and provide clear, well-commented examples. If the user provides code, analyze it "
            "and suggest improvements or explain what it does. Always provide practical, "
            "working examples when possible."
        )
        
        try:
            await ctx.typing()
            
            # Get conversation history for context
            message_id = str(ctx.message.id)
            conversation = await self._get_conversation_history(ctx.guild.id, message_id)
            
            # Add user message to history
            await self._add_to_conversation(ctx.guild.id, message_id, "user", code_or_question)
            
            # Build messages for API
            messages = [{"role": "system", "content": code_system_prompt}]
            
            # Add conversation history
            for msg in conversation[-5:]:  # Last 5 messages for code context
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            reply = await self._api_chat(messages)
            reply = self._clean_response(reply)
            
            # Add AI response to history
            await self._add_to_conversation(ctx.guild.id, message_id, "assistant", reply)
            
            # Create embed with code formatting
            embed = discord.Embed(
                title="💻 Code Assistant",
                description=reply[:4096],
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.set_footer(text=f"Code help for {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)
            
            # Create view with interactive buttons
            view = ChatView(self, ctx, message_id)
            
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, view=view)
            else:
                await ctx.send(embed=embed, view=view)
                
        except Exception as e:
            log.error(f"Error in code assistance: {e}")
            embed = discord.Embed(
                title="❌ Code Assistance Error",
                description="Sorry, I encountered an error while helping with your code. Please try again later.",
                color=discord.Color.red()
            )
            if ctx.interaction:
                await ctx.interaction.followup.send(embed=embed, ephemeral=True)
            else:
                await ctx.send(embed=embed)


async def setup(bot: Red):
    await bot.add_cog(OpenWebUIMemoryBot(bot))