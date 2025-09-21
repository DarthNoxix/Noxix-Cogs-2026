import asyncio
import contextlib
import logging
import io
from typing import Dict, List, Optional, Tuple, Union, Callable
import re
import numpy as np
import discord
from discord.ext import tasks
import httpx
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.i18n import Translator, cog_i18n
from redbot.core.utils.chat_formatting import pagify, box, humanize_number
from rank_bm25 import BM25Okapi
from multiprocessing.pool import Pool
from time import perf_counter

log = logging.getLogger("red.OpenWebUIChat")
_ = Translator("OpenWebUIChat", __file__)

MAX_MSG = 1900
FALLBACK = "I'm here to help! How can I assist you today?"
SIM_THRESHOLD = 0.8  # Cosine similarity gate (0-1), matching ChatGPT
TOP_K = 9  # Max memories sent to the LLM, matching ChatGPT
EMBED_CHUNK = 3900  # Safe embed description chunk size

@cog_i18n(_)
class OpenWebUIMemoryBot(commands.Cog):
    """An AI assistant that can chat and help with various topics, with optional memory/knowledge base functionality."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.q: "asyncio.Queue[Tuple[commands.Context, str]]" = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self.config = Config.get_conf(self, 0xBADA55, force_registration=True)
        self.config.register_global(
            api_base="",
            api_key="",
            chat_model="deepseek-r1:8b",
            embed_model="bge-large-en-v1.5",
            memories={},  # {name: {"text": str, "vec": List[float]}}
        )
        # Per-guild settings
        self.config.register_guild(
            auto_channels=[],  # List[int]
            mention_only=False,
        )

    # ───────────────── lifecycle ─────────────────
    async def cog_load(self):
        self.worker = asyncio.create_task(self._worker())
        log.info("OpenWebUIMemoryBot is ready and ready to assist.")

    async def cog_unload(self):
        if self.worker:
            self.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker

    # ───────────────── backend helpers ───────────
    async def _get_keys(self):
        return await asyncio.gather(
            self.config.api_base(), self.config.api_key(),
            self.config.chat_model(), self.config.embed_model()
        )

    async def _api_chat(self, messages: list) -> str:
        base, key, chat_model, _ = await self._get_keys()
        if not base or not key:
            raise RuntimeError("OpenWebUI URL or key not set.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{base.rstrip('/')}/chat/completions",
                             headers=headers,
                             json={"model": chat_model, "messages": messages})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

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

    def _sanitize_reply(self, text: str) -> str:
        """Remove chain-of-thought markers like <think>...</think> and stage directions."""
        if not text:
            return FALLBACK
        # Remove <think> blocks
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # Remove leading stage directions like (thinking ...)
        text = re.sub(r"^\s*\((?:[^()]*|\([^()]*\))*\)\s*", "", text)
        return text.strip() or FALLBACK

    async def _generate_reply(self, question: str, mems: Optional[Dict[str, Dict]] = None) -> str:
        """Build messages, call chat API, sanitize and return reply."""
        # Start with a general system prompt and discourage chain-of-thought
        system = (
            "You are a helpful AI assistant. Be concise, friendly, factual, and helpful. "
            "Do not include hidden reasoning, chain-of-thought, or <think> content. Answer directly. "
            "If uncertain, say you don't know and suggest next steps."
        )

        # If we have memories, try to find relevant ones and enhance the system prompt
        if mems:
            try:
                prompt_vec = np.array(await self._api_embed(question))
                relevant = await self._best_memories(prompt_vec, question, mems)
                if relevant:
                    system += (
                        "\n\nYou also have access to some specific knowledge that might be relevant:\n"
                        + "\n".join(f"- {t}" for t in relevant)
                    )
            except Exception as e:
                log.warning(f"Failed to retrieve memories: {e}")

        reply = await self._api_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ])
        return self._sanitize_reply(reply)

    def _build_embeds(self, content: str, author: Optional[discord.abc.User], model_name: Optional[str]) -> List[discord.Embed]:
        """Chunk long content into multiple modern embeds."""
        chunks = [content[i:i + EMBED_CHUNK] for i in range(0, len(content), EMBED_CHUNK)] or [FALLBACK]
        embeds: List[discord.Embed] = []
        for idx, chunk in enumerate(chunks, start=1):
            embed = discord.Embed(description=chunk, color=discord.Color.blurple())
            if author:
                try:
                    embed.set_author(name="AI Assistant", icon_url=author.display_avatar.url)
                except Exception:
                    embed.set_author(name="AI Assistant")
            else:
                embed.set_author(name="AI Assistant")
            footer = "OpenWebUIChat"
            if model_name:
                footer += f" • {model_name}"
            if len(chunks) > 1:
                footer += f" • Part {idx}/{len(chunks)}"
            embed.set_footer(text=footer)
            embeds.append(embed)
        return embeds

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
        # Dense retrieval (cosine similarity)
        dense_scored = []
        for name, data in mems.items():
            vec = np.array(data["vec"])
            sim = self._cos(prompt_vec, vec)
            log.info(f"Memory '{name}' dense similarity: {sim:.3f}")
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
            scored[text] = scored.get(text, 0) + sim * 0.7  # Weight dense higher
        for score, text in sparse_scored:
            scored[text] = scored.get(text, 0) + score * 0.3  # Weight sparse lower
        sorted_scored = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        
        # Select top_k or fallback to best match
        relevant = [text for text, score in sorted_scored[:TOP_K]]
        if not relevant and sorted_scored:
            best_text = sorted_scored[0][0]
            log.info(f"No memories above threshold, using best match: '{best_text}' (score: {sorted_scored[0][1]:.3f})")
            relevant.append(best_text)
        
        log.info(f"Selected memories: {relevant}")
        return relevant

    async def _add_memory(self, name: str, text: str):
        mems = await self.config.memories()
        if name in mems:
            raise ValueError("A memory with that name already exists.")
        vec = await self._api_embed(text)
        mems[name] = {"text": text, "vec": vec}
        await self.config.memories.set(mems)
        log.info(f"Added memory '{name}' with embedding length: {len(vec)}")

    # ───────────────── worker loop ───────────────
    async def _worker(self):
        while True:
            ctx, question = await self.q.get()
            try:
                await self._handle(ctx, question)
            except Exception:
                log.exception("Error while processing the courtier's query")
            finally:
                self.q.task_done()

    async def _handle(self, ctx: commands.Context, question: str):
        await ctx.typing()
        mems = await self.config.memories()
        reply = await self._generate_reply(question, mems)
        log.info(f"AI response (sanitized): '{reply}'")
        model = await self.config.chat_model()
        embeds = self._build_embeds(reply, ctx.author, model)
        for embed in embeds:
            await ctx.send(embed=embed)

    # ───────────────── commands ──────────────────
    @commands.hybrid_command()
    async def llmchat(self, ctx: commands.Context, *, message: str):
        """Chat with the AI assistant."""
        if ctx.interaction:
            await ctx.interaction.response.defer()
        await self.q.put((ctx, message))

    @commands.hybrid_command(name="llmmodal")
    async def llm_modal(self, ctx: commands.Context):
        """Open a modal to chat with the AI (slash only)."""
        if not ctx.interaction:
            return await ctx.send("Use the slash version of this command to open a modal.")
        class PromptModal(discord.ui.Modal, title="Ask the AI"):
            prompt = discord.ui.TextInput(label="Your message", style=discord.TextStyle.paragraph, max_length=4000)
            def __init__(self, outer: "OpenWebUIMemoryBot"):
                super().__init__()
                self.outer = outer
            async def on_submit(self, interaction: discord.Interaction):
                await interaction.response.defer()
                reply = await self.outer._generate_reply(str(self.prompt.value), await self.outer.config.memories())
                model = await self.outer.config.chat_model()
                embeds = self.outer._build_embeds(reply, interaction.user, model)
                for embed in embeds:
                    await interaction.followup.send(embed=embed)
        await ctx.interaction.response.send_modal(PromptModal(self))

    # ───────────────── setup & memory management ─────────────────
    @commands.hybrid_group()
    @commands.is_owner()
    async def setopenwebui(self, ctx):
        """Configure the connection to OpenWebUI."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @setopenwebui.command()
    async def url(self, ctx, url: str):
        await self.config.api_base.set(url)
        await ctx.send("✅ URL set.")

    @setopenwebui.command()
    async def key(self, ctx, key: str):
        await self.config.api_key.set(key)
        await ctx.send("✅ Key set.")

    @setopenwebui.command()
    async def chatmodel(self, ctx, model: str):
        await self.config.chat_model.set(model)
        await ctx.send(f"✅ Chat model set to {model}.")

    @setopenwebui.command()
    async def embedmodel(self, ctx, model: str):
        await self.config.embed_model.set(model)
        await ctx.send(f"✅ Embed model set to {model}.")

    @commands.hybrid_group(name="openwebuimemory")
    @commands.is_owner()
    async def openwebuimemory(self, ctx):
        """Manage the knowledge base/memories."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebuimemory.command()
    async def add(self, ctx, name: str, *, text: str):
        """Add a memory to the knowledge base."""
        try:
            await self._add_memory(name, text)
        except ValueError as e:
            await ctx.send(str(e))
        else:
            await ctx.send("✅ Memory added to knowledge base.")

    @openwebuimemory.command(name="list")
    async def _list(self, ctx):
        mems = await self.config.memories()
        if not mems:
            return await ctx.send("*The knowledge base is empty.*")
        out = "\n".join(f"- **{n}**: {d['text'][:80]}…" for n, d in mems.items())
        await ctx.send(out)

    @openwebuimemory.command(name="del")
    async def _del(self, ctx, name: str):
        mems = await self.config.memories()
        if name not in mems:
            return await ctx.send("No such memory exists in the knowledge base.")
        del mems[name]
        await self.config.memories.set(mems)
        await ctx.send("❌ Memory removed from the knowledge base.")

    # ───────────────── auto-channel replies ─────────────────
    @commands.hybrid_group(name="openwebui")
    @commands.is_owner()
    async def openwebui(self, ctx: commands.Context):
        """OpenWebUI settings commands."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebui.group(name="autochannel")
    @commands.is_owner()
    async def autochannel(self, ctx: commands.Context):
        """Manage auto-reply channels."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @autochannel.command(name="add")
    async def autochannel_add(self, ctx: commands.Context, channel: discord.TextChannel):
        conf = self.config.guild(ctx.guild)
        chs = await conf.auto_channels()
        if channel.id in chs:
            return await ctx.send("Channel already enabled for auto replies.")
        chs.append(channel.id)
        await conf.auto_channels.set(chs)
        await ctx.send(f"✅ Enabled auto replies in {channel.mention}")

    @autochannel.command(name="remove")
    async def autochannel_remove(self, ctx: commands.Context, channel: discord.TextChannel):
        conf = self.config.guild(ctx.guild)
        chs = await conf.auto_channels()
        if channel.id not in chs:
            return await ctx.send("Channel is not enabled for auto replies.")
        chs = [c for c in chs if c != channel.id]
        await conf.auto_channels.set(chs)
        await ctx.send(f"❌ Disabled auto replies in {channel.mention}")

    @autochannel.command(name="list")
    async def autochannel_list(self, ctx: commands.Context):
        chs = await self.config.guild(ctx.guild).auto_channels()
        if not chs:
            return await ctx.send("No auto-reply channels configured.")
        mentions = [ctx.guild.get_channel(cid).mention for cid in chs if ctx.guild.get_channel(cid)]
        await ctx.send("Auto-reply channels: " + ", ".join(mentions))

    @autochannel.command(name="mentiononly")
    async def autochannel_mentiononly(self, ctx: commands.Context, value: bool):
        await self.config.guild(ctx.guild).mention_only.set(value)
        await ctx.send(f"Mention-only mode set to {value}.")

    @commands.Cog.listener()
    async def on_message_without_command(self, message: discord.Message):
        if not message.guild or message.author.bot:
            return
        conf = self.config.guild(message.guild)
        auto_channels = await conf.auto_channels()
        if message.channel.id not in auto_channels:
            return
        mention_only = await conf.mention_only()
        if mention_only and self.bot.user not in getattr(message, "mentions", []):
            return
        ctx = await self.bot.get_context(message)
        if ctx.valid:  # don't trigger if it's a command
            return
        await self.q.put((ctx, message.clean_content))


async def setup(bot: Red):
    await bot.add_cog(OpenWebUIMemoryBot(bot))
