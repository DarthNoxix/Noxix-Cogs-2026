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

from .abc import CompositeMetaClass, MixinMeta
from .common.models import DB, GuildSettings, Conversation, Embedding, NoAPIKey, EmbeddingEntryExists
from .common.constants import (
    MODELS, EMBEDDING_MODELS, VISION_MODELS, FUNCTION_CALLING_MODELS,
    GENERATE_IMAGE, SEARCH_INTERNET, CREATE_MEMORY, SEARCH_MEMORIES,
    EDIT_MEMORY, LIST_MEMORIES, DO_NOT_RESPOND_SCHEMA, RESPOND_AND_CONTINUE,
    READ_EXTENSIONS, LOADING
)
from .views import MemoryViewer, SettingsView

log = logging.getLogger("red.OpenWebUIChat")
_ = Translator("OpenWebUIChat", __file__)

MAX_MSG = 1900
FALLBACK = "❌ **OpenWebUI Assistant Error**\nPlease check your configuration:\n1. Set API endpoint: `[p]openwebuiassistant endpoint <url>`\n2. Ensure OpenWebUI is running\n3. Check model availability"
SIM_THRESHOLD = 0.8  # Cosine similarity gate (0-1), matching ChatGPT
TOP_K = 9  # Max memories sent to the LLM, matching ChatGPT

@cog_i18n(_)
class OpenWebUIMemoryBot(MixinMeta, commands.Cog, metaclass=CompositeMetaClass):
    """
    OpenWebUI Assistant - Advanced AI chat bot using OpenWebUI API
    
    Features: Chat with LLMs, memory system, function calling, auto-response, and more!
    
    **Main Commands:**
    - `/openchatask` - Chat with the OpenWebUI assistant
    - `/openchathelp` - Get detailed help information
    - `/openchataddmemory` - Add memories to the system
    - `/openchatmemoryviewer` - View and manage memories
    - `/openchatmemoryquery` - Search memories
    - `/openchatconvostats` - View conversation statistics
    - `/openchatclearconvo` - Clear your conversation
    - `/openchatshowconvo` - Show conversation history
    - `/openchattldr` - Summarize channel activity
    
    **Admin Commands:**
    - `/openwebuiassistant` - Configure the assistant settings
    - `/openwebuiassistant endpoint <url>` - Set OpenWebUI API endpoint
    - `/openwebuiassistant model <model>` - Set the chat model
    - `/openwebuiassistant prompt <prompt>` - Set system prompt
    """

    __author__ = "[Geri](https://github.com/your-repo)"
    __version__ = "1.0.0"

    def format_help_for_context(self, ctx):
        helpcmd = super().format_help_for_context(ctx)
        return f"{helpcmd}\nVersion: {self.__version__}\nAuthor: {self.__author__}"

    def __init__(self, bot: Red, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot: Red = bot
        self.q: "asyncio.Queue[Tuple[commands.Context, str]]" = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self.config = Config.get_conf(self, 0xBADA55, force_registration=True)
        self.config.register_global(db={})
        self.db: DB = DB()
        self.mp_pool = Pool()
        
        # {cog_name: {function_name: {"permission_level": "user", "schema": function_json_schema}}}
        self.registry: Dict[str, Dict[str, dict]] = {}
        
        self.saving = False
        self.first_run = True
        
        # Legacy config for backward compatibility
        self.config.register_global(
            api_base="",
            api_key="",
            chat_model="deepseek-r1:8b",
            embed_model="bge-large-en-v1.5",
            memories={},  # {name: {"text": str, "vec": List[float]}}
        )

    # ───────────────── lifecycle ─────────────────
    async def cog_load(self):
        asyncio.create_task(self.init_cog())
        self.worker = asyncio.create_task(self._worker())
        await self._setup_autocomplete()
        log.info("OpenWebUIMemoryBot, in service to Queen Alicent, is ready.")

    async def cog_unload(self):
        if self.worker:
            self.worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.worker
        self.save_loop.cancel()
        self.mp_pool.close()
        self.bot.dispatch("openwebui_assistant_cog_remove")

    async def init_cog(self):
        await self.bot.wait_until_red_ready()
        start = perf_counter()
        await self.load_db()
        log.info(f"OpenWebUIMemoryBot initialized in {perf_counter() - start:.2f}s")

    async def load_db(self):
        """Load database from config"""
        try:
            data = await self.config.db()
            if data:
                self.db = DB.model_validate(data)
            else:
                self.db = DB()
        except Exception as e:
            log.error("Failed to load database", exc_info=e)
            self.db = DB()

    async def save_conf(self):
        """Save database to config"""
        if self.saving:
            return
        self.saving = True
        try:
            await self.config.db.set(self.db.model_dump())
        except Exception as e:
            log.error("Failed to save database", exc_info=e)
        finally:
            self.saving = False

    # ───────────────── worker ─────────────────
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

        try:
            conf = self.db.get_conf(ctx.guild)
            
            # Check if API is configured
            if not conf.api_base:
                return await ctx.send("❌ **OpenWebUI API not configured!**\nPlease set the endpoint first:\n`[p]openwebuiassistant endpoint <your-openwebui-url>`")
            
            if not await self.can_call_llm(conf, ctx):
                return
            
            # Get conversation
            conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
            
            # Add user message
            conversation.add_message("user", question, ctx.author.id)
            
            # Get response using the proper OpenWebUI chat system
            response = await self.get_chat_response(conversation, conf, ctx)
            
            if response:
                # Add assistant response
                conversation.add_message("assistant", response, self.bot.user.id)
                await self.save_conf()
                
                # Handle output parameters
                params = getattr(ctx, '_openwebui_params', {})
                outputfile = params.get('outputfile')
                extract = params.get('extract', False)
                
                if outputfile:
                    from redbot.core.utils.chat_formatting import text_to_file
                    file = text_to_file(response, outputfile)
                    await ctx.send("Here's the response as a file:", file=file)
                elif extract:
                    # Extract code blocks
                    import re
                    code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)
                    if code_blocks:
                        for lang, code in code_blocks:
                            from redbot.core.utils.chat_formatting import text_to_file
                            ext = lang if lang else 'txt'
                            file = text_to_file(code, f"code.{ext}")
                            await ctx.send(f"Here's the {lang or 'code'} block:", file=file)
                    
                    # Send the rest of the response
                    text_response = re.sub(r'```(\w+)?\n.*?\n```', '', response, flags=re.DOTALL).strip()
                    if text_response:
                        await ctx.send(text_response)
                else:
                    # Send normal response
                    for part in [response[i:i + MAX_MSG] for i in range(0, len(response), MAX_MSG)]:
                        await ctx.send(part)
            else:
                await ctx.send("❌ **Failed to get response from OpenWebUI**\nPlease check:\n1. OpenWebUI is running\n2. API endpoint is correct\n3. Model is available")
                
        except Exception as e:
            log.error("Error handling message", exc_info=e)
            await ctx.send(f"❌ **Error:** {str(e)}\nPlease check your OpenWebUI configuration.")

    # ───────────────── commands ──────────────────
    @commands.hybrid_command(name="openchatask")
    @discord.app_commands.describe(
        message="Your message to the OpenWebUI assistant",
        outputfile="Optional: Save response as a file (filename with extension)",
        extract="Optional: Extract code blocks from the response",
        last="Optional: Resend the last message from the conversation"
    )
    async def openwebuichat(
        self, 
        ctx: commands.Context, 
        *, 
        message: str,
        outputfile: Optional[str] = None,
        extract: bool = False,
        last: bool = False
    ):
        """
        Chat with the OpenWebUI assistant!
        
        **Optional Arguments:**
        `outputfile` - Save response as a file (filename with extension)
        `extract` - Extract code blocks from the response
        `last` - Resend the last message from the conversation
        
        **Examples:**
        `/openchatask message: "Write a Python script" outputfile: "script.py"`
        `/openchatask message: "Explain this code" extract: true`
        `/openchatask message: "Continue" last: true`
        """
        if ctx.interaction:
            await ctx.interaction.response.defer()
        
        # Handle the last parameter
        if last:
            conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
            if not conversation.messages:
                return await ctx.send("No previous messages in this conversation!")
            
            # Get the last assistant message
            for msg in reversed(conversation.messages):
                if msg.get("role") == "assistant":
                    last_message = msg.get("content", "")
                    if outputfile:
                        from redbot.core.utils.chat_formatting import text_to_file
                        file = text_to_file(last_message, outputfile)
                        await ctx.send("Here's the last message as a file:", file=file)
                    else:
                        await ctx.send(last_message)
                    return
            
            return await ctx.send("No previous assistant messages found!")
        
        # Store the parameters for the message handler
        ctx._openwebui_params = {
            'outputfile': outputfile,
            'extract': extract
        }
        
        await self.q.put((ctx, message))

    @commands.hybrid_command(name="openchathelp")
    async def openwebui_chat_help(self, ctx: commands.Context):
        """Get detailed help for the OpenWebUI assistant."""
        embed = discord.Embed(
            title="OpenWebUI Assistant Help",
            description="Advanced AI chat bot using OpenWebUI API",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Main Commands",
            value="`/openchatask` - Chat with the assistant\n"
                  "`/openchathelp` - Get this help\n"
                  "`/openchataddmemory` - Add memories\n"
                  "`/openchatmemoryviewer` - View memories\n"
                  "`/openchatmemoryquery` - Search memories\n"
                  "`/openchatconvostats` - View conversation stats\n"
                  "`/openchatclearconvo` - Clear conversation\n"
                  "`/openchatshowconvo` - Show conversation\n"
                  "`/openchattldr` - Summarize channel activity",
            inline=False
        )
        
        embed.add_field(
            name="Admin Commands",
            value="`[p]openwebuiassistant` - Configure settings\n"
                  "`[p]openwebuiassistant endpoint <url>` - Set API endpoint\n"
                  "`[p]openwebuiassistant model <model>` - Set chat model\n"
                  "`[p]openwebuiassistant prompt <prompt>` - Set system prompt",
            inline=False
        )
        
        embed.add_field(
            name="Setup Instructions",
            value="1. Set API endpoint: `[p]openwebuiassistant endpoint <url>`\n"
                  "2. Set a model: `[p]openwebuiassistant model <model>`\n"
                  "3. Start chatting: `/openchatask message: Hello!`",
            inline=False
        )
        
        embed.set_footer(text="Use [p]openwebuiassistant for configuration options")
        await ctx.send(embed=embed)

    # ───────────────── Advanced Assistant Commands ─────────────────
    
    @commands.group(name="openwebuiassistant")
    async def openwebuiassistant(self, ctx):
        """Configure the OpenWebUI assistant for this server."""
        if ctx.invoked_subcommand is None:
            conf = self.db.get_conf(ctx.guild)
            embed = discord.Embed(
                title="OpenWebUI Assistant Configuration",
                description="Configure your OpenWebUI assistant settings",
                color=discord.Color.blue()
            )
            
            # Current settings
            embed.add_field(
                name="Current Settings",
                value=f"**API Endpoint:** {conf.api_base or 'Not set'}\n"
                      f"**Chat Model:** {conf.model}\n"
                      f"**Embedding Model:** {conf.embed_model}\n"
                      f"**Max Tokens:** {conf.max_tokens}\n"
                      f"**Temperature:** {conf.temperature}\n"
                      f"**Auto-Response:** {'Enabled' if conf.auto_response else 'Disabled'}",
                inline=False
            )
            
            # Available commands
            embed.add_field(
                name="Configuration Commands",
                value="`[p]openwebuiassistant endpoint <url>` - Set OpenWebUI API endpoint\n"
                      "`[p]openwebuiassistant model <model>` - Set the chat model\n"
                      "`[p]openwebuiassistant embedmodel <model>` - Set embedding model\n"
                      "`[p]openwebuiassistant prompt <prompt>` - Set system prompt\n"
                      "`[p]openwebuiassistant maxtokens <number>` - Set max tokens\n"
                      "`[p]openwebuiassistant temperature <0.0-2.0>` - Set temperature\n"
                      "`[p]openwebuiassistant autoreponse <true/false>` - Toggle auto-response",
                inline=False
            )
            
            # Chat commands
            embed.add_field(
                name="Chat Commands",
                value="`/openchatask` - Chat with the assistant\n"
                      "`/openchathelp` - Get detailed help\n"
                      "`/openchataddmemory` - Add memories\n"
                      "`/openchatmemoryviewer` - View memories\n"
                      "`/openchatmemoryquery` - Search memories\n"
                      "`/openchatconvostats` - View conversation stats\n"
                      "`/openchatclearconvo` - Clear conversation\n"
                      "`/openchatshowconvo` - Show conversation\n"
                      "`/openchattldr` - Summarize channel activity",
                inline=False
            )
            
            embed.set_footer(text="Use [p]openwebuiassistant <command> for more details on each command")
            await ctx.send(embed=embed)

    @openwebuiassistant.command()
    @discord.app_commands.describe(url="The OpenWebUI API endpoint URL")
    async def endpoint(self, ctx, url: str):
        """Set the OpenWebUI API endpoint for this server."""
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return await ctx.send("❌ **Invalid URL!** Please provide a valid URL starting with http:// or https://")
        
        # Remove trailing slash
        url = url.rstrip('/')
        
        # Test the endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/api/v1/models", timeout=10.0)
                if response.status_code == 200:
                    models_data = response.json()
                    models = []
                    if isinstance(models_data, dict) and "data" in models_data:
                        models = [model.get("id", model.get("name", "")) for model in models_data["data"]]
                    elif isinstance(models_data, list):
                        models = [model.get("id", model.get("name", "")) for model in models_data]
                    
                    conf = self.db.get_conf(ctx.guild)
                    conf.api_base = url
                    await self.save_conf()
                    
                    embed = discord.Embed(
                        title="✅ OpenWebUI Endpoint Set Successfully!",
                        description=f"**Endpoint:** {url}",
                        color=discord.Color.green()
                    )
                    embed.add_field(
                        name="Available Models",
                        value="\n".join(models[:10]) + (f"\n... and {len(models) - 10} more" if len(models) > 10 else ""),
                        inline=False
                    )
                    embed.add_field(
                        name="Next Steps",
                        value="1. Set a model: `[p]openwebuiassistant model <model>`\n"
                              "2. Test the chat: `/openchatask message: Hello!`",
                        inline=False
                    )
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"❌ **Connection Failed!** Status code: {response.status_code}\nPlease check if OpenWebUI is running and accessible.")
        except Exception as e:
            await ctx.send(f"❌ **Connection Error:** {str(e)}\nPlease check if the URL is correct and OpenWebUI is running.")

    @openwebuiassistant.command()
    @discord.app_commands.describe(model="The chat model to use")
    async def model(self, ctx, model: str):
        """Set the chat model for this server."""
        conf = self.db.get_conf(ctx.guild)
        conf.model = model
        await self.save_conf()
        await ctx.send(f"✅ Chat model set to `{model}`")

    @openwebuiassistant.command()
    @discord.app_commands.describe(model="The embedding model to use")
    async def embedmodel(self, ctx, model: str):
        """Set the embedding model for this server."""
        conf = self.db.get_conf(ctx.guild)
        conf.embed_model = model
        await self.save_conf()
        await ctx.send(f"✅ Embedding model set to `{model}`")

    @openwebuiassistant.command()
    async def prompt(self, ctx, *, prompt: str):
        """Set the system prompt for this server."""
        conf = self.db.get_conf(ctx.guild)
        conf.system_prompt = prompt
        await self.save_conf()
        await ctx.send(f"✅ System prompt updated")

    @openwebuiassistant.command()
    async def maxtokens(self, ctx, tokens: int):
        """Set the maximum tokens for responses."""
        if tokens < 1 or tokens > 10000:
            return await ctx.send("Max tokens must be between 1 and 10000")
        
        conf = self.db.get_conf(ctx.guild)
        conf.max_tokens = tokens
        await self.save_conf()
        await ctx.send(f"✅ Max tokens set to {tokens}")

    @openwebuiassistant.command()
    async def temperature(self, ctx, temp: float):
        """Set the temperature for responses (0.0-2.0)."""
        if temp < 0.0 or temp > 2.0:
            return await ctx.send("Temperature must be between 0.0 and 2.0")
        
        conf = self.db.get_conf(ctx.guild)
        conf.temperature = temp
        await self.save_conf()
        await ctx.send(f"✅ Temperature set to {temp}")

    @openwebuiassistant.command()
    async def autoreponse(self, ctx, enabled: bool = None):
        """Toggle auto-response functionality."""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            await ctx.send(f"Auto-response is currently {'enabled' if conf.auto_response else 'disabled'}")
        else:
            conf.auto_response = enabled
            await ctx.send(f"✅ Auto-response {'enabled' if enabled else 'disabled'}")
            await self.save_conf()

    # ───────────────── Abstract Methods Implementation ─────────────────
    
    async def openwebui_status(self) -> str:
        """Check OpenWebUI status"""
        try:
            conf = self.db.get_conf(None)  # Get global config
            if not conf.api_base:
                return "❌ No API endpoint configured"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{conf.api_base}/api/v1/models", timeout=5.0)
                if response.status_code == 200:
                    return "✅ OpenWebUI is running"
                else:
                    return f"❌ OpenWebUI returned status {response.status_code}"
        except Exception as e:
            return f"❌ Error connecting to OpenWebUI: {str(e)}"

    async def can_call_llm(self, conf: GuildSettings, ctx: commands.Context) -> bool:
        """Check if we can call the LLM"""
        if not conf.api_base:
            return False
        
        # Check if user is blacklisted
        if ctx.author.id in conf.blacklist:
            return False
        
        # Check if channel is blacklisted
        if ctx.channel.id in conf.blacklist:
            return False
        
        return True

    async def get_chat_response(self, conversation: Conversation, conf: GuildSettings, ctx: commands.Context) -> str:
        """Get chat response from OpenWebUI"""
        try:
            # Prepare messages for OpenWebUI
            messages = []
            
            # Add system prompt if configured
            if conf.system_prompt:
                messages.append({"role": "system", "content": conf.system_prompt})
            
            # Add conversation messages
            for msg in conversation.messages[-conf.max_retention:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Call OpenWebUI API
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": conf.model,
                    "messages": messages,
                    "max_tokens": conf.max_tokens,
                    "temperature": conf.temperature,
                    "stream": False
                }
                
                response = await client.post(
                    f"{conf.api_base}/api/v1/chat/completions",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                    else:
                        return "No response generated"
                else:
                    log.error(f"OpenWebUI API error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            log.error("Error getting chat response", exc_info=e)
            return None

    async def _setup_autocomplete(self):
        """Set up autocomplete for commands"""
        try:
            # Set up autocomplete for model selection
            if hasattr(self, 'model_autocomplete'):
                # This would be set up in the actual implementation
                pass
        except Exception as e:
            log.error("Failed to set up autocomplete", exc_info=e)

    # ───────────────── Event Handlers ─────────────────
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle auto-response messages"""
        if message.author.bot:
            return
        
        if not message.guild:
            return
        
        conf = self.db.get_conf(message.guild)
        if not conf.auto_response:
            return
        
        # Check if we can call the LLM
        if not await self.can_call_llm(conf, message):
            return
        
        try:
            # Get conversation for auto-response
            conversation = self.db.get_conversation(message.author.id, message.channel.id, message.guild.id)
            
            # Add user message
            conversation.add_message("user", message.content, message.author.id)
            
            # Get response using the proper OpenWebUI chat system
            response = await self.get_chat_response(conversation, conf, message)
            
            if response:
                # Add assistant response
                conversation.add_message("assistant", response, self.bot.user.id)
                await self.save_conf()
                
                # Split long responses
                for part in [response[i:i + MAX_MSG] for i in range(0, len(response), MAX_MSG)]:
                    await message.channel.send(part)
                    
        except Exception as e:
            log.error(f"Error in auto-response: {e}")

    # ───────────────── Save Loop ─────────────────
    
    @tasks.loop(minutes=5)
    async def save_loop(self):
        """Periodically save the database"""
        await self.save_conf()

    @save_loop.before_loop
    async def before_save_loop(self):
        await self.bot.wait_until_red_ready()

    def cog_unload(self):
        """Clean up when cog is unloaded"""
        if hasattr(self, 'save_loop'):
            self.save_loop.cancel()
        if hasattr(self, 'worker') and self.worker:
            self.worker.cancel()

async def setup(bot: Red):
    await bot.add_cog(OpenWebUIMemoryBot(bot))
