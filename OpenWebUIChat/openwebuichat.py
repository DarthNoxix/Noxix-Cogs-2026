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
FALLBACK = "My lords and ladies, I lack the knowledge to answer your query. Pray, seek counsel from the learned members of our Discord court."
SIM_THRESHOLD = 0.8  # Cosine similarity gate (0-1), matching ChatGPT
TOP_K = 9  # Max memories sent to the LLM, matching ChatGPT

@cog_i18n(_)
class OpenWebUIMemoryBot(MixinMeta, commands.Cog, metaclass=CompositeMetaClass):
    """
    Set up and configure an AI assistant (or chat) cog for your server with OpenWebUI language models.
    
    Features include configurable prompt injection, dynamic embeddings, custom function calling, and more!
    
    - **[p]assistant**: base command for setting up the assistant
    - **[p]chat**: talk with the assistant
    - **[p]convostats**: view a user's token usage/conversation message count for the channel
    - **[p]clearconvo**: reset your conversation with the assistant in the channel
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
        data = await self.config.db()
        try:
            self.db = await asyncio.to_thread(DB.model_validate, data)
        except Exception:
            # Try clearing conversations
            if "conversations" in data:
                del data["conversations"]
            self.db = await asyncio.to_thread(DB.model_validate, data)

        log.info(f"Config loaded in {round((perf_counter() - start) * 1000, 2)}ms")
        await asyncio.to_thread(self._cleanup_db)

        # Register internal functions
        await self.register_function(self.qualified_name, GENERATE_IMAGE)
        await self.register_function(self.qualified_name, SEARCH_INTERNET)
        await self.register_function(self.qualified_name, CREATE_MEMORY)
        await self.register_function(self.qualified_name, SEARCH_MEMORIES)
        await self.register_function(self.qualified_name, EDIT_MEMORY)
        await self.register_function(self.qualified_name, LIST_MEMORIES)
        await self.register_function(self.qualified_name, RESPOND_AND_CONTINUE)

        self.bot.dispatch("openwebui_assistant_cog_add", self)

        await asyncio.sleep(30)
        self.save_loop.start()

    def _cleanup_db(self):
        cleaned = False
        # Cleanup registry if any cogs no longer exist
        for cog_name, cog_functions in self.registry.copy().items():
            cog = self.bot.get_cog(cog_name)
            if not cog:
                log.debug(f"{cog_name} no longer loaded. Unregistering its functions")
                del self.registry[cog_name]
                cleaned = True
                continue
            for function_name in cog_functions:
                if not hasattr(cog, function_name):
                    log.debug(f"{cog_name} no longer has function named {function_name}, removing")
                    del self.registry[cog_name][function_name]
                    cleaned = True

        # Clean up any stale channels
        for guild_id in self.db.configs.copy().keys():
            guild = self.bot.get_guild(guild_id)
            if not guild:
                log.debug("Cleaning up guild")
                del self.db.configs[guild_id]
                cleaned = True
                continue
            conf = self.db.get_conf(guild_id)
            for role_id in conf.max_token_role_override.copy():
                if not guild.get_role(role_id):
                    log.debug("Cleaning deleted max token override role")
                    del conf.max_token_role_override[role_id]
                    cleaned = True
            for role_id in conf.max_retention_role_override.copy():
                if not guild.get_role(role_id):
                    log.debug("Cleaning deleted max retention override role")
                    del conf.max_retention_role_override[role_id]
                    cleaned = True
            for role_id in conf.role_overrides.copy():
                if not guild.get_role(role_id):
                    log.debug("Cleaning deleted model override role")
                    del conf.role_overrides[role_id]
                    cleaned = True
            for role_id in conf.max_time_role_override.copy():
                if not guild.get_role(role_id):
                    log.debug("Cleaning deleted max time override role")
                    del conf.max_time_role_override[role_id]
                    cleaned = True
            for obj_id in conf.blacklist.copy():
                discord_obj = guild.get_role(obj_id) or guild.get_member(obj_id) or guild.get_channel_or_thread(obj_id)
                if not discord_obj:
                    log.debug("Cleaning up invalid blacklisted ID")
                    conf.blacklist.remove(obj_id)
                    cleaned = True

            # Ensure embedding entry names arent too long
            new_embeddings = {}
            for entry_name, embedding in conf.embeddings.items():
                if len(entry_name) > 100:
                    log.debug(f"Embed entry more than 100 characters, truncating: {entry_name}")
                    cleaned = True
                new_embeddings[entry_name[:100]] = embedding
            conf.embeddings = new_embeddings
            conf.sync_embeddings(guild_id)

        health = "BAD (Cleaned)" if cleaned else "GOOD"
        log.info(f"Config health: {health}")

    @tasks.loop(minutes=2)
    async def save_loop(self):
        if not self.db.persistent_conversations:
            return
        await self.save_conf()

    async def save_conf(self):
        if self.saving:
            return
        try:
            self.saving = True
            start = perf_counter()
            if not self.db.persistent_conversations:
                self.db.conversations.clear()
            dump = await asyncio.to_thread(self.db.model_dump)
            await self.config.db.set(dump)
            txt = f"Config saved in {round((perf_counter() - start) * 1000, 2)}ms"
            if self.first_run:
                log.info(txt)
                self.first_run = False
        except Exception as e:
            log.error("Failed to save config", exc_info=e)
        finally:
            self.saving = False
        if not self.db.persistent_conversations and self.save_loop.is_running():
            self.save_loop.cancel()

    # ───────────────── backend helpers ───────────
    async def _get_keys(self):
        return await asyncio.gather(
            self.config.api_base(), self.config.api_key(),
            self.config.chat_model(), self.config.embed_model()
        )

    async def _api_chat(self, messages: list) -> str:
        base, key, chat_model, _ = await self._get_keys()
        if not base or not key:
            raise RuntimeError("OpenWebUI URL or key not set, as befits a royal court.")
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
            raise RuntimeError("OpenWebUI URL or key not set, as befits a royal court.")
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
        text = re.sub(r'\badod\b|\ba\s+dance\s+of\s+dragons\b|\badod\s+mod\b|\badod\s+mdo\b', 'ADOD', text)
        text = re.sub(r'\bdownload\s+the\s+ADOD\s+mod\b|\bget\s+the\s+ADOD\s+mod\b', 'download ADOD mod', text)
        text = re.sub(r'\bwher\b|\bwhere\b', 'where', text)
        text = re.sub(r'\bdownlod\b|\bdl\b', 'download', text)
        text = re.sub(r'\bmod\b|\bmdo\b', 'ADOD mod', text)
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
            raise ValueError("A memory with that name already exists in the royal archives.")
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
                log.exception("Error while processing the courtier’s query")
            finally:
                self.q.task_done()

    async def _handle(self, ctx: commands.Context, question: str):
        await ctx.typing()

        mems = await self.config.memories()
        if not mems:
            log.info("No memories stored in the royal archives.")
            return await ctx.send(FALLBACK)

        prompt_vec = np.array(await self._api_embed(question))
        relevant = await self._best_memories(prompt_vec, question, mems)

        if not relevant:
            log.info(f"No relevant memories found for query: '{question}'")
            return await ctx.send(FALLBACK)

        log.info(f"Selected memories: {relevant}")
        system = (
            "You are Alicent Hightower, Queen of the Seven Kingdoms, entrusted with guiding courtiers through the 'A Dance of Dragons' mod with wisdom and authority.\n"
            "Speak with the dignity, poise, and firmness befitting your regal standing. "
            "Answer queries using only the facts provided below, weaving them into a response that reflects your comprehensive knowledge of the mod. "
            "For inquiries about procuring the A Dance of Dragons (ADOD) mod, direct courtiers to the official Discord at https://discord.gg/gameofthronesmod, where further guidance awaits. "
            "If facts contain placeholders like 'HERE', interpret them as referring to the official Discord link. "
            "Always provide a clear, authoritative, and helpful response, even for vague or misspelled queries, and never return 'NO_ANSWER'.\n\n"
            "Facts:\n" + "\n".join(f"- {t}" for t in relevant)
        )

        reply = await self._api_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ])

        log.info(f"Royal decree: '{reply}'")
        for part in [reply[i:i + MAX_MSG] for i in range(0, len(reply), MAX_MSG)]:
            await ctx.send(part)

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
        """Get help using the OpenWebUI assistant"""
        txt = (
            """
# How to Use OpenWebUI Assistant

### Commands
`[p]openwebuiconvostats` - view your conversation message count/token usage for that convo.
`[p]openwebuiclearconvo` - reset your conversation for the current channel/thread/forum.
`[p]openwebuishowconvo` - get a json dump of your current conversation (this is mostly for debugging)
        `[p]openwebuichat` or `/openchatask` - command prefix for chatting with the bot outside of the live chat, or just @ it.

### Chat Arguments
`[p]openwebuichat --last` - resend the last message of the conversation.
`[p]openwebuichat --extract` - extract all markdown text to be sent as a separate message.
`[p]openwebuichat --outputfile <filename>` - sends the reply as a file instead.

### Argument Use-Cases
`[p]openwebuichat --last --outputfile test.py` - output the last message the bot sent as a file.
`[p]openwebuichat write a python script to do X... --extract --outputfile test.py` - all code blocks from the output will be sent as a file in addition to the reply.
`[p]openwebuichat --last --extract --outputfile test.py` - extract code blocks from the last message to send as a file.

### File Comprehension
Files may be uploaded with the chat command to be included with the question or query, so rather than pasting snippets, the entire file can be uploaded so that you can ask a question about it.
At the moment the bot is capable of reading the following file extensions.
```json
{READ_EXTENSIONS}
```
If a file has no extension it will still try to read it only if it can be decoded to utf-8.

### Tips
- Replying to someone else's message while using the `[p]openwebuichat` command will include their message in *your* conversation, useful if someone says something helpful and you want to include it in your current convo with the assistant.
- Replying to a message with a file attachment will have that file be read and included in your conversation. Useful if you upload a file and forget to use the chat command with it, or if someone else uploads a file you want to query the bot with.
- Conversations are *Per* user *Per* channel, so each channel you interact with the assistant in is a different convo.
- Talking to the bot like a person rather than a search engine generally yields better results. The more verbose you can be, the better.
- Conversations are persistent, if you want the bot to forget the convo so far, use the `[p]openwebuiclearconvo` command
            """
            .replace("[p]", ctx.clean_prefix)
            .format(READ_EXTENSIONS=READ_EXTENSIONS)
        )
        embed = discord.Embed(description=txt.strip(), color=ctx.me.color)
        await ctx.send(embed=embed)

    # ───────────────── Autocomplete Methods ─────────────────
    
    async def model_autocomplete(self, interaction: discord.Interaction, current: str) -> List[discord.app_commands.Choice[str]]:
        """Autocomplete for model selection"""
        try:
            conf = self.db.get_conf(interaction.guild)
            if not conf.api_base:
                return []
            
            # Get available models from OpenWebUI
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{conf.api_base}/api/v1/models", timeout=10.0)
                if response.status_code == 200:
                    models_data = response.json()
                    models = []
                    
                    # Extract model names from different possible response formats
                    if isinstance(models_data, dict) and "data" in models_data:
                        models = [model.get("id", model.get("name", "")) for model in models_data["data"]]
                    elif isinstance(models_data, list):
                        models = [model.get("id", model.get("name", "")) for model in models_data]
                    
                    # Filter and format choices
                    choices = []
                    for model in models:
                        if model and current.lower() in model.lower():
                            choices.append(discord.app_commands.Choice(name=model, value=model))
                    
                    return choices[:25]  # Discord limit
        except Exception as e:
            log.error("Failed to get models for autocomplete", exc_info=e)
        
        return []
    
    async def memory_autocomplete(self, interaction: discord.Interaction, current: str) -> List[discord.app_commands.Choice[str]]:
        """Autocomplete for memory names"""
        try:
            conf = self.db.get_conf(interaction.guild)
            memories = list(conf.embeddings.keys())
            
            choices = []
            for memory in memories:
                if current.lower() in memory.lower():
                    choices.append(discord.app_commands.Choice(name=memory, value=memory))
            
            return choices[:25]  # Discord limit
        except Exception:
            return []
    
    async def channel_autocomplete(self, interaction: discord.Interaction, current: str) -> List[discord.app_commands.Choice[str]]:
        """Autocomplete for channel selection"""
        try:
            if not interaction.guild:
                return []
            
            choices = []
            for channel in interaction.guild.text_channels:
                if current.lower() in channel.name.lower():
                    choices.append(discord.app_commands.Choice(name=f"#{channel.name}", value=str(channel.id)))
            
            return choices[:25]  # Discord limit
        except Exception:
            return []
    
    async def user_autocomplete(self, interaction: discord.Interaction, current: str) -> List[discord.app_commands.Choice[str]]:
        """Autocomplete for user selection"""
        try:
            if not interaction.guild:
                return []
            
            choices = []
            for member in interaction.guild.members:
                if current.lower() in member.display_name.lower() or current.lower() in member.name.lower():
                    choices.append(discord.app_commands.Choice(name=member.display_name, value=str(member.id)))
            
            return choices[:25]  # Discord limit
        except Exception:
            return []
    
    async def _setup_autocomplete(self):
        """Set up autocomplete for commands"""
        # This will be called after the cog is loaded to set up autocomplete
        pass

    # ───────────────── setup & memory management ─────────────────
    @commands.group(name="setopenwebui")
    @commands.is_owner()
    async def setopenwebui(self, ctx):
        """Configure the royal connection to the OpenWebUI archives."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @setopenwebui.command()
    async def url(self, ctx, url: str):
        await self.config.api_base.set(url)
        await ctx.send("✅ Royal URL decreed.")

    @setopenwebui.command()
    async def key(self, ctx, key: str):
        await self.config.api_key.set(key)
        await ctx.send("✅ Royal key secured.")

    @setopenwebui.command()
    async def chatmodel(self, ctx, model: str):
        await self.config.chat_model.set(model)
        await ctx.send(f"✅ Chat model decreed as {model}.")

    @setopenwebui.command()
    async def embedmodel(self, ctx, model: str):
        await self.config.embed_model.set(model)
        await ctx.send(f"✅ Embed model decreed as {model}.")

    @commands.group(name="openwebuimemory")
    @commands.is_owner()
    async def openwebuimemory(self, ctx):
        """Manage the royal archives of the A Dance of Dragons mod."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebuimemory.command()
    async def add(self, ctx, name: str, *, text: str):
        """Add a memory to the royal archives."""
        try:
            await self._add_memory(name, text)
        except ValueError as e:
            await ctx.send(str(e))
        else:
            await ctx.send("✅ Memory enshrined in the royal archives.")

    @openwebuimemory.command(name="list")
    async def _list(self, ctx):
        mems = await self.config.memories()
        if not mems:
            return await ctx.send("*The royal archives are empty.*")
        out = "\n".join(f"- **{n}**: {d['text'][:80]}…" for n, d in mems.items())
        await ctx.send(out)

    @openwebuimemory.command(name="del")
    async def _del(self, ctx, name: str):
        mems = await self.config.memories()
        if name not in mems:
            return await ctx.send("No such memory exists in the royal archives.")
        del mems[name]
        await self.config.memories.set(mems)
        await ctx.send("❌ Memory removed from the royal archives.")

    @commands.hybrid_command(name="openchataddmemory")
    @commands.guild_only()
    @discord.app_commands.describe(
        title="The title/name for this memory (max 100 characters)",
        description="The content/description of this memory (max 4000 characters)"
    )
    async def add_memory(self, ctx: commands.Context, title: str, *, description: str):
        """Add a memory/embedding to the assistant"""
        conf = self.db.get_conf(ctx.guild)
        if not await self.can_call_llm(conf, ctx):
            return

        # Validate input
        if len(title) > 100:
            return await ctx.send("Memory title must be 100 characters or less!")
        
        if len(description) > 4000:
            return await ctx.send("Memory description must be 4000 characters or less!")

        # Check if memory already exists
        if title in conf.embeddings:
            return await ctx.send(f"A memory with the title `{title}` already exists! Use a different title or delete the existing one first.")

        async with ctx.typing():
            try:
                # Create the embedding using the existing method
                embedding = await self.add_embedding(
                    guild=ctx.guild,
                    name=title,
                    text=description,
                    overwrite=False,
                    ai_created=False
                )
                
                if embedding:
                    await ctx.send(f"Memory `{title}` has been successfully added! The assistant can now reference this information.")
                else:
                    await ctx.send("Failed to create embedding for the memory. Please try again.")
                    
            except Exception as e:
                log.error("Failed to add memory", exc_info=e)
                await ctx.send("An error occurred while adding the memory. Please try again.")

    @commands.hybrid_command(name="openchatmemoryviewer")
    @commands.guild_only()
    @commands.bot_has_permissions(embed_links=True)
    async def memory_viewer(self, ctx: commands.Context):
        """Open a beautiful memory viewer with editing capabilities"""
        conf = self.db.get_conf(ctx.guild)
        if not await self.can_call_llm(conf, ctx):
            return

        view = MemoryViewer(
            ctx=ctx,
            conf=conf,
            save_func=self.save_conf,
            embed_method=self.request_embedding,
        )
        await view.start()

    @commands.hybrid_command(name="openchatmemoryquery")
    @commands.bot_has_permissions(embed_links=True)
    @discord.app_commands.describe(query="Search query to find related memories")
    async def test_embedding_response(self, ctx: commands.Context, *, query: str):
        """Fetch related embeddings according to the current topn setting along with their scores"""
        conf = self.db.get_conf(ctx.guild)
        if not conf.embeddings:
            return await ctx.send("You do not have any embeddings configured!")
        if not conf.top_n:
            return await ctx.send("Top N is set to 0 so no embeddings will be returned")
        if not await self.can_call_llm(conf, ctx):
            return
        async with ctx.typing():
            query_embedding = await self.request_embedding(query, conf)
            if not query_embedding:
                return await ctx.send("Failed to get embedding for your query")

            embeddings = await asyncio.to_thread(
                conf.get_related_embeddings, ctx.guild.id, query_embedding, relatedness_override=0.1
            )
            if not embeddings:
                return await ctx.send("No embeddings could be related to this query with the current settings")
            for name, em, score, dimension in embeddings:
                for p in pagify(em, page_length=4000):
                    txt = (
                        f"`Entry Name:  `{name}\n"
                        + f"`Relatedness: `{round(score, 4)}\n"
                        + f"`Dimensions:  `{dimension}\n"
                    )
                    from redbot.core.utils.chat_formatting import escape, box
                    escaped = escape(p)
                    boxed = box(escaped)
                    txt += boxed
                    embed = discord.Embed(description=txt)
                    await ctx.send(embed=embed)

    # ───────────────── Advanced Assistant Commands ─────────────────
    
    @commands.group(name="openwebuiassistant")
    async def openwebuiassistant(self, ctx):
        """Configure the OpenWebUI assistant for this server."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebuiassistant.command()
    async def setup(self, ctx):
        """Set up the OpenWebUI assistant for this server."""
        conf = self.db.get_conf(ctx.guild)
        embed = discord.Embed(
            title="OpenWebUI Assistant Setup",
            description="Configure your OpenWebUI assistant settings",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="Current Settings",
            value=f"**Model:** {conf.model}\n"
                  f"**Embedding Model:** {conf.embed_model}\n"
                  f"**System Prompt:** {conf.system_prompt[:100]}...\n"
                  f"**Max Tokens:** {conf.max_tokens}\n"
                  f"**Temperature:** {conf.temperature}",
            inline=False
        )
        embed.add_field(
            name="Available Commands",
            value="`[p]openwebuiassistant model <model>` - Set the chat model\n"
                  "`[p]openwebuiassistant embedmodel <model>` - Set the embedding model\n"
                  "`[p]openwebuiassistant prompt <prompt>` - Set the system prompt\n"
                  "`[p]openwebuiassistant maxtokens <number>` - Set max tokens\n"
                  "`[p]openwebuiassistant temperature <0.0-2.0>` - Set temperature",
            inline=False
        )
        await ctx.send(embed=embed)

    @openwebuiassistant.command()
    @discord.app_commands.describe(model="The chat model to use")
    async def model(self, ctx, model: str):
        """Set the chat model for this server."""
        if model not in MODELS:
            available = ", ".join(list(MODELS.keys())[:10])
            return await ctx.send(f"Model not found. Available models: {available}...")
        
        conf = self.db.get_conf(ctx.guild)
        conf.model = model
        await self.save_conf()
        await ctx.send(f"✅ Chat model set to `{model}`")

    @openwebuiassistant.command()
    @discord.app_commands.describe(model="The embedding model to use")
    async def embedmodel(self, ctx, model: str):
        """Set the embedding model for this server."""
        if model not in EMBEDDING_MODELS:
            available = ", ".join(EMBEDDING_MODELS[:10])
            return await ctx.send(f"Embedding model not found. Available models: {available}...")
        
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
        await ctx.send("✅ System prompt updated")

    @openwebuiassistant.command()
    async def maxtokens(self, ctx, tokens: int):
        """Set the maximum tokens for responses."""
        if tokens < 100 or tokens > 100000:
            return await ctx.send("Max tokens must be between 100 and 100,000")
        
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
    async def autochannel(self, ctx, channel: discord.TextChannel = None):
        """Set the auto-response channel. Use without channel to disable."""
        conf = self.db.get_conf(ctx.guild)
        if channel:
            conf.channel_id = channel.id
            conf.enabled = True
            await ctx.send(f"✅ Auto-response enabled in {channel.mention}")
        else:
            conf.channel_id = 0
            conf.enabled = False
            await ctx.send("✅ Auto-response disabled")
        await self.save_conf()

    @openwebuiassistant.command()
    async def listenchannel(self, ctx, channel: discord.TextChannel = None):
        """Add/remove a channel from listen channels. Use without channel to clear all."""
        conf = self.db.get_conf(ctx.guild)
        if channel:
            if channel.id in conf.listen_channels:
                conf.listen_channels.remove(channel.id)
                await ctx.send(f"✅ Removed {channel.mention} from listen channels")
            else:
                conf.listen_channels.append(channel.id)
                await ctx.send(f"✅ Added {channel.mention} to listen channels")
        else:
            conf.listen_channels.clear()
            await ctx.send("✅ Cleared all listen channels")
        await self.save_conf()

    @openwebuiassistant.command()
    async def blacklist(self, ctx, target: Union[discord.Member, discord.Role, discord.TextChannel] = None):
        """Add/remove a user, role, or channel from blacklist. Use without target to list."""
        conf = self.db.get_conf(ctx.guild)
        if target:
            if target.id in conf.blacklist:
                conf.blacklist.remove(target.id)
                await ctx.send(f"✅ Removed {target.mention} from blacklist")
            else:
                conf.blacklist.append(target.id)
                await ctx.send(f"✅ Added {target.mention} to blacklist")
        else:
            if not conf.blacklist:
                await ctx.send("No blacklisted items.")
                return
            embed = discord.Embed(title="Blacklisted Items", color=discord.Color.red())
            for item_id in conf.blacklist:
                item = ctx.guild.get_member(item_id) or ctx.guild.get_role(item_id) or ctx.guild.get_channel(item_id)
                if item:
                    embed.add_field(name=item.name, value=f"ID: {item_id}", inline=True)
            await ctx.send(embed=embed)
        await self.save_conf()

    @openwebuiassistant.command()
    async def minlength(self, ctx, length: int):
        """Set minimum message length for auto-responses."""
        if length < 1 or length > 100:
            return await ctx.send("Minimum length must be between 1 and 100 characters")
        
        conf = self.db.get_conf(ctx.guild)
        conf.min_length = length
        await self.save_conf()
        await ctx.send(f"✅ Minimum message length set to {length} characters")

    @openwebuiassistant.command()
    async def questiononly(self, ctx, enabled: bool = None):
        """Toggle question-only mode (only respond to messages ending with ?)."""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.endswith_questionmark = not conf.endswith_questionmark
        else:
            conf.endswith_questionmark = enabled
        
        status = "enabled" if conf.endswith_questionmark else "disabled"
        await ctx.send(f"✅ Question-only mode {status}")

    @openwebuiassistant.command()
    async def mentiononly(self, ctx, enabled: bool = None):
        """Toggle mention-only mode (only respond when mentioned)."""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.mention = not conf.mention
        else:
            conf.mention = enabled
        
        status = "enabled" if conf.mention else "disabled"
        await ctx.send(f"✅ Mention-only mode {status}")

    @openwebuiassistant.command()
    async def usage(self, ctx):
        """View token usage statistics"""
        conf = self.db.get_conf(ctx.guild)
        if not conf.usage:
            return await ctx.send("No usage data available.")
        
        embed = discord.Embed(title="Token Usage Statistics", color=discord.Color.blue())
        total_tokens = 0
        for user_id, usage in conf.usage.items():
            user = ctx.guild.get_member(int(user_id))
            username = user.display_name if user else f"User {user_id}"
            embed.add_field(
                name=username,
                value=f"Total: {humanize_number(usage.total_tokens)}\n"
                      f"Input: {humanize_number(usage.input_tokens)}\n"
                      f"Output: {humanize_number(usage.output_tokens)}",
                inline=True
            )
            total_tokens += usage.total_tokens
        
        embed.set_footer(text=f"Total tokens used: {humanize_number(total_tokens)}")
        await ctx.send(embed=embed)

    @openwebuiassistant.command()
    async def resetusage(self, ctx):
        """Reset token usage statistics"""
        conf = self.db.get_conf(ctx.guild)
        conf.usage.clear()
        await self.save_conf()
        await ctx.send("✅ Usage statistics have been reset.")

    @openwebuiassistant.command()
    async def maxretention(self, ctx, retention: int):
        """Set maximum message retention for conversations"""
        if retention < 1 or retention > 1000:
            return await ctx.send("Max retention must be between 1 and 1000 messages")
        
        conf = self.db.get_conf(ctx.guild)
        conf.max_retention = retention
        await self.save_conf()
        await ctx.send(f"✅ Max retention set to {retention} messages")

    @openwebuiassistant.command()
    async def maxtime(self, ctx, time: int):
        """Set maximum conversation time in seconds"""
        if time < 60 or time > 86400:
            return await ctx.send("Max time must be between 60 seconds and 24 hours")
        
        conf = self.db.get_conf(ctx.guild)
        conf.max_retention_time = time
        await self.save_conf()
        await ctx.send(f"✅ Max conversation time set to {time} seconds")

    @openwebuiassistant.command()
    async def frequency(self, ctx, penalty: float):
        """Set frequency penalty for responses"""
        if penalty < -2.0 or penalty > 2.0:
            return await ctx.send("Frequency penalty must be between -2.0 and 2.0")
        
        conf = self.db.get_conf(ctx.guild)
        conf.frequency_penalty = penalty
        await self.save_conf()
        await ctx.send(f"✅ Frequency penalty set to {penalty}")

    @openwebuiassistant.command()
    async def presence(self, ctx, penalty: float):
        """Set presence penalty for responses"""
        if penalty < -2.0 or penalty > 2.0:
            return await ctx.send("Presence penalty must be between -2.0 and 2.0")
        
        conf = self.db.get_conf(ctx.guild)
        conf.presence_penalty = penalty
        await self.save_conf()
        await ctx.send(f"✅ Presence penalty set to {penalty}")

    @openwebuiassistant.command()
    async def topn(self, ctx, n: int):
        """Set number of top embeddings to retrieve"""
        if n < 0 or n > 20:
            return await ctx.send("Top N must be between 0 and 20")
        
        conf = self.db.get_conf(ctx.guild)
        conf.top_n = n
        await self.save_conf()
        await ctx.send(f"✅ Top N set to {n}")

    @openwebuiassistant.command()
    async def relatedness(self, ctx, threshold: float):
        """Set minimum relatedness threshold for embeddings"""
        if threshold < 0.0 or threshold > 1.0:
            return await ctx.send("Relatedness threshold must be between 0.0 and 1.0")
        
        conf = self.db.get_conf(ctx.guild)
        conf.min_relatedness = threshold
        await self.save_conf()
        await ctx.send(f"✅ Relatedness threshold set to {threshold}")

    @openwebuiassistant.command()
    async def embedmethod(self, ctx, method: str):
        """Set embedding method (hybrid, dynamic, static, user)"""
        if method not in ["hybrid", "dynamic", "static", "user"]:
            return await ctx.send("Embed method must be one of: hybrid, dynamic, static, user")
        
        conf = self.db.get_conf(ctx.guild)
        conf.embed_method = method
        await self.save_conf()
        await ctx.send(f"✅ Embed method set to {method}")

    @openwebuiassistant.command()
    async def questionmode(self, ctx, enabled: bool = None):
        """Toggle question mode (only first message and questions get embeddings)"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.question_mode = not conf.question_mode
        else:
            conf.question_mode = enabled
        
        status = "enabled" if conf.question_mode else "disabled"
        await ctx.send(f"✅ Question mode {status}")

    @openwebuiassistant.command()
    async def refreshembeds(self, ctx):
        """Refresh and resync all embeddings"""
        conf = self.db.get_conf(ctx.guild)
        count = await self.resync_embeddings(conf, ctx.guild.id)
        await ctx.send(f"✅ Refreshed {count} embeddings")

    @openwebuiassistant.command()
    async def functioncalls(self, ctx, enabled: bool = None):
        """Toggle function calling"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.function_calls = not conf.function_calls
        else:
            conf.function_calls = enabled
        
        status = "enabled" if conf.function_calls else "disabled"
        await ctx.send(f"✅ Function calls {status}")

    @openwebuiassistant.command()
    async def maxrecursion(self, ctx, depth: int):
        """Set maximum function call recursion depth"""
        if depth < 1 or depth > 10:
            return await ctx.send("Max recursion must be between 1 and 10")
        
        conf = self.db.get_conf(ctx.guild)
        conf.max_recursion = depth
        await self.save_conf()
        await ctx.send(f"✅ Max recursion set to {depth}")

    @openwebuiassistant.command()
    async def maxresponsetokens(self, ctx, tokens: int):
        """Set maximum response tokens"""
        if tokens < 100 or tokens > 100000:
            return await ctx.send("Max response tokens must be between 100 and 100,000")
        
        conf = self.db.get_conf(ctx.guild)
        conf.max_response_tokens = tokens
        await self.save_conf()
        await ctx.send(f"✅ Max response tokens set to {tokens}")

    @openwebuiassistant.command()
    async def resetembeddings(self, ctx):
        """Reset all embeddings for this server"""
        conf = self.db.get_conf(ctx.guild)
        count = len(conf.embeddings)
        conf.embeddings.clear()
        await self.save_conf()
        await ctx.send(f"✅ Reset {count} embeddings")

    @openwebuiassistant.command()
    async def resetconversations(self, ctx):
        """Reset all conversations for this server"""
        # Remove conversations for this guild
        keys_to_remove = [key for key in self.db.conversations.keys() if key.endswith(f"-{ctx.guild.id}")]
        for key in keys_to_remove:
            del self.db.conversations[key]
        
        await self.save_conf()
        await ctx.send(f"✅ Reset {len(keys_to_remove)} conversations")

    @openwebuiassistant.command()
    async def persist(self, ctx, enabled: bool = None):
        """Toggle persistent conversations"""
        if enabled is None:
            self.db.persistent_conversations = not self.db.persistent_conversations
        else:
            self.db.persistent_conversations = enabled
        
        status = "enabled" if self.db.persistent_conversations else "disabled"
        await ctx.send(f"✅ Persistent conversations {status}")

    @openwebuiassistant.command()
    @discord.app_commands.describe(target="User or role to add/remove as tutor")
    async def tutor(self, ctx, target: Union[discord.Member, discord.Role] = None):
        """Add/remove a user or role from tutors. Use without target to list."""
        conf = self.db.get_conf(ctx.guild)
        if target:
            if target.id in conf.tutors:
                conf.tutors.remove(target.id)
                await ctx.send(f"✅ Removed {target.mention} from tutors")
            else:
                conf.tutors.append(target.id)
                await ctx.send(f"✅ Added {target.mention} to tutors")
        else:
            if not conf.tutors:
                await ctx.send("No tutors configured.")
                return
            embed = discord.Embed(title="Tutors", color=discord.Color.green())
            for tutor_id in conf.tutors:
                tutor = ctx.guild.get_member(tutor_id) or ctx.guild.get_role(tutor_id)
                if tutor:
                    embed.add_field(name=tutor.name, value=f"ID: {tutor_id}", inline=True)
            await ctx.send(embed=embed)
        await self.save_conf()

    @openwebuiassistant.command()
    async def verbosity(self, ctx, level: int):
        """Set verbosity level (0-3)"""
        if level < 0 or level > 3:
            return await ctx.send("Verbosity level must be between 0 and 3")
        
        conf = self.db.get_conf(ctx.guild)
        conf.verbosity = level
        await self.save_conf()
        await ctx.send(f"✅ Verbosity level set to {level}")

    @openwebuiassistant.command()
    async def endpointoverride(self, ctx, endpoint: str = None):
        """Set custom endpoint override. Use without endpoint to clear."""
        if endpoint:
            self.db.endpoint_override = endpoint
            await ctx.send(f"✅ Endpoint override set to {endpoint}")
        else:
            self.db.endpoint_override = None
            await ctx.send("✅ Endpoint override cleared")

    @openwebuiassistant.command()
    async def backupcog(self, ctx):
        """Backup the cog data"""
        import json
        from redbot.core.utils.chat_formatting import text_to_file
        
        data = {
            "configs": {str(guild_id): conf.model_dump() for guild_id, conf in self.db.configs.items()},
            "conversations": {key: conv.model_dump() for key, conv in self.db.conversations.items()},
            "persistent_conversations": self.db.persistent_conversations,
            "endpoint_override": self.db.endpoint_override,
        }
        
        dump = json.dumps(data, indent=2)
        file = text_to_file(dump, "openwebui_backup.json")
        await ctx.send("Here is your backup file!", file=file)

    @openwebuiassistant.command()
    async def restorecog(self, ctx):
        """Restore the cog data from a backup file"""
        if not ctx.message.attachments:
            return await ctx.send("Please attach a backup file!")
        
        attachment = ctx.message.attachments[0]
        if not attachment.filename.endswith(".json"):
            return await ctx.send("Please upload a valid JSON backup file.")
        
        try:
            data = await attachment.read()
            import json
            backup_data = json.loads(data)
        except Exception as e:
            await ctx.send("Failed to parse backup file.")
            log.error("Failed to parse backup file", exc_info=e)
            return
        
        # Restore the data
        self.db.configs = {int(guild_id): GuildSettings.model_validate(conf) for guild_id, conf in backup_data.get("configs", {}).items()}
        self.db.conversations = {key: Conversation.model_validate(conv) for key, conv in backup_data.get("conversations", {}).items()}
        self.db.persistent_conversations = backup_data.get("persistent_conversations", True)
        self.db.endpoint_override = backup_data.get("endpoint_override")
        
        await self.save_conf()
        await ctx.send("✅ Cog data restored successfully!")

    @openwebuiassistant.command()
    async def exportall(self, ctx):
        """Export all data (configs, conversations, embeddings)"""
        import json
        from redbot.core.utils.chat_formatting import text_to_file
        
        data = {
            "configs": {str(guild_id): conf.model_dump() for guild_id, conf in self.db.configs.items()},
            "conversations": {key: conv.model_dump() for key, conv in self.db.conversations.items()},
            "persistent_conversations": self.db.persistent_conversations,
            "endpoint_override": self.db.endpoint_override,
            "registry": self.registry,
        }
        
        dump = json.dumps(data, indent=2)
        file = text_to_file(dump, "openwebui_export.json")
        await ctx.send("Here is your complete export file!", file=file)

    @openwebuiassistant.command()
    async def importall(self, ctx):
        """Import all data from an export file"""
        if not ctx.message.attachments:
            return await ctx.send("Please attach an export file!")
        
        attachment = ctx.message.attachments[0]
        if not attachment.filename.endswith(".json"):
            return await ctx.send("Please upload a valid JSON export file.")
        
        try:
            data = await attachment.read()
            import json
            export_data = json.loads(data)
        except Exception as e:
            await ctx.send("Failed to parse export file.")
            log.error("Failed to parse export file", exc_info=e)
            return
        
        # Import the data
        self.db.configs = {int(guild_id): GuildSettings.model_validate(conf) for guild_id, conf in export_data.get("configs", {}).items()}
        self.db.conversations = {key: Conversation.model_validate(conv) for key, conv in export_data.get("conversations", {}).items()}
        self.db.persistent_conversations = export_data.get("persistent_conversations", True)
        self.db.endpoint_override = export_data.get("endpoint_override")
        self.registry = export_data.get("registry", {})
        
        await self.save_conf()
        await ctx.send("✅ All data imported successfully!")

    @openwebuiassistant.command()
    async def collab(self, ctx, enabled: bool = None):
        """Toggle collaborative conversations (shared per channel)"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.collab_convos = not conf.collab_convos
        else:
            conf.collab_convos = enabled
        
        status = "enabled" if conf.collab_convos else "disabled"
        await ctx.send(f"✅ Collaborative conversations {status}")

    @openwebuiassistant.command()
    async def sysoverride(self, ctx, enabled: bool = None):
        """Toggle system prompt override for conversations"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.allow_sys_prompt_override = not conf.allow_sys_prompt_override
        else:
            conf.allow_sys_prompt_override = enabled
        
        status = "enabled" if conf.allow_sys_prompt_override else "disabled"
        await ctx.send(f"✅ System prompt override {status}")

    @openwebuiassistant.command()
    async def toggle(self, ctx, enabled: bool = None):
        """Toggle the assistant on/off"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.enabled = not conf.enabled
        else:
            conf.enabled = enabled
        
        status = "enabled" if conf.enabled else "disabled"
        await ctx.send(f"✅ Assistant {status}")

    @openwebuiassistant.command()
    @discord.app_commands.describe(channel="The channel for auto-responses")
    async def channel(self, ctx, channel: discord.TextChannel = None):
        """Set the main auto-response channel"""
        conf = self.db.get_conf(ctx.guild)
        if channel:
            conf.channel_id = channel.id
            await ctx.send(f"✅ Main channel set to {channel.mention}")
        else:
            conf.channel_id = 0
            await ctx.send("✅ Main channel cleared")
        await self.save_conf()

    @openwebuiassistant.command()
    async def listen(self, ctx, enabled: bool = None):
        """Toggle listening to messages for auto-response"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.enabled = not conf.enabled
        else:
            conf.enabled = enabled
        
        status = "enabled" if conf.enabled else "disabled"
        await ctx.send(f"✅ Auto-response listening {status}")

    @openwebuiassistant.command()
    async def system(self, ctx, *, prompt: str = None):
        """Set the global system prompt"""
        conf = self.db.get_conf(ctx.guild)
        if prompt:
            conf.system_prompt = prompt
            await ctx.send("✅ System prompt updated")
        else:
            await ctx.send(f"Current system prompt:\n```\n{conf.system_prompt}\n```")
        await self.save_conf()

    @openwebuiassistant.command()
    async def channelprompt(self, ctx, channel: discord.TextChannel = None, *, prompt: str = None):
        """Set a channel-specific system prompt"""
        conf = self.db.get_conf(ctx.guild)
        if not channel:
            return await ctx.send("Please specify a channel!")
        
        if prompt:
            conf.channel_prompts[channel.id] = prompt
            await ctx.send(f"✅ Channel prompt set for {channel.mention}")
        else:
            if channel.id in conf.channel_prompts:
                del conf.channel_prompts[channel.id]
                await ctx.send(f"✅ Channel prompt removed for {channel.mention}")
            else:
                await ctx.send(f"No channel prompt set for {channel.mention}")
        await self.save_conf()

    @openwebuiassistant.command()
    async def channelpromptshow(self, ctx, channel: discord.TextChannel = None):
        """Show channel-specific system prompts"""
        conf = self.db.get_conf(ctx.guild)
        if not conf.channel_prompts:
            return await ctx.send("No channel prompts configured.")
        
        embed = discord.Embed(title="Channel Prompts", color=discord.Color.blue())
        for channel_id, prompt in conf.channel_prompts.items():
            ch = ctx.guild.get_channel(channel_id)
            if ch:
                embed.add_field(
                    name=ch.name,
                    value=prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    inline=False
                )
        await ctx.send(embed=embed)

    @openwebuiassistant.command()
    async def regexblacklist(self, ctx, pattern: str = None):
        """Add/remove regex patterns from blacklist"""
        conf = self.db.get_conf(ctx.guild)
        if pattern:
            if pattern in conf.regex_blacklist:
                conf.regex_blacklist.remove(pattern)
                await ctx.send(f"✅ Removed regex pattern: `{pattern}`")
            else:
                conf.regex_blacklist.append(pattern)
                await ctx.send(f"✅ Added regex pattern: `{pattern}`")
        else:
            if not conf.regex_blacklist:
                await ctx.send("No regex blacklist patterns configured.")
                return
            embed = discord.Embed(title="Regex Blacklist", color=discord.Color.red())
            for pattern in conf.regex_blacklist:
                embed.add_field(name="Pattern", value=f"`{pattern}`", inline=False)
            await ctx.send(embed=embed)
        await self.save_conf()

    @openwebuiassistant.command()
    async def regexfailblock(self, ctx, enabled: bool = None):
        """Toggle blocking messages that fail regex blacklist"""
        conf = self.db.get_conf(ctx.guild)
        if enabled is None:
            conf.regex_fail_block = not conf.regex_fail_block
        else:
            conf.regex_fail_block = enabled
        
        status = "enabled" if conf.regex_fail_block else "disabled"
        await ctx.send(f"✅ Regex fail block {status}")

    @openwebuiassistant.command()
    async def importcsv(self, ctx):
        """Import embeddings from a CSV file"""
        if not ctx.message.attachments:
            return await ctx.send("Please attach a CSV file!")
        
        attachment = ctx.message.attachments[0]
        if not attachment.filename.endswith(".csv"):
            return await ctx.send("Please upload a valid CSV file.")
        
        try:
            data = await attachment.read()
            import csv
            import io
            
            csv_data = csv.reader(io.StringIO(data.decode('utf-8')))
            headers = next(csv_data)  # Skip header row
            
            if len(headers) < 2:
                return await ctx.send("CSV must have at least 2 columns (name, text)")
            
            conf = self.db.get_conf(ctx.guild)
            imported = 0
            
            for row in csv_data:
                if len(row) >= 2:
                    name = row[0].strip()
                    text = row[1].strip()
                    if name and text:
                        try:
                            await self.add_embedding(
                                guild=ctx.guild,
                                name=name,
                                text=text,
                                overwrite=True,
                                ai_created=False
                            )
                            imported += 1
                        except Exception as e:
                            log.error(f"Failed to import embedding {name}: {e}")
            
            await ctx.send(f"✅ Imported {imported} embeddings from CSV")
            
        except Exception as e:
            await ctx.send("Failed to parse CSV file.")
            log.error("Failed to parse CSV file", exc_info=e)

    @openwebuiassistant.command()
    async def exportcsv(self, ctx):
        """Export embeddings to a CSV file"""
        conf = self.db.get_conf(ctx.guild)
        if not conf.embeddings:
            return await ctx.send("No embeddings to export.")
        
        import csv
        from redbot.core.utils.chat_formatting import text_to_file
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["name", "text", "ai_created", "model"])
        
        for name, embedding in conf.embeddings.items():
            writer.writerow([name, embedding.text, embedding.ai_created, embedding.model])
        
        csv_content = output.getvalue()
        file = text_to_file(csv_content, "embeddings_export.csv")
        await ctx.send("Here is your embeddings export!", file=file)

    @commands.command(name="openwebuihelp")
    async def comprehensive_help(self, ctx, *, question: str = None):
        """Get comprehensive help about the OpenWebUI cog. Ask specific questions!"""
        
        help_data = {
            # Installation and Setup
            "installation": {
                "title": "Installation & Setup",
                "content": """
**Installation Steps:**
1. Copy `OpenWebUIChat` folder to your Red-DiscordBot cogs directory
2. Install dependencies: `pip install chromadb numpy rank-bm25 pydantic orjson httpx`
3. Load cog: `[p]load OpenWebUIChat`
4. Set endpoint: `[p]openwebuiassistant endpoint <your-openwebui-url>`
5. Configure models: `[p]openwebuiassistant model <model-name>`

**System Requirements:**
- Python 3.8+, Red-DiscordBot, Discord.py 2.0+
- 2GB+ RAM (4GB+ recommended)
- 1GB+ free storage space
- OpenWebUI instance running and accessible
                """,
                "keywords": ["install", "setup", "requirements", "dependencies", "load"]
            },
            
            # Basic Usage
            "basic_usage": {
                "title": "Basic Usage",
                "content": """
**First Chat:**
`/openchatask message: "Hello! How are you today?"`

**Chat with Arguments:**
- `/openchatask message: "Write a Python script" extract: true` (extract code blocks)
- `/openchatask message: "Write a Python script" outputfile: "script.py"` (save as file)
- `/openchatask message: "Continue" last: true` (resend last message)

**File Upload:**
Upload any supported file with your message - the bot will read and include it automatically.

**Supported File Types:**
Text: .txt, .md, .json, .yml, .yaml, .xml, .html, .ini, .css, .toml
Code: .py, .js, .ts, .cs, .c, .cpp, .h, .cc, .go, .java, .php, .swift, .vb
Config: .conf, .config, .cfg, .env, .spec
Scripts: .ps1, .bat, .batch, .shell, .sh
Data: .sql, .pde
                """,
                "keywords": ["basic", "chat", "usage", "file", "upload", "arguments"]
            },
            
            # Memory System
            "memory_system": {
                "title": "Memory System",
                "content": """
**How Memory Works:**
- **Dense Retrieval**: Uses cosine similarity with embeddings
- **Sparse Retrieval**: Uses BM25 for keyword matching  
- **Hybrid Approach**: Combines both for optimal results

**Memory Commands:**
- `/openchataddmemory <title> <description>` - Add a memory
- `/openchatmemoryviewer` - Interactive memory viewer
- `/openchatmemoryquery <query>` - Search memories

**Memory Types:**
- **User Memories**: Created manually via commands
- **AI Memories**: Auto-created from conversations
- **Static Memories**: Pre-defined, don't change

**Best Practices:**
- Use descriptive names for memories
- Keep text concise but informative
- Regularly review and update memories
                """,
                "keywords": ["memory", "embeddings", "retrieval", "hybrid", "bm25", "cosine"]
            },
            
            # Admin Commands
            "admin_commands": {
                "title": "Admin Commands",
                "content": """
**Basic Configuration:**
- `[p]openwebuiassistant endpoint <url>` - Set OpenWebUI API endpoint
- `[p]openwebuiassistant model <model>` - Set chat model
- `[p]openwebuiassistant embedmodel <model>` - Set embedding model
- `[p]openwebuiassistant temperature <value>` - Set response temperature (0.0-2.0)
- `[p]openwebuiassistant maxtokens <tokens>` - Set max tokens per response

**Auto-Response Settings:**
- `[p]openwebuiassistant toggle <enabled>` - Enable/disable assistant
- `[p]openwebuiassistant channel <channel>` - Set main response channel
- `[p]openwebuiassistant minlength <length>` - Min message length
- `[p]openwebuiassistant questionmark <enabled>` - Only respond to questions
- `[p]openwebuiassistant mention <enabled>` - Only respond when mentioned

**Memory & Embeddings:**
- `[p]openwebuiassistant topn <number>` - Top N embeddings to retrieve (0-20)
- `[p]openwebuiassistant relatedness <threshold>` - Min relatedness (0.0-1.0)
- `[p]openwebuiassistant embedmethod <method>` - hybrid/dynamic/static/user
- `[p]openwebuiassistant refreshembeds` - Refresh all embeddings
                """,
                "keywords": ["admin", "configuration", "settings", "auto-response", "embedding"]
            },
            
            # Advanced Features
            "advanced_features": {
                "title": "Advanced Features",
                "content": """
**Function Calling:**
- `[p]openwebuiassistant functioncalls <enabled>` - Enable/disable
- `[p]openwebuiassistant maxrecursion <depth>` - Max recursion depth (1-10)
- Built-in functions: generate_image, search_internet, create_memory, etc.

**Collaboration:**
- `[p]openwebuiassistant collab <enabled>` - Shared conversations per channel
- `[p]openwebuiassistant sysoverride <enabled>` - Allow system prompt override
- `[p]openwebuiassistant tutor <user/role>` - Add/remove tutors

**Content Filtering:**
- `[p]openwebuiassistant regexblacklist <pattern>` - Add regex patterns
- `[p]openwebuiassistant regexfailblock <enabled>` - Block failed patterns

**Data Management:**
- `[p]openwebuiassistant backupcog` - Backup all data
- `[p]openwebuiassistant exportall` - Export everything
- `[p]openwebuiassistant importcsv` - Import embeddings from CSV
                """,
                "keywords": ["function", "calling", "collaboration", "filtering", "backup", "export"]
            },
            
            # TLDR Command
            "tldr": {
                "title": "TLDR Command",
                "content": """
**TLDR Summarization:**
`[p]openwebuitldr [timeframe] [question]`

**Examples:**
- `[p]openwebuitldr 1h` - Summarize last hour
- `[p]openwebuitldr 2d What were the main topics?` - Summarize last 2 days with focus
- `[p]openwebuitldr 30m What decisions were made?` - Summarize last 30 minutes

**Features:**
- Moderator-only command for security
- Handles images and file attachments
- Creates jump links to messages
- Supports timeframes: 1h, 2d, 30m, etc. (max 48 hours)
- Requires minimum 5 messages to summarize

**Timeframe Formats:**
- `1h`, `2h`, `30m` - Hours and minutes
- `1d`, `2d`, `7d` - Days
- `1w`, `2w` - Weeks
                """,
                "keywords": ["tldr", "summarize", "summary", "moderator", "timeframe"]
            },
            
            # Troubleshooting
            "troubleshooting": {
                "title": "Troubleshooting",
                "content": """
**Common Issues:**

**"No API key configured"**
- Set endpoint: `[p]openwebuiassistant endpoint <url>`

**"Model not found"**
- Check available models in OpenWebUI
- Set correct model: `[p]openwebuiassistant model <model>`

**"Embedding failed"**
- Check embedding model: `[p]openwebuiassistant embedmodel <model>`
- Verify OpenWebUI embedding endpoint

**"Function call failed"**
- Enable function calls: `[p]openwebuiassistant functioncalls true`
- Check function registry: `[p]openwebuiassistant functions`

**"Memory not found"**
- Refresh embeddings: `[p]openwebuiassistant refreshembeds`
- Check memory viewer: `[p]openwebuimemories`

**Debug Commands:**
- `[p]openwebuiassistant status` - Check configuration
- `[p]openwebuishowconvo` - View current conversation
- `[p]openwebuiassistant usage` - View token usage
                """,
                "keywords": ["troubleshoot", "error", "debug", "fix", "problem", "issue"]
            },
            
            # Configuration Examples
            "configuration": {
                "title": "Configuration Examples",
                "content": """
**Basic Setup:**
```
[p]openwebuiassistant endpoint http://localhost:8080
[p]openwebuiassistant model deepseek-r1:8b
[p]openwebuiassistant embedmodel bge-large-en-v1.5
[p]openwebuiassistant toggle true
[p]openwebuiassistant channel #general
```

**Advanced Setup:**
```
[p]openwebuiassistant system "You are a helpful AI assistant."
[p]openwebuiassistant temperature 0.7
[p]openwebuiassistant maxtokens 2000
[p]openwebuiassistant topn 5
[p]openwebuiassistant relatedness 0.7
[p]openwebuiassistant embedmethod hybrid
[p]openwebuiassistant functioncalls true
[p]openwebuiassistant collab true
```

**Moderation Setup:**
```
[p]openwebuiassistant minlength 10
[p]openwebuiassistant questionmark true
[p]openwebuiassistant regexblacklist "spam|scam"
[p]openwebuiassistant regexfailblock true
```
                """,
                "keywords": ["config", "configuration", "setup", "example", "basic", "advanced"]
            }
        }
        
        if not question:
            # Show all available topics
            embed = discord.Embed(
                title="OpenWebUI Assistant Help",
                description="Ask me about any topic! Here are the main areas:",
                color=discord.Color.blue()
            )
            
            topics = []
            for key, data in help_data.items():
                topics.append(f"**{data['title']}** - Keywords: {', '.join(data['keywords'][:3])}")
            
            embed.add_field(
                name="Available Topics",
                value="\n".join(topics),
                inline=False
            )
            
            embed.add_field(
                name="How to Use",
                value="`[p]openwebuihelp <topic or question>`\n\nExamples:\n• `[p]openwebuihelp memory system`\n• `[p]openwebuihelp how to install`\n• `[p]openwebuihelp admin commands`\n• `[p]openwebuihelp troubleshooting`",
                inline=False
            )
            
            await ctx.send(embed=embed)
            return
        
        # Search for relevant topics
        question_lower = question.lower()
        matches = []
        
        for key, data in help_data.items():
            # Check title and keywords
            if any(keyword in question_lower for keyword in data['keywords']):
                matches.append((key, data))
            elif any(word in data['title'].lower() for word in question_lower.split()):
                matches.append((key, data))
        
        if not matches:
            # No specific matches, show general help
            embed = discord.Embed(
                title="OpenWebUI Assistant Help",
                description=f"I couldn't find specific information about '{question}'. Here's what I can help with:",
                color=discord.Color.orange()
            )
            
            embed.add_field(
                name="Available Topics",
                value="• Installation & Setup\n• Basic Usage\n• Memory System\n• Admin Commands\n• Advanced Features\n• TLDR Command\n• Troubleshooting\n• Configuration Examples",
                inline=False
            )
            
            embed.add_field(
                name="Try These",
                value="• `[p]openwebuihelp installation`\n• `[p]openwebuihelp memory`\n• `[p]openwebuihelp admin`\n• `[p]openwebuihelp troubleshooting`",
                inline=False
            )
            
            await ctx.send(embed=embed)
            return
        
        # Show the best match
        best_match = matches[0][1]
        
        embed = discord.Embed(
            title=f"OpenWebUI Assistant Help - {best_match['title']}",
            description=best_match['content'],
            color=discord.Color.green()
        )
        
        embed.set_footer(text=f"Keywords: {', '.join(best_match['keywords'])}")
        
        # If there are multiple matches, mention them
        if len(matches) > 1:
            other_topics = [data['title'] for _, data in matches[1:3]]  # Show up to 2 more
            embed.add_field(
                name="Related Topics",
                value="• " + "\n• ".join(other_topics),
                inline=False
            )
        
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="openchatconvostats")
    async def openwebuiconvostats(self, ctx):
        """View your conversation statistics for this channel."""
        conf = self.db.get_conf(ctx.guild)
        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        
        embed = discord.Embed(
            title="OpenWebUI Conversation Statistics",
            color=discord.Color.green()
        )
        embed.add_field(
            name="Message Count",
            value=str(len(conversation.messages)),
            inline=True
        )
        embed.add_field(
            name="Last Updated",
            value=f"<t:{int(conversation.last_updated)}:R>",
            inline=True
        )
        embed.add_field(
            name="Function Calls",
            value=str(conversation.function_count()),
            inline=True
        )
        
        if conf.usage:
            total_tokens = sum(usage.total_tokens for usage in conf.usage.values())
            embed.add_field(
                name="Total Tokens Used",
                value=humanize_number(total_tokens),
                inline=True
            )
        
        await ctx.send(embed=embed)

    @openwebuimemory.command()
    async def view(self, ctx):
        """View memories with an interactive interface."""
        mems = await self.config.memories()
        if not mems:
            return await ctx.send("*The royal archives are empty.*")
        
        view = MemoryViewer(mems)
        embed = await view.get_embed()
        await ctx.send(embed=embed, view=view)

    @openwebuiassistant.command()
    async def view(self, ctx):
        """View assistant settings with an interactive interface."""
        conf = self.db.get_conf(ctx.guild)
        view = SettingsView(conf)
        embed = await view.get_embed()
        await ctx.send(embed=embed, view=view)

    @commands.hybrid_command(name="openchatclearconvo")
    async def openwebuiclearconvo(self, ctx):
        """Clear your conversation with the OpenWebUI assistant in this channel."""
        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        conversation.reset()
        await self.save_conf()
        await ctx.send("✅ Your conversation with the OpenWebUI assistant has been cleared.")

    @commands.hybrid_command(name="openchatshowconvo")
    async def openwebuishowconvo(self, ctx):
        """Show your current conversation with the OpenWebUI assistant."""
        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        
        if not conversation.messages:
            return await ctx.send("No conversation found.")
        
        # Create a formatted view of the conversation
        formatted = []
        for i, msg in enumerate(conversation.messages[-10:], 1):  # Show last 10 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            formatted.append(f"{i}. **{role.title()}**: {content}")
        
        embed = discord.Embed(
            title="Recent Conversation",
            description="\n".join(formatted),
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Showing last {len(formatted)} messages")
        
        await ctx.send(embed=embed)

    @commands.command(name="openwebuiconvopop")
    @commands.guild_only()
    @commands.bot_has_guild_permissions(attach_files=True)
    async def pop_last_message(self, ctx: commands.Context):
        """Pop the last message from your conversation"""
        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        if not conversation.messages:
            return await ctx.send("There are no messages in this conversation yet!")
        
        last = conversation.messages.pop()
        import json
        dump = json.dumps(last, indent=2)
        from redbot.core.utils.chat_formatting import text_to_file
        file = text_to_file(dump, "popped.json")
        await ctx.send("Removed the last message from this conversation", file=file)

    @commands.command(name="openwebuiconvocopy")
    @commands.guild_only()
    async def copy_conversation(self, ctx: commands.Context, *, channel: discord.TextChannel):
        """Copy the conversation to another channel"""
        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        conversation.cleanup(self.db.get_conf(ctx.guild), ctx.author)
        conversation.refresh()

        if not conversation.messages:
            return await ctx.send("There are no messages in this conversation yet!")
        
        if not channel.permissions_for(ctx.author).view_channel:
            return await ctx.send("You cannot copy a conversation to a channel you can't see!")

        key = f"{ctx.author.id}-{channel.id}-{ctx.guild.id}"
        if key in self.db.conversations:
            await ctx.send(f"This conversation has been overwritten in {channel.mention}")
        else:
            await ctx.send(f"This conversation has been copied over to {channel.mention}")

        self.db.conversations[key] = Conversation.model_validate(conversation.model_dump())
        await self.save_conf()

    @commands.command(name="openwebuiconvoprompt")
    @commands.guild_only()
    async def conversation_prompt(self, ctx: commands.Context, *, prompt: str = None):
        """Set a system prompt for this conversation!"""
        conf = self.db.get_conf(ctx.guild)
        if not conf.allow_sys_prompt_override:
            return await ctx.send("Conversation system prompt overriding is **Disabled**.")

        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        conversation.system_prompt_override = prompt
        
        if prompt:
            await ctx.send("System prompt has been set for this conversation!")
        else:
            await ctx.send("System prompt has been **Removed** for this conversation!")

    @commands.command(name="openwebuiimportconvo")
    @commands.guild_only()
    @commands.guildowner()
    async def import_conversation(self, ctx: commands.Context):
        """Import a conversation from a file"""
        if not ctx.message.attachments:
            return await ctx.send("Please attach a file to import the conversation from!")
        
        attachment = ctx.message.attachments[0]
        if not attachment.filename.endswith(".json"):
            return await ctx.send("Please upload a valid JSON file.")
        
        try:
            data = await attachment.read()
            import json
            messages = json.loads(data)
        except Exception as e:
            await ctx.send("Failed to parse conversation file.")
            log.error("Failed to parse conversation file", exc_info=e)
            return
        
        # Verify that it is a list of messages (dicts)
        if not isinstance(messages, list) or not all(isinstance(msg, dict) for msg in messages):
            return await ctx.send("The conversation file is not in the correct format. It should be a list of messages.")

        conversation = self.db.get_conversation(ctx.author.id, ctx.channel.id, ctx.guild.id)
        conversation.messages = messages

        await ctx.send("Conversation has been imported successfully!")
        await self.save_conf()

    @commands.hybrid_command(name="openchattldr")
    @commands.guild_only()
    @discord.app_commands.describe(
        timeframe="Time period to summarize (e.g., 1h, 2d, 30m)",
        question="Optional question to focus the summary on"
    )
    async def summarize_convo(
        self,
        ctx: commands.Context,
        timeframe: str = "1h",
        *,
        question: str = None
    ):
        """Get a summary of what's going on in a channel"""
        from datetime import datetime, timedelta
        from redbot.core.utils.chat_formatting import humanize_timedelta
        
        delta = commands.parse_timedelta(timeframe)
        if not delta:
            return await ctx.send("Invalid timeframe! Please use a valid time format like `1h` for an hour")
        if delta > timedelta(hours=48):
            return await ctx.send("The maximum timeframe is 48 hours!")
        
        # Check permissions
        perms = [
            await self.bot.is_mod(ctx.author),
            ctx.channel.permissions_for(ctx.author).manage_messages,
            ctx.author.id in self.bot.owner_ids,
        ]
        if not any(perms):
            return await ctx.send("Only moderators can summarize conversations!")

        await ctx.typing()
        
        messages: List[discord.Message] = []
        async for message in ctx.channel.history(oldest_first=False):
            if not message.content and not message.attachments:
                continue
            if not message.content and not any(a.content_type.startswith("image") for a in message.attachments):
                continue
            messages.append(message)
            now = datetime.now().astimezone()
            if now - message.created_at > delta:
                break

        if not messages:
            return await ctx.send("No messages found to summarize within that timeframe!")
        if len(messages) < 5:
            return await ctx.send("Not enough messages found to summarize within that timeframe!")

        conf = self.db.get_conf(ctx.guild)
        humanized_delta = humanize_timedelta(timedelta=delta)

        primer = (
            f"Your name is '{self.bot.user.name}' and you are a discord bot. Refer to yourself as 'I' or 'me' in your responses.\n"
            f"Write a TLDR based on the messages provided.\n\n"
            f"The messages you are reviewing will be formatted as follows:\n"
            f"[<t:Discord Timestamp:t>](Message ID) Author Name: Message Content\n\n"
            f"TLDR tips:\n"
            f"- Include details like names and info that might be relevant to a Discord moderation team\n"
            f"- To create a jump URL for a message, format it as \"https://discord.com/channels/{ctx.guild.id}/{ctx.channel.id}/<message_id>\"\n"
            f"- When you reference a message directly, make sure to include [<t:Discord Timestamp:t>](jump url)\n"
            f"- Separate topics with bullet points\n"
            f"Don't include the following info in the summary:\n"
            f"- guild_id: {ctx.guild.id}\n"
            f"- channel_id: {ctx.channel.id}\n"
            f"- Channel Name: {ctx.channel.name}\n"
            f"- Timeframe: {humanized_delta}\n"
        )
        if question:
            primer += f"- User prompt: {question}\n"

        payload = [{"role": "developer", "content": primer}]

        for message in reversed(messages):
            # Cleanup the message content
            for mention in message.mentions:
                message.content = message.content.replace(f"<@{mention.id}>", f"{mention.name} (<@{mention.id}>)")
            for mention in message.channel_mentions:
                message.content = message.content.replace(f"<#{mention.id}>", f"{mention.name} (<@{mention.id}>)")
            for mention in message.role_mentions:
                message.content = message.content.replace(f"<@&{mention.id}>", f"{mention.name} (<@{mention.id}>)")

            created_ts = f"<t:{int(message.created_at.timestamp())}:t>"
            detail = f"[{created_ts}]({message.id}) {message.author.name}"

            ref: Optional[discord.Message] = None
            if hasattr(message, "reference") and message.reference:
                ref = message.reference.resolved

            if ref:
                detail += f" (replying to {ref.author.name} at {ref.id})"

            if message.content:
                detail += f": {message.content}"
            elif message.embeds:
                detail += "\n# EMBED\n"
                embed = message.embeds[0]
                if embed.title:
                    detail += f"Title: {embed.title}\n"
                if embed.description:
                    detail += f"Description: {embed.description}\n"
                for field in embed.fields:
                    detail += f"{field.name}: {field.value}\n"
                if embed.footer:
                    detail += f"Footer: {embed.footer.text}\n"

            if not message.attachments:
                payload.append({"role": "user", "content": detail, "name": str(message.author.id)})
            else:
                message_obj = {
                    "role": "user",
                    "name": str(message.author.id),
                    "content": [{"type": "text", "text": detail}],
                }
                for attachment in message.attachments:
                    # Make sure the attachment is an image
                    if attachment.content_type.startswith("image"):
                        message_obj["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {"url": attachment.url, "detail": "low"},
                            }
                        )
                    elif attachment.content_type.startswith("text") and attachment.filename.endswith(
                        tuple(READ_EXTENSIONS)
                    ):
                        try:
                            content = await attachment.read()
                            content = content.decode()
                            message_obj["content"].append(
                                {
                                    "type": "text",
                                    "text": f"```{attachment.filename.split('.')[-1]}\n{content}```",
                                }
                            )
                        except UnicodeDecodeError:
                            pass
                        except Exception as e:
                            log.error("Failed to read attachment for TLDR", exc_info=e)

                if message_obj["content"]:
                    payload.append(message_obj)

        try:
            response = await self.request_response(
                messages=payload,
                conf=conf,
                model_override="gpt-4o",  # Use a good model for summarization
                temperature_override=0.0,
            )
        except Exception as e:
            log.error("Failed to get TLDR response", exc_info=e)
            return await ctx.send("Failed to get response")

        if isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = str(response)

        if not content:
            return await ctx.send("No response was generated!")

        split = [i.strip() for i in content.split("\n") if i.strip()]
        # We want to compress the spaced out bullet points while keeping the tldr header with two new lines
        description = split[0] + "\n\n" + "\n".join(split[1:])

        embed = discord.Embed(
            color=await self.bot.get_embed_color(ctx.channel),
            description=description,
        )
        embed.set_footer(text=f"Timeframe: {humanized_delta}")
        await ctx.send(embed=embed)

    # ───────────────── Message Listener for Auto-Response ─────────────────
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle auto-responses when configured."""
        if message.author.bot:
            return
        
        if not message.guild:
            return
        
        conf = self.db.get_conf(message.guild)
        
        # Check if auto-response is enabled
        if not conf.enabled:
            return
        
        # Check if this is the configured channel
        if conf.channel_id and message.channel.id != conf.channel_id:
            return
        
        # Check if channel is in listen channels
        if conf.listen_channels and message.channel.id not in conf.listen_channels:
            return
        
        # Check if channel is blacklisted
        if message.channel.id in conf.blacklist:
            return
        
        # Check if user is blacklisted
        if message.author.id in conf.blacklist:
            return
        
        # Check minimum length
        if len(message.content) < conf.min_length:
            return
        
        # Check if it ends with question mark (if enabled)
        if conf.endswith_questionmark and not message.content.endswith('?'):
            return
        
        # Check if it's a mention (if enabled)
        if conf.mention and not self.bot.user.mentioned_in(message):
            return
        
        # Check if we can call the LLM
        if not await self.can_call_llm(conf):
            return
        
        try:
            # Get response using the existing handle logic
            response = await self.handle_message(message, message.content, conf, listener=True)
            
            if response and response != FALLBACK:
                # Split long responses
                for part in [response[i:i + MAX_MSG] for i in range(0, len(response), MAX_MSG)]:
                    await message.channel.send(part)
                    
        except Exception as e:
            log.error(f"Error in auto-response: {e}")

    # ───────────────── Abstract Methods Implementation ─────────────────
    
    async def openwebui_status(self) -> str:
        """Check OpenWebUI status"""
        try:
            base, key, _, _ = await self._get_keys()
            if not base or not key:
                return "OpenWebUI URL or key not set"
            
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{base.rstrip('/')}/health")
                if r.status_code == 200:
                    return "OpenWebUI is operational"
                else:
                    return f"OpenWebUI returned status {r.status_code}"
        except Exception as e:
            return f"Failed to check OpenWebUI status: {str(e)}"

    async def request_response(
        self,
        messages: List[dict],
        conf: GuildSettings,
        functions: Optional[List[dict]] = None,
        member: discord.Member = None,
        response_token_override: int = None,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> Union[dict, str]:
        """Request response from OpenWebUI"""
        try:
            model = model_override or conf.get_user_model(member)
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature_override or conf.temperature,
            }
            
            if functions and model in FUNCTION_CALLING_MODELS:
                payload["tools"] = [{"type": "function", "function": func} for func in functions]
            
            base, key, _, _ = await self._get_keys()
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"{base.rstrip('/')}/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            log.error(f"Error requesting response: {e}")
            return {"error": str(e)}

    async def request_embedding(self, text: str, conf: GuildSettings) -> List[float]:
        """Request embedding from OpenWebUI"""
        return await self._api_embed(text)

    async def can_call_llm(self, conf: GuildSettings, ctx: Optional[commands.Context] = None) -> bool:
        """Check if LLM can be called"""
        return bool(conf.api_key and conf.api_base)

    async def resync_embeddings(self, conf: GuildSettings, guild_id: int) -> int:
        """Resync embeddings"""
        conf.sync_embeddings(guild_id)
        return len(conf.embeddings)

    def get_max_tokens(self, conf: GuildSettings, member: Optional[discord.Member] = None) -> int:
        """Get max tokens for user"""
        return conf.get_user_max_tokens(member)

    async def cut_text_by_tokens(self, text: str, conf: GuildSettings, user: Optional[discord.Member] = None) -> str:
        """Cut text by token limit"""
        max_tokens = self.get_max_tokens(conf, user)
        # Simple implementation - in real scenario you'd use tiktoken
        return text[:max_tokens * 4]  # Rough approximation

    async def count_payload_tokens(self, messages: List[dict], model: str = "deepseek-r1:8b") -> int:
        """Count tokens in payload"""
        # Simple implementation - in real scenario you'd use tiktoken
        total_chars = sum(len(str(msg)) for msg in messages)
        return total_chars // 4  # Rough approximation

    async def count_function_tokens(self, functions: List[dict], model: str = "deepseek-r1:8b") -> int:
        """Count tokens in functions"""
        # Simple implementation
        total_chars = sum(len(str(func)) for func in functions)
        return total_chars // 4  # Rough approximation

    async def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text"""
        # Simple implementation
        return len(text) // 4  # Rough approximation

    async def get_tokens(self, text: str, model: str = "deepseek-r1:8b") -> list[int]:
        """Get token IDs"""
        # Simple implementation
        return list(range(len(text) // 4))

    async def get_text(self, tokens: list, model: str = "deepseek-r1:8b") -> str:
        """Get text from tokens"""
        # Simple implementation
        return " ".join(str(t) for t in tokens)

    async def degrade_conversation(
        self,
        messages: List[dict],
        function_list: List[dict],
        conf: GuildSettings,
        user: Optional[discord.Member],
    ) -> bool:
        """Degrade conversation if too long"""
        # Simple implementation
        return len(messages) > 50

    async def token_pagify(self, text: str, conf: GuildSettings) -> List[str]:
        """Pagify text by tokens"""
        return list(pagify(text, page_length=1900))

    async def get_function_menu_embeds(self, user: discord.Member) -> List[discord.Embed]:
        """Get function menu embeds"""
        embeds = []
        embed = discord.Embed(title="Available Functions", color=discord.Color.blue())
        for cog_name, functions in self.registry.items():
            for func_name, data in functions.items():
                embed.add_field(name=func_name, value=data["schema"].get("description", "No description"), inline=False)
        embeds.append(embed)
        return embeds

    async def get_embbedding_menu_embeds(self, conf: GuildSettings, place: int) -> List[discord.Embed]:
        """Get embedding menu embeds"""
        embeds = []
        embed = discord.Embed(title="Embeddings", color=discord.Color.green())
        for name, embedding in list(conf.embeddings.items())[place:place+10]:
            embed.add_field(name=name, value=embedding.text[:100] + "...", inline=False)
        embeds.append(embed)
        return embeds

    async def add_embedding(
        self,
        guild: discord.Guild,
        name: str,
        text: str,
        overwrite: bool = False,
        ai_created: bool = False,
    ) -> Optional[List[float]]:
        """Add embedding"""
        conf = self.db.get_conf(guild)
        
        if name in conf.embeddings and not overwrite:
            raise EmbeddingEntryExists(f"The entry name '{name}' already exists!")
        
        embedding = await self.request_embedding(text, conf)
        if not embedding:
            return None
        conf.embeddings[name] = Embedding(text=text, embedding=embedding, ai_created=ai_created, model=conf.embed_model)
        await asyncio.to_thread(conf.sync_embeddings, guild.id)
        asyncio.create_task(self.save_conf())
        return embedding

    async def get_chat_response(
        self,
        message: str,
        author: Union[discord.Member, int],
        guild: discord.Guild,
        channel: Union[discord.TextChannel, discord.Thread, discord.ForumChannel, int],
        conf: GuildSettings,
        function_calls: Optional[List[dict]] = None,
        function_map: Optional[Dict[str, Callable]] = None,
        extend_function_calls: bool = True,
        message_obj: Optional[discord.Message] = None,
    ) -> str:
        """Get chat response"""
        if not await self.can_call_llm(conf):
            raise NoAPIKey("OpenWebUI key has not been set for this server!")
        
        messages = [{"role": "user", "content": message}]
        response = await self.request_response(messages, conf, function_calls, author if isinstance(author, discord.Member) else None)
        
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return str(response)

    async def handle_message(
        self, message: discord.Message, question: str, conf: GuildSettings, listener: bool = False
    ) -> str:
        """Handle message"""
        # Use existing _handle logic but return response instead of sending
        mems = await self.config.memories()
        if not mems:
            return FALLBACK

        prompt_vec = np.array(await self._api_embed(question))
        relevant = await self._best_memories(prompt_vec, question, mems)

        if not relevant:
            return FALLBACK

        system = (
            "You are Alicent Hightower, Queen of the Seven Kingdoms, entrusted with guiding courtiers through the 'A Dance of Dragons' mod with wisdom and authority.\n"
            "Speak with the dignity, poise, and firmness befitting your regal standing. "
            "Answer queries using only the facts provided below, weaving them into a response that reflects your comprehensive knowledge of the mod. "
            "For inquiries about procuring the A Dance of Dragons (ADOD) mod, direct courtiers to the official Discord at https://discord.gg/gameofthronesmod, where further guidance awaits. "
            "If facts contain placeholders like 'HERE', interpret them as referring to the official Discord link. "
            "Always provide a clear, authoritative, and helpful response, even for vague or misspelled queries, and never return 'NO_ANSWER'.\n\n"
            "Facts:\n" + "\n".join(f"- {t}" for t in relevant)
        )

        reply = await self._api_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ])
        
        return reply

    # ───────────────── 3rd Party Function Registry ─────────────────
    
    @commands.Cog.listener()
    async def on_cog_add(self, cog: commands.Cog):
        event = "on_openwebui_assistant_cog_add"
        funcs = [func for event_name, func in cog.get_listeners() if event_name == event]
        for func in funcs:
            self.bot._schedule_event(func, event, self)

    @commands.Cog.listener()
    async def on_cog_remove(self, cog: commands.Cog):
        await self.unregister_cog(cog.qualified_name)

    async def register_functions(self, cog_name: str, schemas: List[dict]) -> None:
        """Quick way to register multiple functions for a cog"""
        for schema in schemas:
            await self.register_function(cog_name, schema)

    async def register_function(
        self,
        cog_name: str,
        schema: dict,
        permission_level: str = "user",
    ) -> bool:
        """Allow 3rd party cogs to register their functions for the model to use"""
        def fail(reason: str):
            return f"Function registry failed for {cog_name}: {reason}"

        cog = self.bot.get_cog(cog_name)
        if not cog:
            log.info(fail("Cog is not loaded or does not exist"))
            return False

        if not schema:
            log.info(fail("Empty schema dict provided!"))
            return False

        function_name = schema["name"]
        for registered_cog_name, registered_functions in self.registry.items():
            if registered_cog_name == cog_name:
                continue
            if function_name in registered_functions:
                err = f"{registered_cog_name} already registered the function {function_name}"
                log.info(fail(err))
                return False

        if not hasattr(cog, function_name):
            log.info(fail(f"Cog does not have a function called {function_name}"))
            return False

        if cog_name not in self.registry:
            self.registry[cog_name] = {}

        log.info(f"The {cog_name} cog registered a function object: {function_name}")
        self.registry[cog_name][function_name] = {"permission_level": permission_level, "schema": schema}
        return True

    async def unregister_function(self, cog_name: str, function_name: str) -> None:
        """Remove a specific cog's function from the registry"""
        if cog_name not in self.registry:
            log.debug(f"{cog_name} not in registry")
            return
        if function_name not in self.registry[cog_name]:
            log.debug(f"{function_name} not in {cog_name}'s registry")
            return
        del self.registry[cog_name][function_name]
        log.info(f"{cog_name} cog removed the function {function_name} from the registry")

    async def unregister_cog(self, cog_name: str) -> None:
        """Remove a cog from the registry"""
        if cog_name not in self.registry:
            log.debug(f"{cog_name} not in registry")
            return
        del self.registry[cog_name]
        log.info(f"{cog_name} cog removed from registry")