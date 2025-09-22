import asyncio
import contextlib
import hashlib
import logging
import io
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set
import re
import numpy as np
import discord
import httpx
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.i18n import Translator, cog_i18n
from redbot.core.utils.chat_formatting import pagify, box, humanize_number
from rank_bm25 import BM25Okapi
import aiofiles
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
import sqlite3
import threading
from pathlib import Path
from redbot.core.data_manager import cog_data_path

# Helpers for serialization
def _to_jsonable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_jsonable(v) for v in obj)
    return obj

log = logging.getLogger("red.OpenWebUIChat")
_ = Translator("OpenWebUIChat", __file__)

MAX_MSG = 1900
FALLBACK = "I'm here to help! How can I assist you today?"
SIM_THRESHOLD = 0.8  # Cosine similarity gate (0-1), matching ChatGPT
TOP_K = 9  # Max memories sent to the LLM, matching ChatGPT
EMBED_CHUNK = 3900  # Safe embed description chunk size

# Enhanced constants
MAX_CONVERSATION_TURNS = 50
MEMORY_TTL_HOURS = 24
AUTO_SUMMARIZE_THRESHOLD = 20
MAX_TOKENS_PER_REQUEST = 4000
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 10

class PersonaType(Enum):
    ALICENT = "alicent"
    COURT_SCRIBE = "court_scribe"
    BATTLE_SAGE = "battle_sage"
    CODE_MAESTER = "code_maester"
    HELPER = "helper"
    MOD_ASSISTANT = "mod_assistant"

class EmbeddingProvider(Enum):
    OPENWEBUI = "openwebui"
    OLLAMA = "ollama"
    FAISS = "faiss"
    CHROMA = "chroma"
    PGVECTOR = "pgvector"

class MemoryScope(Enum):
    GLOBAL = "global"
    GUILD = "guild"
    CHANNEL = "channel"
    USER = "user"

class ResponseMode(Enum):
    NORMAL = "normal"
    SHOW_WORK = "show_work"
    SNIPPET = "snippet"
    THREAD = "thread"

class ToolCategory(Enum):
    WEB_SEARCH = "web_search"
    CODE = "code"
    IMAGE = "image"
    AUDIO = "audio"
    MATH = "math"
    CALENDAR = "calendar"
    FILE = "file"
    DOCKER = "docker"
    MODERATION = "moderation"

@dataclass
class UserMemoryProfile:
    user_id: int
    short_term_memory: List["ConversationTurn"]  # Recent conversation turns
    long_term_profile: Dict[str, Any]  # Persistent user preferences and facts
    task_log: List[Dict[str, Any]]  # Active and completed tasks
    preferences: Dict[str, Any]  # User settings and preferences
    created_at: datetime
    updated_at: datetime
    ttl_expires: Optional[datetime] = None

@dataclass
class GuildKnowledgeBase:
    guild_id: int
    name: str
    scope: MemoryScope
    documents: Dict[str, Dict[str, Any]]  # doc_id -> {text, metadata, embeddings}
    faqs: Dict[str, str]  # question -> answer
    house_rules: List[str]
    project_docs: Dict[str, str]  # project_name -> description
    created_at: datetime
    updated_at: datetime

@dataclass
class MemoryEntry:
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    scope: MemoryScope
    created_at: datetime
    ttl_expires: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class ToolDefinition:
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    allowed_guilds: Set[int]
    allowed_channels: Set[int]
    requires_auth: bool = False
    rate_limit: int = 10  # per minute

@dataclass
class ConversationTurn:
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tokens_used: int = 0

@dataclass
class RateLimitInfo:
    user_id: int
    requests: List[datetime]
    last_reset: datetime
    burst_allowance: int = 5

@cog_i18n(_)
class OpenWebUIMemoryBot(commands.Cog):
    """An advanced AI assistant with comprehensive memory, knowledge management, and tool integration."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.q: "asyncio.Queue[Tuple[commands.Context, str]]" = asyncio.Queue()
        self.worker: Optional[asyncio.Task] = None
        self.config = Config.get_conf(self, 0xBADA55, force_registration=True)
        
        # Enhanced global configuration
        self.config.register_global(
            api_base="",
            api_key="",
            chat_model="deepseek-r1:8b",
            embed_model="bge-large-en-v1.5",
            embedding_provider=EmbeddingProvider.OPENWEBUI.value,
            memories={},  # Legacy format for backward compatibility
            user_profiles={},  # user_id -> UserMemoryProfile
            guild_knowledge_bases={},  # guild_id -> GuildKnowledgeBase
            memory_entries={},  # memory_id -> MemoryEntry
            tools={},  # tool_name -> ToolDefinition
            rate_limits={},  # user_id -> RateLimitInfo
            model_presets={},  # preset_name -> model_config
            persona_templates={},  # persona_name -> template
            prompt_templates={},  # channel_id -> template
            system_health={},  # Health monitoring data
            backup_config={},  # Backup and restore settings
        )
        
        # Enhanced per-guild settings
        self.config.register_guild(
            auto_channels=[],  # List[int]
            mention_only=False,
            persona_type=PersonaType.ALICENT.value,
            response_mode=ResponseMode.NORMAL.value,
            allowed_tools=[],  # List[str] - tool names
            blocked_users=[],  # List[int] - user IDs
            blocked_roles=[],  # List[int] - role IDs
            policy_preset="default",  # SFW-only, strict, roleplay, etc.
            quotas={},  # Usage quotas and limits
            feature_flags={},  # Per-role feature access
            knowledge_base_enabled=True,
            moderation_enabled=False,
            web_search_enabled=True,
            file_upload_enabled=True,
            thread_first_mode=False,
            ephemeral_responses=False,
            language_detection=True,
            auto_translate=False,
            reply_language="auto",
        )
        
        # Enhanced per-channel settings
        self.config.register_channel(
            history_enabled=True,
            history_max_turns=20,
            history=[],  # list[{role, content}]
            prompt_template="",  # Custom prompt template
            response_mode=ResponseMode.NORMAL.value,
            allowed_tools=[],  # Channel-specific tool allowlist
            auto_thread=False,  # Auto-create threads for responses
            snippet_mode=False,  # Auto-detect and handle code snippets
            show_sources=False,  # Show knowledge base sources
            rate_limit_override=None,  # Override global rate limits
        )
        
        # Enhanced per-user settings
        self.config.register_user(
            preferences={},  # User preferences
            session_controls={},  # Current session settings
            language_preference="auto",
            persona_preference=PersonaType.ALICENT.value,
            response_length="normal",  # short, normal, detailed, ultra
            creativity_level=0.7,  # 0.0 to 1.0
            verbosity_level=0.5,  # 0.0 to 1.0
            tools_enabled=True,
            consent_given=False,  # Privacy consent
            data_retention_days=30,
        )
        
        # Runtime data structures
        self.user_memory_profiles: Dict[int, UserMemoryProfile] = {}
        self.guild_knowledge_bases: Dict[int, GuildKnowledgeBase] = {}
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.tools: Dict[str, ToolDefinition] = {}
        self.rate_limits: Dict[int, RateLimitInfo] = {}
        self.active_conversations: Dict[str, List[ConversationTurn]] = {}
        self.embedding_cache: Dict[str, List[float]] = {}
        self.response_cache: Dict[str, str] = {}
        self.health_status = {"last_check": None, "status": "unknown", "details": {}}
        
        # Background tasks
        self.memory_cleanup_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.backup_task: Optional[asyncio.Task] = None
        self.analytics_task: Optional[asyncio.Task] = None
        
        # Database connections
        self.db_path = None  # Will be set in cog_load
        self.db_lock = threading.Lock()

    # ───────────────── lifecycle ─────────────────
    async def cog_load(self):
        """Initialize the enhanced AI assistant with all background tasks."""
        # Set database path now that cog is loaded
        self.db_path = cog_data_path(self) / "memories.db"
        await self._initialize_database()
        await self._load_persistent_data()
        await self._initialize_tools()
        await self._setup_persona_templates()
        
        # Start background tasks
        self.worker = asyncio.create_task(self._worker())
        self.memory_cleanup_task = asyncio.create_task(self._memory_cleanup_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.backup_task = asyncio.create_task(self._backup_loop())
        self.analytics_task = asyncio.create_task(self._analytics_loop())
        
        log.info("Enhanced OpenWebUIMemoryBot is ready with advanced features.")

    async def cog_unload(self):
        """Clean shutdown of all tasks and data persistence."""
        tasks_to_cancel = [
            self.worker, self.memory_cleanup_task, 
            self.health_check_task, self.backup_task, self.analytics_task
        ]
        
        for task in tasks_to_cancel:
            if task:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                    await task
        
        await self._persist_data()
        log.info("OpenWebUIMemoryBot shutdown complete.")

    async def _initialize_database(self):
        """Initialize SQLite database for persistent storage."""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for enhanced data structures
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    short_term_memory TEXT,
                    long_term_profile TEXT,
                    task_log TEXT,
                    preferences TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    ttl_expires TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS guild_knowledge_bases (
                    guild_id INTEGER PRIMARY KEY,
                    name TEXT,
                    scope TEXT,
                    documents TEXT,
                    faqs TEXT,
                    house_rules TEXT,
                    project_docs TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    scope TEXT,
                    created_at TEXT,
                    ttl_expires TEXT,
                    access_count INTEGER,
                    last_accessed TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    tokens_used INTEGER
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER PRIMARY KEY,
                    requests TEXT,
                    burst_allowance INTEGER,
                    last_reset TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archived_memories (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    scope TEXT,
                    created_at TEXT,
                    archived_at TEXT
                )
            """)
            
            conn.commit()
            conn.close()

    async def _load_persistent_data(self):
        """Load all persistent data from database and config."""
        # Load user profiles
        user_profiles_data = await self.config.user_profiles()
        for user_id_str, profile_data in user_profiles_data.items():
            user_id = int(user_id_str)
            # Rehydrate datetimes and conversation turns
            stm = []
            for t in profile_data.get("short_term_memory", []) or []:
                if isinstance(t, dict):
                    stm.append(ConversationTurn(
                        role=t.get("role", "user"),
                        content=t.get("content", ""),
                        timestamp=datetime.fromisoformat(t.get("timestamp")) if isinstance(t.get("timestamp"), str) else (t.get("timestamp") or datetime.now()),
                        metadata=t.get("metadata", {}),
                        tokens_used=int(t.get("tokens_used", 0)),
                    ))
            self.user_memory_profiles[user_id] = UserMemoryProfile(
                user_id=user_id,
                short_term_memory=stm,
                long_term_profile=profile_data.get("long_term_profile", {}),
                task_log=profile_data.get("task_log", []),
                preferences=profile_data.get("preferences", {}),
                created_at=datetime.fromisoformat(profile_data.get("created_at")) if isinstance(profile_data.get("created_at"), str) else datetime.now(),
                updated_at=datetime.fromisoformat(profile_data.get("updated_at")) if isinstance(profile_data.get("updated_at"), str) else datetime.now(),
                ttl_expires=datetime.fromisoformat(profile_data.get("ttl_expires")) if isinstance(profile_data.get("ttl_expires"), str) else None,
            )
        
        # Load guild knowledge bases
        guild_kb_data = await self.config.guild_knowledge_bases()
        for guild_id_str, kb_data in guild_kb_data.items():
            guild_id = int(guild_id_str)
            self.guild_knowledge_bases[guild_id] = GuildKnowledgeBase(**kb_data)
        
        # Load memory entries
        memory_entries_data = await self.config.memory_entries()
        for memory_id, entry_data in memory_entries_data.items():
            self.memory_entries[memory_id] = MemoryEntry(
                id=memory_id,
                text=entry_data.get("text", ""),
                embedding=entry_data.get("embedding", []),
                metadata=entry_data.get("metadata", {}),
                scope=MemoryScope(entry_data.get("scope", MemoryScope.GLOBAL.value)) if not isinstance(entry_data.get("scope"), MemoryScope) else entry_data.get("scope"),
                created_at=datetime.fromisoformat(entry_data.get("created_at")) if isinstance(entry_data.get("created_at"), str) else datetime.now(),
                ttl_expires=datetime.fromisoformat(entry_data.get("ttl_expires")) if isinstance(entry_data.get("ttl_expires"), str) else None,
                access_count=int(entry_data.get("access_count", 0)),
                last_accessed=datetime.fromisoformat(entry_data.get("last_accessed")) if isinstance(entry_data.get("last_accessed"), str) else None,
            )
        
        # Load tools
        tools_data = await self.config.tools()
        for tool_name, tool_data in tools_data.items():
            self.tools[tool_name] = ToolDefinition(
                name=tool_data.get("name", tool_name),
                description=tool_data.get("description", ""),
                category=ToolCategory(tool_data.get("category", ToolCategory.WEB_SEARCH.value)) if not isinstance(tool_data.get("category"), ToolCategory) else tool_data.get("category"),
                parameters=tool_data.get("parameters", {}),
                allowed_guilds=set(tool_data.get("allowed_guilds", [])),
                allowed_channels=set(tool_data.get("allowed_channels", [])),
                requires_auth=bool(tool_data.get("requires_auth", False)),
                rate_limit=int(tool_data.get("rate_limit", 10)),
            )
        
        # Load rate limits
        rate_limits_data = await self.config.rate_limits()
        for user_id_str, rate_data in rate_limits_data.items():
            user_id = int(user_id_str)
            self.rate_limits[user_id] = RateLimitInfo(
                user_id=user_id,
                requests=[datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.now() for ts in (rate_data.get("requests") or [])],
                burst_allowance=int(rate_data.get("burst_allowance", 5)),
                last_reset=datetime.fromisoformat(rate_data.get("last_reset")) if isinstance(rate_data.get("last_reset"), str) else datetime.now(),
            )

    async def _persist_data(self):
        """Save all runtime data to persistent storage."""
        # Save user profiles
        user_profiles_data = {}
        for user_id, profile in self.user_memory_profiles.items():
            data = asdict(profile)
            data["short_term_memory"] = [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                    "metadata": t.metadata,
                    "tokens_used": t.tokens_used,
                } for t in profile.short_term_memory
            ]
            data["created_at"] = profile.created_at.isoformat()
            data["updated_at"] = profile.updated_at.isoformat()
            data["ttl_expires"] = profile.ttl_expires.isoformat() if profile.ttl_expires else None
            user_profiles_data[str(user_id)] = data
        await self.config.user_profiles.set(user_profiles_data)
        
        # Save guild knowledge bases
        guild_kb_data = {
            str(guild_id): asdict(kb) 
            for guild_id, kb in self.guild_knowledge_bases.items()
        }
        await self.config.guild_knowledge_bases.set(guild_kb_data)
        
        # Save memory entries
        memory_entries_data = {}
        for memory_id, entry in self.memory_entries.items():
            data = asdict(entry)
            data["created_at"] = entry.created_at.isoformat()
            data["ttl_expires"] = entry.ttl_expires.isoformat() if entry.ttl_expires else None
            data["last_accessed"] = entry.last_accessed.isoformat() if entry.last_accessed else None
            data["scope"] = entry.scope.value if isinstance(entry.scope, MemoryScope) else str(entry.scope)
            memory_entries_data[memory_id] = data
        await self.config.memory_entries.set(memory_entries_data)
        
        # Save tools
        tools_data = {}
        for tool_name, tool in self.tools.items():
            data = asdict(tool)
            data["category"] = tool.category.value if isinstance(tool.category, ToolCategory) else str(tool.category)
            data["allowed_guilds"] = list(tool.allowed_guilds)
            data["allowed_channels"] = list(tool.allowed_channels)
            tools_data[tool_name] = data
        await self.config.tools.set(tools_data)
        
        # Save rate limits
        rate_limits_data = {}
        for user_id, rate_info in self.rate_limits.items():
            data = asdict(rate_info)
            data["requests"] = [ts.isoformat() for ts in rate_info.requests]
            data["last_reset"] = rate_info.last_reset.isoformat()
            rate_limits_data[str(user_id)] = data
        await self.config.rate_limits.set(rate_limits_data)

    async def _initialize_tools(self):
        """Initialize the default tool set."""
        default_tools = [
            ToolDefinition(
                name="web_search",
                description="Search the web for current information",
                category=ToolCategory.WEB_SEARCH,
                parameters={"query": "string", "max_results": "integer"},
                allowed_guilds=set(),
                allowed_channels=set(),
                rate_limit=20
            ),
            ToolDefinition(
                name="code_lint",
                description="Lint and analyze code for issues",
                category=ToolCategory.CODE,
                parameters={"code": "string", "language": "string"},
                allowed_guilds=set(),
                allowed_channels=set(),
                rate_limit=30
            ),
            ToolDefinition(
                name="image_generate",
                description="Generate images from text prompts",
                category=ToolCategory.IMAGE,
                parameters={"prompt": "string", "style": "string"},
                allowed_guilds=set(),
                allowed_channels=set(),
                rate_limit=5
            ),
            ToolDefinition(
                name="math_solve",
                description="Solve mathematical problems and equations",
                category=ToolCategory.MATH,
                parameters={"expression": "string"},
                allowed_guilds=set(),
                allowed_channels=set(),
                rate_limit=50
            ),
            ToolDefinition(
                name="file_analyze",
                description="Analyze uploaded files and extract information",
                category=ToolCategory.FILE,
                parameters={"file_content": "string", "file_type": "string"},
                allowed_guilds=set(),
                allowed_channels=set(),
                rate_limit=10
            ),
        ]
        
        for tool in default_tools:
            if tool.name not in self.tools:
                self.tools[tool.name] = tool

    async def _setup_persona_templates(self):
        """Setup default persona templates."""
        persona_templates = {
            PersonaType.ALICENT.value: {
                "name": "Alicent",
                "description": "A wise and helpful AI assistant",
                "system_prompt": "You are Alicent, a helpful AI assistant. Be concise, friendly, factual, and helpful. Do not include hidden reasoning or chain-of-thought content. Answer directly.",
                "response_style": "friendly and informative"
            },
            PersonaType.COURT_SCRIBE.value: {
                "name": "Court Scribe",
                "description": "A formal and precise assistant for official matters",
                "system_prompt": "You are the Court Scribe, a formal and precise AI assistant. Maintain a professional tone and provide detailed, accurate information.",
                "response_style": "formal and detailed"
            },
            PersonaType.BATTLE_SAGE.value: {
                "name": "Battle Sage",
                "description": "A strategic and analytical assistant for complex problems",
                "system_prompt": "You are the Battle Sage, a strategic AI assistant. Analyze problems thoroughly and provide tactical solutions with clear reasoning.",
                "response_style": "analytical and strategic"
            },
            PersonaType.CODE_MAESTER.value: {
                "name": "Code Maester",
                "description": "A technical expert focused on programming and development",
                "system_prompt": "You are the Code Maester, a technical AI assistant specializing in programming. Provide clear, well-commented code examples and technical explanations.",
                "response_style": "technical and precise"
            },
            PersonaType.HELPER.value: {
                "name": "Helper",
                "description": "A general-purpose assistant for everyday tasks",
                "system_prompt": "You are a helpful AI assistant. Provide clear, practical answers and step-by-step guidance for various tasks.",
                "response_style": "practical and clear"
            },
            PersonaType.MOD_ASSISTANT.value: {
                "name": "Mod Assistant",
                "description": "A moderation-focused assistant for server management",
                "system_prompt": "You are a Mod Assistant, specialized in Discord server moderation. Help with rule enforcement, user management, and community guidelines.",
                "response_style": "authoritative and fair"
            }
        }
        
        await self.config.persona_templates.set(persona_templates)

    # ───────────────── background tasks ───────────
    async def _memory_cleanup_loop(self):
        """Background task to clean up expired memories and perform maintenance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired_memories()
                await self._decay_memory_importance()
                await self._auto_summarize_long_conversations()
                await self._compact_memory_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in memory cleanup loop: {e}")

    async def _health_check_loop(self):
        """Background task to monitor system health."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._check_system_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in health check loop: {e}")

    async def _backup_loop(self):
        """Background task to perform regular backups."""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                await self._perform_backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in backup loop: {e}")

    async def _analytics_loop(self):
        """Background task to collect and process analytics."""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                await self._collect_analytics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in analytics loop: {e}")

    async def _cleanup_expired_memories(self):
        """Remove expired memories and update TTLs."""
        current_time = datetime.now()
        expired_memories = []
        archived_memories = []
        
        for memory_id, entry in self.memory_entries.items():
            if entry.ttl_expires and entry.ttl_expires < current_time:
                # Check if this is a valuable memory that should be archived
                if entry.access_count > 5 or entry.metadata.get("type") in ["faq", "house_rule", "project_doc"]:
                    # Archive to long-term storage instead of deleting
                    await self._archive_memory(entry)
                    archived_memories.append(memory_id)
                else:
                    expired_memories.append(memory_id)
        
        # Remove expired memories
        for memory_id in expired_memories:
            del self.memory_entries[memory_id]
        
        # Remove archived memories from active storage
        for memory_id in archived_memories:
            del self.memory_entries[memory_id]
        
        if expired_memories or archived_memories:
            log.info(f"Cleaned up {len(expired_memories)} expired memories and archived {len(archived_memories)} valuable memories")

    async def _archive_memory(self, entry: MemoryEntry):
        """Archive a memory to long-term storage."""
        try:
            # Create a summary of the memory for long-term storage
            summary_prompt = f"Summarize this memory for long-term storage: {entry.text}"
            summary = await self._api_chat([{"role": "user", "content": summary_prompt}])
            
            # Create archived memory entry
            archived_entry = MemoryEntry(
                id=f"archived_{entry.id}",
                text=summary,
                embedding=entry.embedding,
                metadata={
                    **entry.metadata,
                    "archived_at": datetime.now().isoformat(),
                    "original_text": entry.text,
                    "access_count": entry.access_count,
                    "original_created": entry.created_at.isoformat()
                },
                scope=entry.scope,
                created_at=entry.created_at,
                ttl_expires=None,  # Archived memories don't expire
                access_count=0,  # Reset access count
                last_accessed=None
            )
            
            # Store in database for long-term persistence
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO archived_memories 
                    (id, text, embedding, metadata, scope, created_at, archived_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    archived_entry.id,
                    archived_entry.text,
                    json.dumps(archived_entry.embedding),
                    json.dumps(archived_entry.metadata),
                    archived_entry.scope.value,
                    archived_entry.created_at.isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
                conn.close()
            
            log.info(f"Archived memory {entry.id} to long-term storage")
            
        except Exception as e:
            log.error(f"Failed to archive memory {entry.id}: {e}")

    async def _decay_memory_importance(self):
        """Apply decay to memory importance based on access patterns."""
        current_time = datetime.now()
        decay_threshold = current_time - timedelta(days=7)  # Decay memories older than 7 days
        
        for memory_id, entry in self.memory_entries.items():
            if entry.last_accessed and entry.last_accessed < decay_threshold:
                # Reduce TTL for rarely accessed memories
                if entry.ttl_expires:
                    # Reduce TTL by 25% for each week of inactivity
                    days_since_access = (current_time - entry.last_accessed).days
                    decay_factor = 0.75 ** (days_since_access // 7)
                    new_ttl = entry.created_at + timedelta(
                        hours=MEMORY_TTL_HOURS * decay_factor
                    )
                    entry.ttl_expires = new_ttl
                    
                    # If TTL is very short, mark for early cleanup
                    if decay_factor < 0.1:
                        entry.metadata["decay_flag"] = True

    async def _auto_summarize_long_conversations(self):
        """Auto-summarize conversations that exceed the threshold."""
        for conversation_id, turns in self.active_conversations.items():
            if len(turns) > AUTO_SUMMARIZE_THRESHOLD:
                await self._summarize_conversation(conversation_id, turns)

    async def _summarize_conversation(self, conversation_id: str, turns: List[ConversationTurn]):
        """Summarize a long conversation into a concise memory note.
        Replaces older turns with a short summary to keep context lean.
        """
        try:
            # Build a brief summary request using the last N turns
            recent = turns[-(AUTO_SUMMARIZE_THRESHOLD * 2):]
            messages = [{"role": "system", "content": "Summarize the following conversation into 5-8 bullet points capturing key facts, decisions, and action items. Do not include chain-of-thought."}]
            for t in recent:
                if isinstance(t, dict):
                    role = t.get("role", "user")
                    content = t.get("content", "")
                else:
                    role = t.role
                    content = t.content
                messages.append({"role": role, "content": content})
            summary_text = await self._api_chat(messages)
            summary_text = self._sanitize_reply(summary_text)
            # Add as a memory entry (global scope as fallback)
            await self.add_memory_entry(summary_text, MemoryScope.GLOBAL, {"type": "conversation_summary", "conversation_id": conversation_id})
            # Compact stored turns: keep only the last few plus a marker
            self.active_conversations[conversation_id] = recent[-6:]
        except Exception:
            log.exception("Failed to summarize conversation")

    async def _compact_memory_entries(self):
        """Compact and optimize memory storage."""
        # Remove rarely accessed memories
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=30)
        
        to_remove = []
        for memory_id, entry in self.memory_entries.items():
            if (entry.last_accessed and entry.last_accessed < cutoff_time and 
                entry.access_count < 3):
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self.memory_entries[memory_id]
        
        if to_remove:
            log.info(f"Compacted {len(to_remove)} rarely accessed memories")

    async def _check_system_health(self):
        """Check system health and update status."""
        health_details = {}
        
        try:
            # Check API connectivity
            base, key, _, _ = await self._get_keys()
            if base and key:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{base.rstrip('/')}/health")
                    health_details["api_status"] = "healthy" if response.status_code == 200 else "unhealthy"
            else:
                health_details["api_status"] = "not_configured"
        except Exception as e:
            health_details["api_status"] = f"error: {str(e)}"
        
        # Check database
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memory_entries")
                health_details["memory_count"] = cursor.fetchone()[0]
                conn.close()
        except Exception as e:
            health_details["database_status"] = f"error: {str(e)}"
        
        # Check memory usage
        health_details["active_conversations"] = len(self.active_conversations)
        health_details["user_profiles"] = len(self.user_memory_profiles)
        health_details["guild_knowledge_bases"] = len(self.guild_knowledge_bases)
        
        self.health_status = {
            "last_check": datetime.now().isoformat(),
            "status": "healthy" if health_details.get("api_status") == "healthy" else "degraded",
            "details": health_details
        }
        
        await self.config.system_health.set(self.health_status)

    async def _perform_backup(self):
        """Perform backup of all data."""
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "user_profiles": {str(k): asdict(v) for k, v in self.user_memory_profiles.items()},
            "guild_knowledge_bases": {str(k): asdict(v) for k, v in self.guild_knowledge_bases.items()},
            "memory_entries": {k: asdict(v) for k, v in self.memory_entries.items()},
            "tools": {k: asdict(v) for k, v in self.tools.items()},
            "config": await self.config.all()
        }
        
        backup_path = Path("data/OpenWebUIChat/backups")
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"backup_{timestamp}.json"
        
        async with aiofiles.open(backup_file, 'w') as f:
            await f.write(json.dumps(backup_data, indent=2, default=str))
        
        log.info(f"Backup completed: {backup_file}")

    async def _collect_analytics(self):
        """Collect usage analytics."""
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "total_memories": len(self.memory_entries),
            "active_users": len(self.user_memory_profiles),
            "active_guilds": len(self.guild_knowledge_bases),
            "total_tools": len(self.tools),
            "health_status": self.health_status
        }
        
        # Store analytics (could be sent to external service)
        analytics_path = Path("data/OpenWebUIChat/analytics")
        analytics_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analytics_file = analytics_path / f"analytics_{timestamp}.json"
        
        async with aiofiles.open(analytics_file, 'w') as f:
            await f.write(json.dumps(analytics, indent=2, default=str))

    # ───────────────── memory management ───────────
    async def get_user_memory_profile(self, user_id: int) -> UserMemoryProfile:
        """Get or create a user memory profile."""
        if user_id not in self.user_memory_profiles:
            self.user_memory_profiles[user_id] = UserMemoryProfile(
                user_id=user_id,
                short_term_memory=[],
                long_term_profile={},
                task_log=[],
                preferences={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                ttl_expires=datetime.now() + timedelta(hours=MEMORY_TTL_HOURS)
            )
        return self.user_memory_profiles[user_id]

    async def update_user_memory(self, user_id: int, role: str, content: str, metadata: Dict[str, Any] = None):
        """Update user's short-term memory with a new conversation turn."""
        profile = await self.get_user_memory_profile(user_id)
        
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            tokens_used=len(content.split())  # Rough token estimate
        )
        
        profile.short_term_memory.append(turn)
        
        # Keep only recent turns
        if len(profile.short_term_memory) > MAX_CONVERSATION_TURNS:
            profile.short_term_memory = profile.short_term_memory[-MAX_CONVERSATION_TURNS:]
        
        profile.updated_at = datetime.now()
        profile.ttl_expires = datetime.now() + timedelta(hours=MEMORY_TTL_HOURS)

    async def get_guild_knowledge_base(self, guild_id: int) -> GuildKnowledgeBase:
        """Get or create a guild knowledge base."""
        if guild_id not in self.guild_knowledge_bases:
            self.guild_knowledge_bases[guild_id] = GuildKnowledgeBase(
                guild_id=guild_id,
                name=f"Guild {guild_id} Knowledge Base",
                scope=MemoryScope.GUILD,
                documents={},
                faqs={},
                house_rules=[],
                project_docs={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        return self.guild_knowledge_bases[guild_id]

    async def add_memory_entry(self, text: str, scope: MemoryScope, metadata: Dict[str, Any] = None) -> str:
        """Add a new memory entry with embedding."""
        memory_id = str(uuid.uuid4())
        embedding = await self._api_embed(text)
        
        entry = MemoryEntry(
            id=memory_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            scope=scope,
            created_at=datetime.now(),
            ttl_expires=datetime.now() + timedelta(hours=MEMORY_TTL_HOURS)
        )
        
        self.memory_entries[memory_id] = entry
        return memory_id

    async def search_memories(self, query: str, scope: MemoryScope = None, limit: int = TOP_K) -> List[MemoryEntry]:
        """Search memories using hybrid retrieval."""
        if not self.memory_entries:
            return []
        
        # Filter by scope if specified
        candidates = list(self.memory_entries.values())
        if scope:
            candidates = [m for m in candidates if m.scope == scope]
        
        if not candidates:
            return []
        
        # Get query embedding
        query_embedding = np.array(await self._api_embed(query))
        
        # Dense retrieval (cosine similarity)
        dense_scores = []
        for entry in candidates:
            vec = np.array(entry.embedding)
            similarity = self._cos(query_embedding, vec)
            if similarity >= SIM_THRESHOLD:
                dense_scores.append((similarity, entry))
        
        dense_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Sparse retrieval (BM25)
        texts = [entry.text for entry in candidates]
        bm25 = BM25Okapi([self._normalize_text(t).split() for t in texts])
        query_tokens = self._normalize_text(query).split()
        bm25_scores = bm25.get_scores(query_tokens)
        
        sparse_scores = []
        for i, entry in enumerate(candidates):
            score = bm25_scores[i]
            if score > 0:
                sparse_scores.append((score, entry))
        
        sparse_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Combine scores
        combined_scores = {}
        for sim, entry in dense_scores:
            combined_scores[entry.id] = combined_scores.get(entry.id, 0) + sim * 0.7
        
        for score, entry in sparse_scores:
            combined_scores[entry.id] = combined_scores.get(entry.id, 0) + score * 0.3
        
        # Sort by combined score and return top results
        sorted_entries = sorted(
            [(score, self.memory_entries[entry_id]) for entry_id, score in combined_scores.items()],
            key=lambda x: x[0],
            reverse=True
        )
        
        results = [entry for _, entry in sorted_entries[:limit]]
        
        # Update access statistics
        for entry in results:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        
        return results

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
        """Generate embeddings using the configured provider."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Get provider and configuration
        provider = EmbeddingProvider(await self.config.embedding_provider())
        base, key, _, embed_model = await self._get_keys()
        
        if not base or not key:
            raise RuntimeError("OpenWebUI URL or key not set.")
        
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        
        try:
            if provider == EmbeddingProvider.OPENWEBUI:
                embedding = await self._embed_openwebui(base, headers, embed_model, text)
            elif provider == EmbeddingProvider.OLLAMA:
                embedding = await self._embed_ollama(base, headers, embed_model, text)
            elif provider == EmbeddingProvider.FAISS:
                embedding = await self._embed_faiss(text)
            elif provider == EmbeddingProvider.CHROMA:
                embedding = await self._embed_chroma(text)
            elif provider == EmbeddingProvider.PGVECTOR:
                embedding = await self._embed_pgvector(text)
            else:
                # Fallback to OpenWebUI
                embedding = await self._embed_openwebui(base, headers, embed_model, text)
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            return embedding
            
        except Exception as e:
            log.error(f"Embedding generation failed with {provider.value}: {e}")
            # Fallback to OpenWebUI
            if provider != EmbeddingProvider.OPENWEBUI:
                log.info("Falling back to OpenWebUI embedding provider")
                embedding = await self._embed_openwebui(base, headers, embed_model, text)
                self.embedding_cache[text_hash] = embedding
                return embedding
            raise

    async def _embed_openwebui(self, base: str, headers: dict, model: str, text: str) -> List[float]:
        """Generate embeddings using OpenWebUI API."""
        ollama_base = base.replace('/api', '/ollama')
        async with httpx.AsyncClient(timeout=60) as c:
            try:
                r = await c.post(f"{ollama_base.rstrip('/')}/api/embed",
                                headers=headers,
                                json={"model": model, "input": [self._normalize_text(text)]})
                r.raise_for_status()
                return r.json()["embeddings"][0]
            except httpx.HTTPStatusError:
                r = await c.post(f"{base.rstrip('/')}/api/embeddings",
                                headers=headers,
                                json={"model": model, "input": [self._normalize_text(text)]})
                r.raise_for_status()
                return r.json()["data"][0]["embedding"]

    async def _embed_ollama(self, base: str, headers: dict, model: str, text: str) -> List[float]:
        """Generate embeddings using Ollama API."""
        ollama_base = base.replace('/api', '/ollama')
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{ollama_base.rstrip('/')}/api/embed",
                            headers=headers,
                            json={"model": model, "input": [self._normalize_text(text)]})
            r.raise_for_status()
            return r.json()["embeddings"][0]

    async def _embed_faiss(self, text: str) -> List[float]:
        """Generate embeddings using local FAISS (placeholder implementation)."""
        # This would integrate with a local FAISS index
        # For now, return a placeholder embedding
        import random
        return [random.random() for _ in range(384)]  # Typical embedding size

    async def _embed_chroma(self, text: str) -> List[float]:
        """Generate embeddings using ChromaDB (placeholder implementation)."""
        # This would integrate with ChromaDB
        # For now, return a placeholder embedding
        import random
        return [random.random() for _ in range(384)]  # Typical embedding size

    async def _embed_pgvector(self, text: str) -> List[float]:
        """Generate embeddings using PGVector (placeholder implementation)."""
        # This would integrate with PostgreSQL with pgvector extension
        # For now, return a placeholder embedding
        import random
        return [random.random() for _ in range(384)]  # Typical embedding size

    def _sanitize_reply(self, text: str) -> str:
        """Remove chain-of-thought markers like <think>...</think> and stage directions."""
        if not text:
            return FALLBACK
        # Remove <think> blocks
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # Remove leading stage directions like (thinking ...)
        text = re.sub(r"^\s*\((?:[^()]*|\([^()]*\))*\)\s*", "", text)
        return text.strip() or FALLBACK

    async def _generate_reply(self, question: str, mems: Optional[Dict[str, Dict]] = None, *, 
                            channel: Optional[discord.abc.GuildChannel] = None, 
                            user: Optional[discord.abc.User] = None,
                            guild: Optional[discord.Guild] = None,
                            response_mode: ResponseMode = ResponseMode.NORMAL) -> Tuple[str, Dict[str, Any]]:
        """Enhanced reply generation with persona awareness, memory integration, and tool support."""
        
        # Get user and guild settings
        user_prefs = await self._get_user_preferences(user) if user else {}
        guild_settings = await self._get_guild_settings(guild) if guild else {}
        channel_settings = await self._get_channel_settings(channel) if channel else {}
        
        # Determine persona and system prompt
        persona_type = PersonaType(guild_settings.get("persona_type", PersonaType.ALICENT.value))
        
        # Build message context
        message_context = {}
        if channel:
            message_context.update({
                "channel_type": getattr(channel, 'type', 'text'),
                "is_thread": hasattr(channel, 'parent') and channel.parent is not None,
                "is_voice": getattr(channel, 'type', None) == discord.ChannelType.voice,
                "has_attachments": False,  # Will be updated if we have access to the original message
                "message_length": len(question)
            })
        
        system_prompt = await self._build_system_prompt(
            persona_type, user_prefs, guild_settings, channel_settings, response_mode, message_context
        )
        
        # Enhanced memory retrieval
        relevant_memories = []
        source_items: List[str] = []
        if mems or self.memory_entries:
            try:
                # Search both legacy memories and new memory entries
                if mems:
                    prompt_vec = np.array(await self._api_embed(question))
                    legacy_memories = await self._best_memories(prompt_vec, question, mems)
                    relevant_memories.extend(legacy_memories)
                    for mem_text in legacy_memories[:TOP_K]:
                        source_items.append(f"legacy: {mem_text[:180]}{'…' if len(mem_text) > 180 else ''}")
                
                # Search new memory system
                memory_scope = MemoryScope.GUILD if guild else MemoryScope.GLOBAL
                if channel:
                    memory_scope = MemoryScope.CHANNEL
                if user:
                    memory_scope = MemoryScope.USER
                
                new_memories = await self.search_memories(question, memory_scope, limit=TOP_K)
                relevant_memories.extend([m.text for m in new_memories])
                for m in new_memories[:TOP_K]:
                    label = m.metadata.get("type", "memory")
                    source_items.append(f"{label}: {m.text[:180]}{'…' if len(m.text) > 180 else ''}")
                
                # Add guild knowledge base content
                if guild:
                    guild_kb = await self.get_guild_knowledge_base(guild.id)
                    if guild_kb.faqs:
                        # Simple FAQ matching
                        question_lower = question.lower()
                        for faq_q, faq_a in guild_kb.faqs.items():
                            if any(word in question_lower for word in faq_q.lower().split()):
                                faq_text = f"FAQ: {faq_q} - {faq_a}"
                                relevant_memories.append(faq_text)
                                source_items.append(faq_text[:200] + ("…" if len(faq_text) > 200 else ""))
                
            except Exception as e:
                log.warning(f"Failed to retrieve memories: {e}")

        # Enhance system prompt with relevant knowledge
        if relevant_memories:
            system_prompt += (
                "\n\nYou have access to relevant knowledge:\n"
                + "\n".join(f"- {mem}" for mem in relevant_memories[:TOP_K])
            )
        
        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add user's conversation history
        if user:
            user_profile = await self.get_user_memory_profile(user.id)
            for turn in user_profile.short_term_memory[-10:]:  # Last 10 turns
                messages.append({"role": turn.role, "content": turn.content})
        
        # Add channel history if enabled
        if channel and channel_settings.get("history_enabled", True):
            hist = await self.config.channel(channel).history()
            if hist:
                max_turns = channel_settings.get("history_max_turns", 20)
                messages.extend(hist[-(max_turns * 2):])
        
        # Add current question
        messages.append({"role": "user", "content": question})

        # Check for tool usage
        tools_to_use = await self._determine_tools_to_use(question, guild, channel, user)
        tool_results = {}
        
        if tools_to_use:
            tool_results = await self._execute_tools(tools_to_use, question, user, guild)
            if tool_results:
                # Add tool results to context
                tool_context = "\n\nTool Results:\n"
                for tool_name, result in tool_results.items():
                    tool_context += f"{tool_name}: {result}\n"
                messages.append({"role": "system", "content": tool_context})
        
        # Generate response
        reply = await self._api_chat(messages)
        sanitized_reply = self._sanitize_reply(reply)

        # Apply response guardrails
        sanitized_reply, guard_flags = self._apply_guardrails(question, sanitized_reply)
        
        # Update user memory
        if user:
            await self.update_user_memory(user.id, "user", question, {
                "channel_id": channel.id if channel else None,
                "guild_id": guild.id if guild else None,
                "response_mode": response_mode.value
            })
            await self.update_user_memory(user.id, "assistant", sanitized_reply, {
                "tools_used": list(tool_results.keys()),
                "memories_used": len(relevant_memories)
            })
        
        # Prepare metadata for response
        metadata = {
            "persona": persona_type.value,
            "memories_used": len(relevant_memories),
            "tools_used": list(tool_results.keys()),
            "response_mode": response_mode.value,
            "user_preferences": user_prefs,
            "guild_settings": guild_settings,
            "sources": source_items[:5],
            "guardrails": guard_flags
        }
        
        return sanitized_reply, metadata

    def _apply_guardrails(self, question: str, reply: str) -> Tuple[str, Dict[str, Any]]:
        """Simple response guardrails: basic toxicity filter and safety fallback.
        Returns possibly modified reply and flags about what was applied.
        """
        flags: Dict[str, Any] = {"toxic": False}
        try:
            lower = f"{question}\n{reply}".lower()
            banned = [
                "kill yourself", "kys", "hate speech", "racist", "nazi", "slur"
            ]
            if any(term in lower for term in banned):
                flags["toxic"] = True
                safe = (
                    "I can't assist with that. Let's keep the conversation respectful and safe. "
                    "If you need support, consider reaching out to a moderator."
                )
                return safe, flags
        except Exception:
            pass
        return reply, flags

    async def _get_user_preferences(self, user: discord.abc.User) -> Dict[str, Any]:
        """Get user preferences and settings."""
        if not user:
            return {}
        
        user_conf = self.config.user(user)
        return {
            "language_preference": await user_conf.language_preference(),
            "persona_preference": await user_conf.persona_preference(),
            "response_length": await user_conf.response_length(),
            "creativity_level": await user_conf.creativity_level(),
            "verbosity_level": await user_conf.verbosity_level(),
            "tools_enabled": await user_conf.tools_enabled(),
        }

    async def _get_guild_settings(self, guild: discord.Guild) -> Dict[str, Any]:
        """Get guild-specific settings."""
        if not guild:
            return {}
        
        guild_conf = self.config.guild(guild)
        return {
            "persona_type": await guild_conf.persona_type(),
            "response_mode": await guild_conf.response_mode(),
            "allowed_tools": await guild_conf.allowed_tools(),
            "policy_preset": await guild_conf.policy_preset(),
            "knowledge_base_enabled": await guild_conf.knowledge_base_enabled(),
            "moderation_enabled": await guild_conf.moderation_enabled(),
            "web_search_enabled": await guild_conf.web_search_enabled(),
            "thread_first_mode": await guild_conf.thread_first_mode(),
            "ephemeral_responses": await guild_conf.ephemeral_responses(),
        }

    async def _get_channel_settings(self, channel: discord.abc.GuildChannel) -> Dict[str, Any]:
        """Get channel-specific settings."""
        if not channel:
            return {}
        
        channel_conf = self.config.channel(channel)
        return {
            "history_enabled": await channel_conf.history_enabled(),
            "history_max_turns": await channel_conf.history_max_turns(),
            "prompt_template": await channel_conf.prompt_template(),
            "response_mode": await channel_conf.response_mode(),
            "allowed_tools": await channel_conf.allowed_tools(),
            "auto_thread": await channel_conf.auto_thread(),
            "snippet_mode": await channel_conf.snippet_mode(),
            "show_sources": await channel_conf.show_sources(),
        }

    async def _build_system_prompt(self, persona_type: PersonaType, user_prefs: Dict[str, Any], 
                                 guild_settings: Dict[str, Any], channel_settings: Dict[str, Any],
                                 response_mode: ResponseMode, message_context: Dict[str, Any] = None) -> str:
        """Build context-aware system prompt based on persona, settings, and message context."""
        
        # Get persona template
        persona_templates = await self.config.persona_templates()
        persona_template = persona_templates.get(persona_type.value, {})
        base_prompt = persona_template.get("system_prompt", "You are a helpful AI assistant.")
        
        # Add channel-specific template if available
        if channel_settings.get("prompt_template"):
            base_prompt = channel_settings["prompt_template"]
        
        # Add message-aware context
        if message_context:
            channel_type = message_context.get("channel_type", "text")
            is_thread = message_context.get("is_thread", False)
            is_voice = message_context.get("is_voice", False)
            has_attachments = message_context.get("has_attachments", False)
            message_length = message_context.get("message_length", 0)
            
            # Adjust prompt based on channel type
            if channel_type == "forum":
                base_prompt += "\n\nThis is a forum discussion. Provide comprehensive responses that contribute meaningfully to the conversation."
            elif is_thread:
                base_prompt += "\n\nThis is a thread conversation. Keep responses focused on the thread topic."
            elif is_voice:
                base_prompt += "\n\nThis is a voice channel context. Provide concise responses suitable for voice communication."
            
            # Adjust based on message characteristics
            if has_attachments:
                base_prompt += "\n\nThe user has shared attachments. Consider these in your response and offer to help analyze them."
            
            if message_length > 500:
                base_prompt += "\n\nThe user has provided a detailed message. Take time to address all their points thoroughly."
            elif message_length < 50:
                base_prompt += "\n\nThe user's message is brief. Provide a concise but helpful response."
        
        # Add response mode instructions
        if response_mode == ResponseMode.SHOW_WORK:
            base_prompt += "\n\nShow your reasoning and cite sources when available. Be transparent about your process."
        elif response_mode == ResponseMode.SNIPPET:
            base_prompt += "\n\nWhen providing code or technical content, format it clearly and provide complete examples."
        elif response_mode == ResponseMode.THREAD:
            base_prompt += "\n\nProvide detailed, comprehensive responses suitable for threaded discussions."
        
        # Add user preference adjustments
        if user_prefs.get("response_length") == "short":
            base_prompt += "\n\nKeep responses concise and to the point."
        elif user_prefs.get("response_length") == "detailed":
            base_prompt += "\n\nProvide detailed explanations and examples."
        elif user_prefs.get("response_length") == "ultra":
            base_prompt += "\n\nProvide comprehensive, in-depth responses with extensive detail."
        
        # Add creativity level
        creativity = user_prefs.get("creativity_level", 0.7)
        if creativity < 0.3:
            base_prompt += "\n\nBe conservative and factual in your responses."
        elif creativity > 0.8:
            base_prompt += "\n\nBe creative and imaginative while staying helpful."
        
        # Add policy restrictions
        policy_preset = guild_settings.get("policy_preset", "default")
        if policy_preset == "sfw_only":
            base_prompt += "\n\nKeep all content appropriate for all audiences."
        elif policy_preset == "strict":
            base_prompt += "\n\nMaintain a formal, professional tone at all times."
        elif policy_preset == "roleplay":
            base_prompt += "\n\nYou may engage in roleplay scenarios when appropriate."
        
        return base_prompt

    async def _determine_tools_to_use(self, question: str, guild: Optional[discord.Guild], 
                                    channel: Optional[discord.abc.GuildChannel], 
                                    user: Optional[discord.abc.User]) -> List[str]:
        """Determine which tools should be used for this question."""
        tools_to_use = []
        
        # Check if tools are enabled for user
        if user:
            user_prefs = await self._get_user_preferences(user)
            if not user_prefs.get("tools_enabled", True):
                return tools_to_use
        
        # Check guild tool allowlist
        guild_allowed_tools = set()
        if guild:
            guild_settings = await self._get_guild_settings(guild)
            guild_allowed_tools = set(guild_settings.get("allowed_tools", []))
        
        # Check channel tool allowlist
        channel_allowed_tools = set()
        if channel:
            channel_settings = await self._get_channel_settings(channel)
            channel_allowed_tools = set(channel_settings.get("allowed_tools", []))
        
        # Simple keyword-based tool selection
        question_lower = question.lower()
        
        if "search" in question_lower or "find" in question_lower or "look up" in question_lower:
            if "web_search" in self.tools and self._is_tool_allowed("web_search", guild_allowed_tools, channel_allowed_tools):
                tools_to_use.append("web_search")
        
        if any(keyword in question_lower for keyword in ["code", "program", "function", "script", "bug", "error"]):
            if "code_lint" in self.tools and self._is_tool_allowed("code_lint", guild_allowed_tools, channel_allowed_tools):
                tools_to_use.append("code_lint")
        
        if any(keyword in question_lower for keyword in ["image", "picture", "generate", "create", "draw"]):
            if "image_generate" in self.tools and self._is_tool_allowed("image_generate", guild_allowed_tools, channel_allowed_tools):
                tools_to_use.append("image_generate")
        
        if any(keyword in question_lower for keyword in ["math", "calculate", "solve", "equation", "formula"]):
            if "math_solve" in self.tools and self._is_tool_allowed("math_solve", guild_allowed_tools, channel_allowed_tools):
                tools_to_use.append("math_solve")
        
        return tools_to_use

    def _is_tool_allowed(self, tool_name: str, guild_allowed: Set[str], channel_allowed: Set[str]) -> bool:
        """Check if a tool is allowed based on guild and channel settings."""
        # If no restrictions are set, allow all tools
        if not guild_allowed and not channel_allowed:
            return True
        
        # If channel has specific allowlist, use that
        if channel_allowed:
            return tool_name in channel_allowed
        
        # Otherwise use guild allowlist
        return tool_name in guild_allowed

    async def _execute_tools(self, tools_to_use: List[str], question: str, 
                           user: Optional[discord.abc.User], guild: Optional[discord.Guild]) -> Dict[str, str]:
        """Execute the selected tools and return results."""
        results = {}
        
        for tool_name in tools_to_use:
            if tool_name not in self.tools:
                continue
            
            tool = self.tools[tool_name]
            
            try:
                if tool_name == "web_search":
                    results[tool_name] = await self._web_search_tool(question)
                elif tool_name == "code_lint":
                    results[tool_name] = await self._code_lint_tool(question)
                elif tool_name == "image_generate":
                    results[tool_name] = await self._image_generate_tool(question)
                elif tool_name == "math_solve":
                    results[tool_name] = await self._math_solve_tool(question)
                elif tool_name == "file_analyze":
                    results[tool_name] = await self._file_analyze_tool(question)
                
            except Exception as e:
                log.error(f"Error executing tool {tool_name}: {e}")
                results[tool_name] = f"Tool execution failed: {str(e)}"
        
        return results

    # Tool implementations (placeholder - would be fully implemented)
    async def _web_search_tool(self, query: str) -> str:
        """Web search tool implementation."""
        # This would integrate with a real search API
        return f"Web search results for: {query} (placeholder implementation)"

    async def _code_lint_tool(self, code: str) -> str:
        """Code linting tool implementation."""
        # This would integrate with real linting tools
        return f"Code analysis for: {code[:100]}... (placeholder implementation)"

    async def _image_generate_tool(self, prompt: str) -> str:
        """Image generation tool implementation."""
        # This would integrate with image generation APIs
        return f"Image generation for: {prompt} (placeholder implementation)"

    async def _math_solve_tool(self, expression: str) -> str:
        """Math solving tool implementation."""
        # This would integrate with math solving libraries
        return f"Math solution for: {expression} (placeholder implementation)"

    async def _file_analyze_tool(self, content: str) -> str:
        """File analysis tool implementation."""
        # This would analyze uploaded files
        return f"File analysis for: {content[:100]}... (placeholder implementation)"

    def _build_embeds(self, content: str, author: Optional[discord.abc.User], model_name: Optional[str]) -> List[discord.Embed]:
        """Chunk long content into multiple modern embeds."""
        chunks = [content[i:i + EMBED_CHUNK] for i in range(0, len(content), EMBED_CHUNK)] or [FALLBACK]
        embeds: List[discord.Embed] = []
        for idx, chunk in enumerate(chunks, start=1):
            embed = discord.Embed(description=chunk, color=discord.Color.blurple())
            if author:
                try:
                    embed.set_author(name="Alicent", icon_url=author.display_avatar.url)
                except Exception:
                    embed.set_author(name="Alicent")
            else:
                embed.set_author(name="Alicent")
            footer = "OpenWebUIChat"
            if model_name:
                footer += f" • {model_name}"
            if len(chunks) > 1:
                footer += f" • Part {idx}/{len(chunks)}"
            embed.set_footer(text=footer)
            embeds.append(embed)
        return embeds

    # ───────────────── conversation history helpers ─────────────────
    async def _append_channel_history(self, channel: discord.abc.GuildChannel, role: str, content: str):
        chan_conf = self.config.channel(channel)
        if not await chan_conf.history_enabled():
            return
        hist = await chan_conf.history()
        hist.append({"role": role, "content": content})
        max_turns = await chan_conf.history_max_turns()
        # keep last max_turns user+assistant pairs plus system is not stored here
        max_messages = max_turns * 2
        if len(hist) > max_messages:
            hist = hist[-max_messages:]
        await chan_conf.history.set(hist)

    # ───────────────── interactive controls ─────────────────
    class _RegeneratePayload:
        def __init__(self, user_text: str):
            self.user_text = user_text

    class _ClearHistoryPayload:
        pass

    def _build_controls_view(self, channel: discord.abc.GuildChannel, author: discord.abc.User, original_user: str):
        view = discord.ui.View(timeout=120)

        async def regen_callback(interaction: discord.Interaction):
            if interaction.user.id != author.id:
                return await interaction.response.send_message("Only the requester can use these controls.", ephemeral=True)
            await interaction.response.defer()
            await self._append_channel_history(channel, "user", original_user)
            mems = await self.config.memories()
            reply, _meta = await self._generate_reply(
                original_user, mems, channel=channel, user=author, guild=channel.guild if hasattr(channel, 'guild') else None
            )
            model = await self.config.chat_model()
            embeds = self._build_embeds(reply, author, model)
            await self._append_channel_history(channel, "assistant", reply)
            for embed in embeds:
                await interaction.followup.send(embed=embed)

        async def clear_callback(interaction: discord.Interaction):
            if interaction.user.id != author.id:
                return await interaction.response.send_message("Only the requester can use these controls.", ephemeral=True)
            await self.config.channel(channel).history.set([])
            await interaction.response.send_message("🧹 Channel history cleared.", ephemeral=True)

        async def like_callback(interaction: discord.Interaction):
            await interaction.response.send_message("Thanks for the feedback!", ephemeral=True)

        async def dislike_callback(interaction: discord.Interaction):
            await interaction.response.send_message("Feedback noted.", ephemeral=True)

        view.add_item(discord.ui.Button(label="Regenerate", style=discord.ButtonStyle.secondary))
        view.children[-1].callback = regen_callback
        view.add_item(discord.ui.Button(label="Clear History", style=discord.ButtonStyle.danger))
        view.children[-1].callback = clear_callback
        view.add_item(discord.ui.Button(emoji="👍", style=discord.ButtonStyle.success))
        view.children[-1].callback = like_callback
        view.add_item(discord.ui.Button(emoji="👎", style=discord.ButtonStyle.secondary))
        view.children[-1].callback = dislike_callback
        return view

    def _can_use_controls(self, channel: Optional[discord.abc.GuildChannel]) -> bool:
        return isinstance(channel, discord.abc.GuildChannel)

    def _controls_for(self, channel: discord.abc.GuildChannel, author: discord.abc.User, original_user: str):
        try:
            return self._build_controls_view(channel, author, original_user)
        except Exception:
            return None

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
        # If this is an auto-reply channel, append the user's message BEFORE generating the reply
        try:
            auto_channels = await self.config.guild(ctx.guild).auto_channels()
            is_auto = ctx.channel.id in auto_channels
        except Exception:
            is_auto = False
        if is_auto and isinstance(ctx.channel, discord.abc.GuildChannel):
            await self._append_channel_history(ctx.channel, "user", question)

        mems = await self.config.memories()
        reply, meta = await self._generate_reply(
            question,
            mems,
            channel=ctx.channel if isinstance(ctx.channel, discord.abc.GuildChannel) else None,
            user=ctx.author,
            guild=ctx.guild,
            response_mode=ResponseMode(await self.config.guild(ctx.guild).response_mode()) if ctx.guild else ResponseMode.NORMAL,
        )
        log.info(f"AI response (sanitized): '{reply}' | meta={meta}")
        model = await self.config.chat_model()
        embeds = self._build_embeds(reply, ctx.author, model)
        # Append sources card if enabled
        try:
            if isinstance(ctx.channel, discord.abc.GuildChannel):
                if await self.config.channel(ctx.channel).show_sources():
                    sources = []
                    # Best-effort: retrieve last metadata by regenerating lightweight context
                    # In this simple version, we only indicate that sources are available via knowledge base and memory count.
                    sources.append("Sources are available via knowledge base and memory retrieval.")
                    src_embed = discord.Embed(title="Sources", description="\n".join(sources), color=discord.Color.dark_gray())
                    embeds.append(src_embed)
        except Exception:
            pass
        # Send first embed with controls, rest plain
        if embeds:
            view = self._controls_for(ctx.channel, ctx.author, question) if self._can_use_controls(ctx.channel) else None
            await ctx.send(embed=embeds[0], view=view)
            for embed in embeds[1:]:
                await ctx.send(embed=embed)
        # Update channel history for auto-reply channels only
        try:
            auto_channels = await self.config.guild(ctx.guild).auto_channels()
            if ctx.channel.id in auto_channels:
                await self._append_channel_history(ctx.channel, "assistant", reply)
        except Exception:
            pass

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
                reply, _meta = await self.outer._generate_reply(
                    str(self.prompt.value), await self.outer.config.memories(), user=interaction.user, guild=interaction.guild
                )
                model = await self.outer.config.chat_model()
                embeds = self.outer._build_embeds(reply, interaction.user, model)
                try:
                    ch = interaction.channel
                    if ch and isinstance(ch, discord.abc.GuildChannel):
                        if await self.outer.config.channel(ch).show_sources():
                            src = discord.Embed(title="Sources", description="Knowledge base and memory retrieval used where applicable.", color=discord.Color.dark_gray())
                            embeds.append(src)
                except Exception:
                    pass
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

    @setopenwebui.command()
    async def embedprovider(self, ctx, provider: str):
        """Set the embedding provider (openwebui, ollama, faiss, chroma, pgvector)."""
        try:
            provider_enum = EmbeddingProvider(provider.lower())
            await self.config.embedding_provider.set(provider_enum.value)
            await ctx.send(f"✅ Embedding provider set to {provider_enum.value}.")
        except ValueError:
            valid_providers = [p.value for p in EmbeddingProvider]
            await ctx.send(f"❌ Invalid provider. Valid options: {', '.join(valid_providers)}")

    @setopenwebui.command()
    async def embedproviders(self, ctx):
        """List available embedding providers."""
        providers = []
        for provider in EmbeddingProvider:
            current = " (current)" if provider.value == await self.config.embedding_provider() else ""
            providers.append(f"- **{provider.value}**{current}")
        
        embed = discord.Embed(
            title="Available Embedding Providers",
            description="\n".join(providers),
            color=discord.Color.blurple()
        )
        embed.add_field(
            name="Usage",
            value="Use `[p]setopenwebui embedprovider <provider>` to switch",
            inline=False
        )
        await ctx.send(embed=embed)

    # ───────────────── prompt templates management ───────────────
    @commands.hybrid_group(name="prompt")
    @commands.is_owner()
    async def prompt(self, ctx: commands.Context):
        """Manage prompt templates per-channel."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @prompt.command(name="set")
    async def prompt_set(self, ctx: commands.Context, *, template: str):
        """Set a custom prompt template for this channel."""
        await self.config.channel(ctx.channel).prompt_template.set(template)
        await ctx.send("✅ Prompt template set for this channel.")

    @prompt.command(name="clear")
    async def prompt_clear(self, ctx: commands.Context):
        """Clear the custom prompt template for this channel."""
        await self.config.channel(ctx.channel).prompt_template.set("")
        await ctx.send("🧹 Prompt template cleared for this channel.")

    @prompt.command(name="show")
    async def prompt_show(self, ctx: commands.Context):
        """Show the current prompt template for this channel."""
        tpl = await self.config.channel(ctx.channel).prompt_template()
        await ctx.send(box(tpl or "<none>", lang="md"))

    @prompt.command(name="mode")
    async def prompt_mode(self, ctx: commands.Context, mode: str):
        """Set response mode: normal | show_work | snippet | thread."""
        try:
            mode_enum = ResponseMode(mode.lower())
        except ValueError:
            return await ctx.send("❌ Invalid mode. Use one of: normal, show_work, snippet, thread")
        await self.config.channel(ctx.channel).response_mode.set(mode_enum.value)
        await ctx.send(f"✅ Response mode set to **{mode_enum.value}** for this channel.")

    @prompt.command(name="sources")
    async def prompt_sources(self, ctx: commands.Context, show: bool):
        """Toggle showing sources/KB hits under replies in this channel."""
        await self.config.channel(ctx.channel).show_sources.set(bool(show))
        await ctx.send(f"✅ Show sources set to **{bool(show)}** for this channel.")

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

    @openwebuimemory.command(name="stats")
    async def memory_stats(self, ctx):
        """Show memory system statistics."""
        total_memories = len(self.memory_entries)
        expired_count = sum(1 for entry in self.memory_entries.values() 
                           if entry.ttl_expires and entry.ttl_expires < datetime.now())
        archived_count = 0
        
        # Count archived memories from database
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM archived_memories")
                archived_count = cursor.fetchone()[0]
                conn.close()
        except Exception:
            pass
        
        # Count by scope
        scope_counts = {}
        for entry in self.memory_entries.values():
            scope = entry.scope.value
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        
        # Count by type
        type_counts = {}
        for entry in self.memory_entries.values():
            entry_type = entry.metadata.get("type", "general")
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        embed = discord.Embed(
            title="Memory System Statistics",
            color=discord.Color.blurple()
        )
        
        embed.add_field(
            name="📊 Active Memories",
            value=str(total_memories),
            inline=True
        )
        embed.add_field(
            name="🗄️ Archived Memories",
            value=str(archived_count),
            inline=True
        )
        embed.add_field(
            name="⏰ Expired Memories",
            value=str(expired_count),
            inline=True
        )
        
        embed.add_field(
            name="🎯 By Scope",
            value="\n".join(f"{k}: {v}" for k, v in scope_counts.items()) if scope_counts else "None",
            inline=True
        )
        embed.add_field(
            name="📝 By Type",
            value="\n".join(f"{k}: {v}" for k, v in type_counts.items()) if type_counts else "None",
            inline=True
        )
        embed.add_field(
            name="💾 Cache Size",
            value=f"{len(self.embedding_cache)} embeddings",
            inline=True
        )
        
        await ctx.send(embed=embed)

    @openwebuimemory.command(name="cleanup")
    async def memory_cleanup(self, ctx):
        """Manually trigger memory cleanup and archiving."""
        await ctx.send("🧹 Starting memory cleanup...")
        
        try:
            await self._cleanup_expired_memories()
            await self._decay_memory_importance()
            await self._compact_memory_entries()
            await ctx.send("✅ Memory cleanup completed successfully.")
        except Exception as e:
            await ctx.send(f"❌ Memory cleanup failed: {str(e)}")

    @openwebuimemory.command(name="ttl")
    async def memory_ttl(self, ctx, hours: int = None):
        """Set or view memory TTL (Time To Live) in hours."""
        global MEMORY_TTL_HOURS
        
        if hours is None:
            current_ttl = MEMORY_TTL_HOURS
            await ctx.send(f"Current memory TTL: **{current_ttl} hours**")
        else:
            if hours < 1 or hours > 8760:  # Max 1 year
                return await ctx.send("❌ TTL must be between 1 and 8760 hours (1 year).")
            
            # Update global TTL constant (in a real implementation, this would be configurable)
            MEMORY_TTL_HOURS = hours
            await ctx.send(f"✅ Memory TTL set to **{hours} hours**")

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

    # ───────────────── history management ─────────────────
    @openwebui.group(name="history")
    @commands.is_owner()
    async def history(self, ctx: commands.Context):
        """Manage per-channel conversation history."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebui.command(name="help")
    async def openweb_help(self, ctx: commands.Context, *, query: Optional[str] = None):
        """Interactive help for OpenWebUIChat. Optionally search with a query."""
        parts = [
            "OpenWebUIChat Help",
            "\nChat:",
            "- [p]llmchat <message>",
            "- /llmmodal (open a modal)",
            "\nAuto-Reply:",
            "- [p]openwebui autochannel add #channel",
            "- [p]openwebui autochannel remove #channel",
            "- [p]openwebui autochannel list",
            "- [p]openwebui autochannel mentiononly <true|false>",
            "\nHistory:",
            "- [p]openwebui history enable <true|false>",
            "- [p]openwebui history setmax <turns>",
            "- [p]openwebui history show | clear",
            "\nKnowledge Base:",
            "- [p]openwebui faq add <question> <answer>",
            "- [p]openwebui faq remove <question>",
            "- [p]openwebui faq list",
            "- [p]openwebui rules add <rule>",
            "- [p]openwebui rules remove <number>",
            "- [p]openwebui rules list",
            "- [p]openwebui projects add <name> <description>",
            "- [p]openwebui projects remove <name>",
            "- [p]openwebui projects list",
            "- [p]openwebui knowledge search <query>",
            "- [p]openwebui knowledge stats",
            "- [p]openwebui knowledge export | import",
            "\nTools:",
            "- [p]openwebui toolsguild add <tool_name>",
            "- [p]openwebui toolsguild remove <tool_name>",
            "- [p]openwebui toolsguild list",
            "- [p]openwebui toolschannel add <tool_name>",
            "- [p]openwebui toolschannel remove <tool_name>",
            "- [p]openwebui toolschannel list",
            "- [p]openwebui tools available",
            "\nModels:",
            "- [p]setopenwebui chatmodel <name>",
            "- [p]setopenwebui embedmodel <name>",
            "- [p]setopenwebui url <endpoint>",
            "- [p]setopenwebui key <key>",
        ]
        base_text = "\n".join(parts)
        if not query:
            for page in pagify(base_text):
                await ctx.send(page)
            return
        # Simple search (case-insensitive substring) across help text
        q = query.lower()
        hits = [line for line in base_text.splitlines() if q in line.lower()]
        if not hits:
            return await ctx.send(f"No help entries match '{query}'.")
        for page in pagify("\n".join(hits)):
            await ctx.send(page)

    @history.command(name="enable")
    async def history_enable(self, ctx: commands.Context, value: bool):
        await self.config.channel(ctx.channel).history_enabled.set(value)
        await ctx.send(f"History enabled set to {value} for this channel.")

    @history.command(name="setmax")
    async def history_setmax(self, ctx: commands.Context, turns: int):
        if turns < 0:
            return await ctx.send("Turns must be 0 or greater.")
        await self.config.channel(ctx.channel).history_max_turns.set(turns)
        await ctx.send(f"Max history turns set to {turns} for this channel.")

    @history.command(name="clear")
    async def history_clear(self, ctx: commands.Context):
        await self.config.channel(ctx.channel).history.set([])
        await ctx.send("🧹 Cleared history for this channel.")

    @history.command(name="show")
    async def history_show(self, ctx: commands.Context):
        hist = await self.config.channel(ctx.channel).history()
        if not hist:
            return await ctx.send("No history for this channel.")
        text = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in hist)
        for page in pagify(text):
            await ctx.send(page)

    # ───────────────── guild knowledge base management ─────────────────
    @openwebui.group(name="knowledge")
    @commands.is_owner()
    async def knowledge(self, ctx: commands.Context):
        """Manage guild knowledge bases (FAQs, house rules, project docs)."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebui.group(name="faq")
    @commands.is_owner()
    async def faq(self, ctx: commands.Context):
        """Manage server FAQs."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @faq.command(name="add")
    async def faq_add(self, ctx: commands.Context, question: str, *, answer: str):
        """Add a FAQ entry to the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        guild_kb.faqs[question] = answer
        guild_kb.updated_at = datetime.now()
        
        # Also add to memory entries for better retrieval
        faq_text = f"FAQ: {question} - {answer}"
        await self.add_memory_entry(faq_text, MemoryScope.GUILD, {
            "type": "faq",
            "guild_id": ctx.guild.id,
            "question": question,
            "answer": answer
        })
        
        await ctx.send(f"✅ Added FAQ: **{question}**")

    @faq.command(name="remove")
    async def faq_remove(self, ctx: commands.Context, *, question: str):
        """Remove a FAQ entry from the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        if question not in guild_kb.faqs:
            return await ctx.send("FAQ not found.")
        
        del guild_kb.faqs[question]
        guild_kb.updated_at = datetime.now()
        
        # Remove from memory entries
        to_remove = []
        for memory_id, entry in self.memory_entries.items():
            if (entry.metadata.get("type") == "faq" and 
                entry.metadata.get("guild_id") == ctx.guild.id and
                entry.metadata.get("question") == question):
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self.memory_entries[memory_id]
        
        await ctx.send(f"❌ Removed FAQ: **{question}**")

    @faq.command(name="list")
    async def faq_list(self, ctx: commands.Context):
        """List all FAQs in the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        if not guild_kb.faqs:
            return await ctx.send("No FAQs found in this server's knowledge base.")
        
        faq_list = []
        for i, (question, answer) in enumerate(guild_kb.faqs.items(), 1):
            faq_list.append(f"**{i}.** {question}\n   {answer[:200]}{'...' if len(answer) > 200 else ''}")
        
        text = "**Server FAQs:**\n\n" + "\n\n".join(faq_list)
        for page in pagify(text):
            await ctx.send(page)

    # Nested under knowledge for intuitive usage
    @knowledge.group(name="faq")
    @commands.is_owner()
    async def knowledge_faq(self, ctx: commands.Context):
        """Manage server FAQs (under knowledge)."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @knowledge_faq.command(name="add")
    async def knowledge_faq_add(self, ctx: commands.Context, question: str, *, answer: str):
        return await self.faq_add(ctx, question, answer=answer)

    @knowledge_faq.command(name="remove")
    async def knowledge_faq_remove(self, ctx: commands.Context, *, question: str):
        return await self.faq_remove(ctx, question=question)

    @knowledge_faq.command(name="list")
    async def knowledge_faq_list(self, ctx: commands.Context):
        return await self.faq_list(ctx)

    @openwebui.group(name="rules")
    @commands.is_owner()
    async def rules(self, ctx: commands.Context):
        """Manage server house rules."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @rules.command(name="add")
    async def rules_add(self, ctx: commands.Context, *, rule: str):
        """Add a house rule to the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        guild_kb.house_rules.append(rule)
        guild_kb.updated_at = datetime.now()
        
        # Add to memory entries
        rule_text = f"House Rule: {rule}"
        await self.add_memory_entry(rule_text, MemoryScope.GUILD, {
            "type": "house_rule",
            "guild_id": ctx.guild.id,
            "rule": rule
        })
        
        await ctx.send(f"✅ Added house rule: **{rule}**")

    @rules.command(name="remove")
    async def rules_remove(self, ctx: commands.Context, rule_number: int):
        """Remove a house rule by number."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        if not guild_kb.house_rules:
            return await ctx.send("No house rules found.")
        
        if rule_number < 1 or rule_number > len(guild_kb.house_rules):
            return await ctx.send(f"Invalid rule number. Please use 1-{len(guild_kb.house_rules)}")
        
        removed_rule = guild_kb.house_rules.pop(rule_number - 1)
        guild_kb.updated_at = datetime.now()
        
        # Remove from memory entries
        to_remove = []
        for memory_id, entry in self.memory_entries.items():
            if (entry.metadata.get("type") == "house_rule" and 
                entry.metadata.get("guild_id") == ctx.guild.id and
                entry.metadata.get("rule") == removed_rule):
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self.memory_entries[memory_id]
        
        await ctx.send(f"❌ Removed house rule: **{removed_rule}**")

    @rules.command(name="list")
    async def rules_list(self, ctx: commands.Context):
        """List all house rules in the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        if not guild_kb.house_rules:
            return await ctx.send("No house rules found in this server's knowledge base.")
        
        rules_list = []
        for i, rule in enumerate(guild_kb.house_rules, 1):
            rules_list.append(f"**{i}.** {rule}")
        
        text = "**Server House Rules:**\n\n" + "\n".join(rules_list)
        for page in pagify(text):
            await ctx.send(page)

    # Nested under knowledge for intuitive usage
    @knowledge.group(name="rules")
    @commands.is_owner()
    async def knowledge_rules(self, ctx: commands.Context):
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @knowledge_rules.command(name="add")
    async def knowledge_rules_add(self, ctx: commands.Context, *, rule: str):
        return await self.rules_add(ctx, rule=rule)

    @knowledge_rules.command(name="remove")
    async def knowledge_rules_remove(self, ctx: commands.Context, rule_number: int):
        return await self.rules_remove(ctx, rule_number=rule_number)

    @knowledge_rules.command(name="list")
    async def knowledge_rules_list(self, ctx: commands.Context):
        return await self.rules_list(ctx)

    @openwebui.group(name="projects")
    @commands.is_owner()
    async def projects(self, ctx: commands.Context):
        """Manage project documentation."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @projects.command(name="add")
    async def projects_add(self, ctx: commands.Context, project_name: str, *, description: str):
        """Add project documentation to the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        guild_kb.project_docs[project_name] = description
        guild_kb.updated_at = datetime.now()
        
        # Add to memory entries
        project_text = f"Project: {project_name} - {description}"
        await self.add_memory_entry(project_text, MemoryScope.GUILD, {
            "type": "project_doc",
            "guild_id": ctx.guild.id,
            "project_name": project_name,
            "description": description
        })
        
        await ctx.send(f"✅ Added project documentation: **{project_name}**")

    @projects.command(name="remove")
    async def projects_remove(self, ctx: commands.Context, *, project_name: str):
        """Remove project documentation from the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        if project_name not in guild_kb.project_docs:
            return await ctx.send("Project not found.")
        
        del guild_kb.project_docs[project_name]
        guild_kb.updated_at = datetime.now()
        
        # Remove from memory entries
        to_remove = []
        for memory_id, entry in self.memory_entries.items():
            if (entry.metadata.get("type") == "project_doc" and 
                entry.metadata.get("guild_id") == ctx.guild.id and
                entry.metadata.get("project_name") == project_name):
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self.memory_entries[memory_id]
        
        await ctx.send(f"❌ Removed project documentation: **{project_name}**")

    @projects.command(name="list")
    async def projects_list(self, ctx: commands.Context):
        """List all project documentation in the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        if not guild_kb.project_docs:
            return await ctx.send("No project documentation found in this server's knowledge base.")
        
        projects_list = []
        for project_name, description in guild_kb.project_docs.items():
            projects_list.append(f"**{project_name}**\n   {description[:200]}{'...' if len(description) > 200 else ''}")
        
        text = "**Project Documentation:**\n\n" + "\n\n".join(projects_list)
        for page in pagify(text):
            await ctx.send(page)

    # Nested under knowledge for intuitive usage
    @knowledge.group(name="projects")
    @commands.is_owner()
    async def knowledge_projects(self, ctx: commands.Context):
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @knowledge_projects.command(name="add")
    async def knowledge_projects_add(self, ctx: commands.Context, project_name: str, *, description: str):
        return await self.projects_add(ctx, project_name=project_name, description=description)

    @knowledge_projects.command(name="remove")
    async def knowledge_projects_remove(self, ctx: commands.Context, *, project_name: str):
        return await self.projects_remove(ctx, project_name=project_name)

    @knowledge_projects.command(name="list")
    async def knowledge_projects_list(self, ctx: commands.Context):
        return await self.projects_list(ctx)

    @knowledge.command(name="search")
    async def knowledge_search(self, ctx: commands.Context, *, query: str):
        """Search the guild knowledge base for relevant information."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        # Search guild-specific memories
        results = await self.search_memories(query, MemoryScope.GUILD, limit=5)
        
        if not results:
            return await ctx.send("No relevant information found in the server's knowledge base.")
        
        search_results = []
        for i, entry in enumerate(results, 1):
            result_type = entry.metadata.get("type", "unknown")
            if result_type == "faq":
                question = entry.metadata.get("question", "Unknown")
                answer = entry.metadata.get("answer", entry.text)
                search_results.append(f"**{i}.** FAQ: {question}\n   {answer[:200]}{'...' if len(answer) > 200 else ''}")
            elif result_type == "house_rule":
                rule = entry.metadata.get("rule", entry.text)
                search_results.append(f"**{i}.** House Rule: {rule}")
            elif result_type == "project_doc":
                project_name = entry.metadata.get("project_name", "Unknown")
                description = entry.metadata.get("description", entry.text)
                search_results.append(f"**{i}.** Project: {project_name}\n   {description[:200]}{'...' if len(description) > 200 else ''}")
            else:
                search_results.append(f"**{i}.** {entry.text[:200]}{'...' if len(entry.text) > 200 else ''}")
        
        text = f"**Search results for '{query}':**\n\n" + "\n\n".join(search_results)
        for page in pagify(text):
            await ctx.send(page)

    @knowledge.command(name="stats")
    async def knowledge_stats(self, ctx: commands.Context):
        """Show statistics about the guild knowledge base."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        
        # Count memory entries for this guild
        guild_memories = [entry for entry in self.memory_entries.values() 
                         if entry.metadata.get("guild_id") == ctx.guild.id]
        
        memory_types = {}
        for entry in guild_memories:
            entry_type = entry.metadata.get("type", "other")
            memory_types[entry_type] = memory_types.get(entry_type, 0) + 1
        
        embed = discord.Embed(
            title=f"Knowledge Base Statistics - {ctx.guild.name}",
            color=discord.Color.blurple()
        )
        
        embed.add_field(
            name="📋 FAQs",
            value=str(len(guild_kb.faqs)),
            inline=True
        )
        embed.add_field(
            name="📜 House Rules",
            value=str(len(guild_kb.house_rules)),
            inline=True
        )
        embed.add_field(
            name="📁 Projects",
            value=str(len(guild_kb.project_docs)),
            inline=True
        )
        
        embed.add_field(
            name="🧠 Memory Entries",
            value=str(len(guild_memories)),
            inline=True
        )
        embed.add_field(
            name="📊 Entry Types",
            value="\n".join(f"{k}: {v}" for k, v in memory_types.items()) if memory_types else "None",
            inline=True
        )
        embed.add_field(
            name="🕒 Last Updated",
            value=guild_kb.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            inline=True
        )
        
        await ctx.send(embed=embed)

    @knowledge.command(name="export")
    async def knowledge_export(self, ctx: commands.Context):
        """Export the guild knowledge base as a JSON file."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
        
        export_data = {
            "guild_id": ctx.guild.id,
            "guild_name": ctx.guild.name,
            "exported_at": datetime.now().isoformat(),
            "faqs": guild_kb.faqs,
            "house_rules": guild_kb.house_rules,
            "project_docs": guild_kb.project_docs,
            "metadata": {
                "created_at": guild_kb.created_at.isoformat(),
                "updated_at": guild_kb.updated_at.isoformat(),
                "scope": guild_kb.scope.value
            }
        }
        
        # Create a temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f, indent=2)
            temp_path = f.name
        
        try:
            # Send the file
            with open(temp_path, 'rb') as f:
                file_obj = discord.File(f, filename=f"{ctx.guild.name}_knowledge_base.json")
                await ctx.send("📁 Guild knowledge base exported:", file=file_obj)
        finally:
            # Clean up
            os.unlink(temp_path)

    @knowledge.command(name="import")
    async def knowledge_import(self, ctx: commands.Context):
        """Import a guild knowledge base from a JSON file."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        if not ctx.message.attachments:
            return await ctx.send("Please attach a JSON file to import.")
        
        attachment = ctx.message.attachments[0]
        if not attachment.filename.endswith('.json'):
            return await ctx.send("Please attach a JSON file.")
        
        try:
            # Download and parse the file
            content = await attachment.read()
            import_data = json.loads(content.decode('utf-8'))
            
            # Validate the data structure
            if not all(key in import_data for key in ['faqs', 'house_rules', 'project_docs']):
                return await ctx.send("Invalid knowledge base file format.")
            
            guild_kb = await self.get_guild_knowledge_base(ctx.guild.id)
            
            # Import the data
            imported_count = 0
            
            # Import FAQs
            for question, answer in import_data['faqs'].items():
                guild_kb.faqs[question] = answer
                faq_text = f"FAQ: {question} - {answer}"
                await self.add_memory_entry(faq_text, MemoryScope.GUILD, {
                    "type": "faq",
                    "guild_id": ctx.guild.id,
                    "question": question,
                    "answer": answer
                })
                imported_count += 1
            
            # Import house rules
            for rule in import_data['house_rules']:
                guild_kb.house_rules.append(rule)
                rule_text = f"House Rule: {rule}"
                await self.add_memory_entry(rule_text, MemoryScope.GUILD, {
                    "type": "house_rule",
                    "guild_id": ctx.guild.id,
                    "rule": rule
                })
                imported_count += 1
            
            # Import project docs
            for project_name, description in import_data['project_docs'].items():
                guild_kb.project_docs[project_name] = description
                project_text = f"Project: {project_name} - {description}"
                await self.add_memory_entry(project_text, MemoryScope.GUILD, {
                    "type": "project_doc",
                    "guild_id": ctx.guild.id,
                    "project_name": project_name,
                    "description": description
                })
                imported_count += 1
            
            guild_kb.updated_at = datetime.now()
            
            await ctx.send(f"✅ Successfully imported {imported_count} knowledge base entries from {attachment.filename}")
            
        except json.JSONDecodeError:
            await ctx.send("❌ Invalid JSON file format.")
        except Exception as e:
            await ctx.send(f"❌ Error importing knowledge base: {str(e)}")

    # ───────────────── tool management ─────────────────
    @openwebui.group(name="tools")
    @commands.is_owner()
    async def tools(self, ctx: commands.Context):
        """Manage tool allowlists and permissions."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @openwebui.group(name="toolsguild")
    @commands.is_owner()
    async def toolsguild(self, ctx: commands.Context):
        """Manage guild-level tool allowlists."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @toolsguild.command(name="add")
    async def tools_guild_add(self, ctx: commands.Context, tool_name: str):
        """Add a tool to the guild allowlist."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        if tool_name not in self.tools:
            available_tools = ", ".join(self.tools.keys())
            return await ctx.send(f"❌ Unknown tool. Available tools: {available_tools}")
        
        guild_conf = self.config.guild(ctx.guild)
        allowed_tools = await guild_conf.allowed_tools()
        if tool_name not in allowed_tools:
            allowed_tools.append(tool_name)
            await guild_conf.allowed_tools.set(allowed_tools)
            await ctx.send(f"✅ Added **{tool_name}** to guild tool allowlist.")
        else:
            await ctx.send(f"**{tool_name}** is already in the guild allowlist.")

    @toolsguild.command(name="remove")
    async def tools_guild_remove(self, ctx: commands.Context, tool_name: str):
        """Remove a tool from the guild allowlist."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_conf = self.config.guild(ctx.guild)
        allowed_tools = await guild_conf.allowed_tools()
        if tool_name in allowed_tools:
            allowed_tools.remove(tool_name)
            await guild_conf.allowed_tools.set(allowed_tools)
            await ctx.send(f"❌ Removed **{tool_name}** from guild tool allowlist.")
        else:
            await ctx.send(f"**{tool_name}** is not in the guild allowlist.")

    @toolsguild.command(name="list")
    async def tools_guild_list(self, ctx: commands.Context):
        """List guild tool allowlist."""
        if not ctx.guild:
            return await ctx.send("This command can only be used in a server.")
        
        guild_conf = self.config.guild(ctx.guild)
        allowed_tools = await guild_conf.allowed_tools()
        
        if not allowed_tools:
            await ctx.send("No tools are specifically allowed in this guild. All tools are available.")
        else:
            tool_list = []
            for tool_name in allowed_tools:
                tool_info = self.tools.get(tool_name, {})
                description = tool_info.get("description", "No description")
                tool_list.append(f"**{tool_name}**: {description}")
            
            embed = discord.Embed(
                title=f"Guild Tool Allowlist - {ctx.guild.name}",
                description="\n".join(tool_list),
                color=discord.Color.blurple()
            )
            await ctx.send(embed=embed)

    @openwebui.group(name="toolschannel")
    @commands.is_owner()
    async def toolschannel(self, ctx: commands.Context):
        """Manage channel-level tool allowlists."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @toolschannel.command(name="add")
    async def tools_channel_add(self, ctx: commands.Context, tool_name: str):
        """Add a tool to the channel allowlist."""
        if tool_name not in self.tools:
            available_tools = ", ".join(self.tools.keys())
            return await ctx.send(f"❌ Unknown tool. Available tools: {available_tools}")
        
        channel_conf = self.config.channel(ctx.channel)
        allowed_tools = await channel_conf.allowed_tools()
        if tool_name not in allowed_tools:
            allowed_tools.append(tool_name)
            await channel_conf.allowed_tools.set(allowed_tools)
            await ctx.send(f"✅ Added **{tool_name}** to channel tool allowlist.")
        else:
            await ctx.send(f"**{tool_name}** is already in the channel allowlist.")

    @toolschannel.command(name="remove")
    async def tools_channel_remove(self, ctx: commands.Context, tool_name: str):
        """Remove a tool from the channel allowlist."""
        channel_conf = self.config.channel(ctx.channel)
        allowed_tools = await channel_conf.allowed_tools()
        if tool_name in allowed_tools:
            allowed_tools.remove(tool_name)
            await channel_conf.allowed_tools.set(allowed_tools)
            await ctx.send(f"❌ Removed **{tool_name}** from channel tool allowlist.")
        else:
            await ctx.send(f"**{tool_name}** is not in the channel allowlist.")

    @toolschannel.command(name="list")
    async def tools_channel_list(self, ctx: commands.Context):
        """List channel tool allowlist."""
        channel_conf = self.config.channel(ctx.channel)
        allowed_tools = await channel_conf.allowed_tools()
        
        if not allowed_tools:
            await ctx.send("No tools are specifically allowed in this channel. Guild settings apply.")
        else:
            tool_list = []
            for tool_name in allowed_tools:
                tool_info = self.tools.get(tool_name, {})
                description = tool_info.get("description", "No description")
                tool_list.append(f"**{tool_name}**: {description}")
            
            embed = discord.Embed(
                title=f"Channel Tool Allowlist - {ctx.channel.name}",
                description="\n".join(tool_list),
                color=discord.Color.blurple()
            )
            await ctx.send(embed=embed)

    @tools.command(name="available")
    async def tools_available(self, ctx: commands.Context):
        """List all available tools."""
        tool_list = []
        for tool_name, tool_info in self.tools.items():
            description = tool_info.get("description", "No description")
            category = tool_info.get("category", "general")
            tool_list.append(f"**{tool_name}** ({category}): {description}")
        
        embed = discord.Embed(
            title="Available Tools",
            description="\n".join(tool_list),
            color=discord.Color.blurple()
        )
        embed.add_field(
            name="Usage",
            value="Use `[p]openwebui toolsguild/toolschannel add <tool_name>` to allow tools",
            inline=False
        )
        await ctx.send(embed=embed)


async def setup(bot: Red):
    await bot.add_cog(OpenWebUIMemoryBot(bot))
