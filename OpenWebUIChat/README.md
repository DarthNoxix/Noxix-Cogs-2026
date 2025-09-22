# OpenWebUI Chat Bot - Advanced AI Assistant

A comprehensive Discord bot cog that integrates with OpenWebUI to provide an advanced AI assistant with memory, per-guild knowledge bases, tool integration, prompt templates, and intelligent conversation management.

## 🚀 What You Get

- Per-user memory profiles (short-term + preferences)
- Per-guild knowledge bases (FAQs, house rules, projects)
- Multi-store embeddings with provider selection and fallback
- Message-aware system prompts and response modes
- Tool-calling layer with guild/channel allowlists
- Memory TTL, decay, and archiving
- Background tasks: cleanup, health checks, backups, analytics
- Prompt templates per-channel and sources toggle

## 📋 Requirements

### Dependencies
- httpx
- numpy
- rank-bm25
- aiofiles

Install:
```bash
pip install httpx numpy rank-bm25 aiofiles
```

### OpenWebUI Setup
- Run OpenWebUI and ensure you have chat + embedding models available
- Note your API endpoint (e.g., `http://localhost:8080`)

## ⚙️ Installation

1) Copy this cog to your Red-DiscordBot cogs directory
2) Load the cog:
```
[p]load OpenWebUIChat
```

## 🎯 Quick Start (Essential Setup)

1) Configure connection and models
```
[p]setopenwebui url http://localhost:8080
[p]setopenwebui key your-api-key   # if required
[p]setopenwebui chatmodel deepseek-r1:8b
[p]setopenwebui embedmodel bge-large-en-v1.5
```

2) Choose embedding provider (optional)
```
[p]setopenwebui embedproviders
[p]setopenwebui embedprovider openwebui
```

3) Start chatting
```
[p]llmchat Hello! How are you today?
```

## 🧠 Knowledge Base (Per-Guild)

All commands below are server-only (won't work in DMs).

### FAQs
```
[p]openwebui faq add "How do I get help?" "Use the support channel or ping moderators"
[p]openwebui faq remove "How do I get help?"
[p]openwebui faq list
```

### House Rules
```
[p]openwebui rules add "Be respectful to all members"
[p]openwebui rules remove 1
[p]openwebui rules list
```

### Projects
```
[p]openwebui projects add "MyBot" "A Discord bot for server management"
[p]openwebui projects remove "MyBot"
[p]openwebui projects list
```

### Knowledge Ops
```
[p]openwebui knowledge search "rules"
[p]openwebui knowledge stats
[p]openwebui knowledge export
[p]openwebui knowledge import   # attach JSON file
```

Tips:
- Adding FAQs/Rules/Projects also creates structured memory entries for better retrieval
- `knowledge stats` shows KB size and memory types

## 🛠️ Tool Management

Tools are available but must be allowed per-guild or per-channel as needed.

### Inspect
```
[p]openwebui tools available
```

### Guild allowlist
```
[p]openwebui toolsguild add web_search
[p]openwebui toolsguild remove web_search
[p]openwebui toolsguild list
```

### Channel allowlist
```
[p]openwebui toolschannel add code_lint
[p]openwebui toolschannel remove code_lint
[p]openwebui toolschannel list
```

Notes:
- Guild list applies to the whole server unless a channel list is present
- Channel list overrides guild list for that channel

## 🧩 Prompt Templates, Modes, and Sources

Per-channel settings to shape answers.

```
[p]prompt set You are Alicent. Be concise and helpful.
[p]prompt show
[p]prompt clear
[p]prompt mode show_work     # normal | show_work | snippet | thread
[p]prompt sources true       # show a "Sources" card when available
```

Modes:
- normal: default
- show_work: safe transparency (no chain-of-thought), adds sources card if enabled
- snippet: bias toward code snippets/technical answers
- thread: longer, threaded-discussion style responses

## 🗂️ Memory System

### Add and list (legacy KB)
```
[p]openwebuimemory add "topic" "Helpful text"
[p]openwebuimemory list
```

### Maintenance & TTL
```
[p]openwebuimemory stats
[p]openwebuimemory cleanup
[p]openwebuimemory ttl 24
```

How it works:
- Short-term user memory is kept per user and refreshed on activity
- Memories have TTL (default 24h) and decay; valuable entries are archived
- Hybrid retrieval (embeddings + BM25) selects the best context

## 🤖 Auto-Reply Channels (Owner only)
```
[p]openwebui autochannel add #channel
[p]openwebui autochannel list
[p]openwebui autochannel mentiononly true
[p]openwebui autochannel remove #channel
```
- If mention-only is true, the bot replies only when mentioned in those channels

## 🧪 Testing Flow (Suggested)

1) Knowledge base
```
[p]openwebui faq add "What is this server about?" "A gaming community"
[p]openwebui knowledge search "server"
[p]openwebui knowledge stats
```

2) Tools
```
[p]openwebui tools available
[p]openwebui toolsguild add web_search
[p]llmchat Search for news about Discord bots
```

3) Memory & TTL
```
[p]openwebuimemory add "pref" "I like detailed explanations"
[p]openwebuimemory stats
[p]llmchat What are my preferences?
```

4) Prompts & Sources
```
[p]prompt set You are Alicent. Be concise and helpful.
[p]prompt mode show_work
[p]prompt sources true
[p]llmchat Summarize the server rules
```

## 🔧 Configuration Reference

```
[p]setopenwebui url <url>
[p]setopenwebui key <key>
[p]setopenwebui chatmodel <model>
[p]setopenwebui embedmodel <model>
[p]setopenwebui embedprovider <provider>
[p]setopenwebui embedproviders
```
Providers: openwebui, ollama, faiss, chroma, pgvector

## 🧠 How It Works (Under the Hood)

- Persona-aware prompts with per-channel templates and response modes
- Memory retrieval blends dense embeddings (cosine) and BM25 sparse scores
- Per-scope memories: user, channel, guild, global
- Background jobs: hourly cleanup (TTL/decay/archive), health checks, backups, analytics
- Guardrails: simple toxicity filter with safe fallback
- Sources: optional embed listing knowledge snippets used

## 🚨 Troubleshooting

- "OpenWebUI URL or key not set":
  - `[p]setopenwebui url <url>`, `[p]setopenwebui key <key>`
- Memory retrieval weak:
  - Check embedding model/provider config, verify OpenWebUI up
- Tools not appearing/working:
  - `[p]openwebui tools available`, ensure guild/channel allowlists set
- Auto-reply not triggering:
  - Ensure channel is added and mention-only is off or you mention the bot

## 📁 Files

```
OpenWebUIChat/
├── __init__.py
├── info.json
├── openwebuichat.py
└── README.md
```

## 📄 License

This cog is part of the Noxix-Cogs collection. See the main repository for licensing.