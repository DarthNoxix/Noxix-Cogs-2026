# OpenWebUI Chat Bot - Advanced AI Assistant

A comprehensive Discord bot cog that integrates with OpenWebUI to provide an advanced AI assistant with memory, knowledge bases, tool integration, and intelligent conversation management.

## 🚀 **NEW: Advanced Features Implemented**

### **🧠 Per-User Memory Profiles**
- **Short-term conversation window** with automatic TTL management
- **Long-term user profiles** with preferences and task logs
- **Personalized responses** based on user history and preferences
- **Memory decay system** that archives valuable memories automatically

### **📚 Per-Guild Knowledge Bases**
- **Server FAQs** with semantic search and automatic retrieval
- **House Rules** management with scoped access
- **Project Documentation** with structured storage
- **Export/Import** functionality for knowledge base backup
- **Statistics dashboard** with usage analytics

### **🔧 Multi-Store Embeddings**
- **Provider Selection**: OpenWebUI, Ollama, FAISS, Chroma, PGVector
- **Automatic Fallback** to OpenWebUI if other providers fail
- **Embedding Caching** to reduce API calls and improve performance
- **Hot-swappable providers** with runtime configuration

### **⚡ Message-Aware System Prompts**
- **Context Detection**: Automatically detects channel type, threads, voice channels
- **Dynamic Prompting**: Adjusts responses based on message characteristics
- **Persona Integration**: Works with all persona types (Alicent, Court Scribe, etc.)
- **Response Modes**: Show work, snippet mode, thread mode integration

### **🛠️ Tool-Calling Layer with Allowlists**
- **Guild-Level Controls**: Manage tool permissions per server
- **Channel-Level Controls**: Fine-grained tool access per channel
- **Tool Categories**: Web search, code linting, image generation, math solving
- **Permission System**: Hierarchical allowlists with inheritance

### **⏰ Memory TTLs & Decay**
- **Auto-Archiving**: Valuable memories are archived instead of deleted
- **Memory Decay**: Rarely accessed memories get shorter TTLs
- **Background Cleanup**: Hourly maintenance with intelligent archiving
- **Long-term Storage**: SQLite database for persistent memory management

## 📋 Requirements

### Dependencies
- `httpx` - HTTP client for OpenWebUI API calls
- `numpy` - Numerical computations for embeddings
- `rank-bm25` - BM25 sparse retrieval algorithm
- `aiofiles` - Async file operations for data persistence

### OpenWebUI Setup
1. Install and run OpenWebUI
2. Ensure you have chat and embedding models available
3. Note your OpenWebUI API endpoint (typically `http://localhost:8080`)

## ⚙️ Installation

1. **Copy the cog** to your Red-DiscordBot cogs directory
2. **Install dependencies**:
   ```bash
   pip install httpx numpy rank-bm25 aiofiles
   ```
3. **Load the cog**:
   ```
   [p]load OpenWebUIChat
   ```

## 🎯 Quick Start

### Basic Setup
1. **Configure OpenWebUI endpoint**:
   ```
   [p]setopenwebui url http://localhost:8080
   ```

2. **Set your API key** (if required):
   ```
   [p]setopenwebui key your-api-key
   ```

3. **Configure models**:
   ```
   [p]setopenwebui chatmodel deepseek-r1:8b
   [p]setopenwebui embedmodel bge-large-en-v1.5
   ```

4. **Set embedding provider**:
   ```
   [p]setopenwebui embedprovider openwebui
   [p]setopenwebui embedproviders  # List available providers
   ```

5. **Start chatting**:
   ```
   [p]llmchat Hello! How are you today?
   ```

## 📚 Commands

### Chat Commands

#### `[p]llmchat [message]`
Chat with the AI assistant with full memory and tool integration.

#### `/llmmodal`
Open a modal for long prompts with enhanced context awareness.

### **NEW: Knowledge Base Management**

#### FAQ Management
```
[p]openwebui faq add "How do I get help?" "Use the support channel or ping moderators"
[p]openwebui faq remove "How do I get help?"
[p]openwebui faq list
```

#### House Rules Management
```
[p]openwebui rules add "Be respectful to all members"
[p]openwebui rules remove 1
[p]openwebui rules list
```

#### Project Documentation
```
[p]openwebui projects add "MyBot" "A Discord bot for server management"
[p]openwebui projects remove "MyBot"
[p]openwebui projects list
```

#### Knowledge Base Operations
```
[p]openwebui knowledge search "help"
[p]openwebui knowledge stats
[p]openwebui knowledge export
[p]openwebui knowledge import  # Attach JSON file
```

### **NEW: Tool Management**

#### Guild Tool Controls
```
[p]openwebui toolsguild add web_search
[p]openwebui toolsguild remove web_search
[p]openwebui toolsguild list
```

#### Channel Tool Controls
```
[p]openwebui toolschannel add code_lint
[p]openwebui toolschannel remove code_lint
[p]openwebui toolschannel list
```

#### Available Tools
```
[p]openwebui tools available  # List all tools with descriptions
```

### **NEW: Memory System Management**

#### Memory Statistics & Maintenance
```
[p]openwebuimemory stats      # View memory system statistics
[p]openwebuimemory cleanup    # Manual cleanup and archiving
[p]openwebuimemory ttl 24     # Set memory TTL in hours
```

### **NEW: Embedding Provider Management**

#### Provider Configuration
```
[p]setopenwebui embedprovider openwebui    # Set provider
[p]setopenwebui embedproviders             # List available providers
```

### Auto-Reply (Owner Only)

#### `[p]openwebui autochannel add <#channel>`
Enable auto replies in a channel with enhanced context awareness.

#### `[p]openwebui autochannel remove <#channel>`
Disable auto replies in a channel.

#### `[p]openwebui autochannel list`
List configured auto-reply channels.

#### `[p]openwebui autochannel mentiononly <true|false>`
If true, only reply when the bot is mentioned in those channels.

### Configuration Commands (Owner Only)

#### `[p]setopenwebui url <url>`
Set the OpenWebUI API endpoint.

#### `[p]setopenwebui key <key>`
Set the API key for authentication.

#### `[p]setopenwebui chatmodel <model>`
Set the chat model to use.

#### `[p]setopenwebui embedmodel <model>`
Set the embedding model for memory system.

### Legacy Memory Management (Owner Only)

#### `[p]openwebuimemory add <name> <text>`
Add a memory to the knowledge base.

#### `[p]openwebuimemory list`
List all memories in the knowledge base.

#### `[p]openwebuimemory del <name>`
Delete a memory from the knowledge base.

## 🧠 How It Works

### **Enhanced Chat with Memory Integration**
- **User Profiles**: Remembers user preferences and conversation history
- **Guild Knowledge**: Automatically includes relevant FAQs, rules, and project docs
- **Context Awareness**: Adapts responses based on channel type and message characteristics
- **Tool Integration**: Automatically uses appropriate tools based on question content

### **Advanced Memory System**
- **Hybrid Retrieval**: Combines dense (embeddings) and sparse (BM25) search
- **Scoped Access**: User, channel, guild, and global memory scopes
- **TTL Management**: Automatic cleanup with intelligent archiving
- **Multi-Provider**: Support for multiple embedding providers with fallback

### **Tool-Calling System**
- **Smart Detection**: Automatically determines which tools to use
- **Permission Layers**: Guild → Channel → User allowlist hierarchy
- **Safe Execution**: Tool results are integrated into AI responses
- **Extensible**: Easy to add new tools and categories

## 🔧 Configuration

### Recommended Models
- **Chat Models**: `deepseek-r1:8b`, `gpt-4o`, `claude-3-sonnet`, `llama-3.1-8b`
- **Embedding Models**: `bge-large-en-v1.5`, `text-embedding-3-large`, `nomic-embed-text`

### **NEW: Embedding Providers**
- **OpenWebUI**: Default provider, uses your OpenWebUI instance
- **Ollama**: Direct Ollama integration for local embeddings
- **FAISS**: Local FAISS index (placeholder implementation)
- **Chroma**: ChromaDB integration (placeholder implementation)
- **PGVector**: PostgreSQL with pgvector extension (placeholder implementation)

### Memory Settings
- **Similarity Threshold**: 0.8 (configurable in code)
- **Top K Results**: 9 memories maximum per query
- **Hybrid Weights**: 70% dense, 30% sparse retrieval
- **TTL Default**: 24 hours (configurable)
- **Auto-Archive**: Memories with >5 accesses or special types

## 🚨 Troubleshooting

### Common Issues

#### "OpenWebUI URL or key not set"
- Set the endpoint: `[p]setopenwebui url <your-openwebui-url>`
- Set the API key if required: `[p]setopenwebui key <your-key>`

#### "Failed to retrieve memories"
- Check your embedding model: `[p]setopenwebui embedmodel <model>`
- Check embedding provider: `[p]setopenwebui embedprovider <provider>`
- Verify OpenWebUI is running and accessible
- The bot will continue with general chat even if memory retrieval fails

#### No response from chat
- Verify OpenWebUI is running
- Check your chat model: `[p]setopenwebui chatmodel <model>`
- Ensure the API endpoint is correct

#### Tool not working
- Check tool allowlists: `[p]openwebui tools guild list`
- Verify tool is available: `[p]openwebui tools available`
- Check channel permissions: `[p]openwebui tools channel list`

### Debug Commands
```
[p]setopenwebui url http://localhost:8080     # Check current URL
[p]openwebuimemory stats                      # Check memory system status
[p]openwebui knowledge stats                  # Check knowledge base status
[p]openwebui tools available                  # List available tools
```

## 📁 File Structure

```
OpenWebUIChat/
├── __init__.py          # Cog setup
├── info.json           # Cog metadata with dependencies
├── openwebuichat.py    # Main implementation (2600+ lines)
└── README.md           # This file
```

## 🎯 **Testing the New Features**

### 1. **Test Knowledge Base**
```
[p]openwebui faq add "What is this server about?" "This is a gaming community server"
[p]openwebui knowledge search "server"
[p]openwebui knowledge stats
```

### 2. **Test Tool System**
```
[p]openwebui tools available
[p]openwebui toolsguild add web_search
[p]llmchat "Search for the latest news about AI"
```

### 3. **Test Memory System**
```
[p]openwebuimemory stats
[p]openwebuimemory cleanup
[p]llmchat "Remember that I prefer detailed explanations"
```

### 4. **Test Embedding Providers**
```
[p]setopenwebui embedproviders
[p]setopenwebui embedprovider ollama
[p]openwebuimemory add "test" "This is a test memory"
```

## 🤝 Support

For issues or questions:
1. Check this README first
2. Verify your OpenWebUI setup
3. Check the bot logs for error messages
4. Ensure all dependencies are installed
5. Test with the debug commands above

## 📄 License

This cog is part of the Noxix-Cogs collection. Please refer to the main repository for licensing information.

---

**Ready to test your advanced AI assistant!** 🤖✨

**Next Steps**: After testing these features, we can continue implementing the remaining 40+ advanced features including response guardrails, file ingestion, web search tools, and much more!