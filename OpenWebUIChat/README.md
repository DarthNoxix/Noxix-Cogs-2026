# OpenWebUI Assistant Cog

A comprehensive Discord bot cog that integrates with OpenWebUI to provide advanced AI assistant capabilities, including conversation management, memory systems, function calling, and auto-responses. This cog is designed to work alongside the original Assistant cog without conflicts, providing a complete AI assistant experience powered by OpenWebUI.

## 🚀 Features

### Core Functionality
- **OpenWebUI Integration**: Full compatibility with OpenWebUI API endpoints for chat and embeddings
- **Conversation Management**: Persistent conversations with token tracking and message history
- **Memory System**: Advanced embedding-based memory with hybrid retrieval (dense + sparse)
- **Function Calling**: Extensible function system for AI interactions with custom functions
- **Auto-Response**: Intelligent message monitoring and responses with configurable triggers
- **TLDR Summarization**: Channel activity summarization for moderators with time-based filtering

### Advanced Features
- **Hybrid Retrieval**: Combines dense (cosine similarity) and sparse (BM25) retrieval for optimal memory recall
- **File Comprehension**: Reads and processes 30+ file types including code, configs, and documents
- **Interactive UI**: Discord UI components for memory and settings management with pagination
- **Data Management**: Import/export capabilities for conversations, embeddings, and configurations
- **Regex Blacklist**: Content filtering system with pattern matching and fail-blocking
- **Collaborative Conversations**: Shared conversations per channel for team collaboration
- **Channel-Specific Prompts**: Custom system prompts per channel for specialized behavior
- **Token Usage Tracking**: Detailed statistics on input/output tokens per user
- **Conversation Persistence**: Optional conversation saving across bot restarts
- **Multi-Model Support**: Support for various OpenWebUI models (chat, embedding, vision, function-calling)
- **Embedding Synchronization**: Automatic sync between in-memory and ChromaDB storage
- **Function Registry**: Dynamic function registration from other cogs
- **Advanced Configuration**: 50+ configurable parameters for fine-tuning behavior

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Red-DiscordBot**: Latest version
- **Discord.py**: 2.0 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended for large embedding collections)
- **Storage**: 1GB+ free space for ChromaDB and conversation storage

### Dependencies
- `chromadb` - Vector database for embeddings and similarity search
- `numpy` - Numerical computations for embedding operations
- `rank-bm25` - BM25 sparse retrieval algorithm implementation
- `pydantic` - Data validation and serialization for configuration models
- `orjson` - Fast JSON processing for conversation storage
- `httpx` - Async HTTP client for OpenWebUI API calls
- `discord.py` - Discord API wrapper (included with Red-DiscordBot)

### OpenWebUI Setup
1. **Install OpenWebUI**: Follow the official OpenWebUI installation guide
2. **Configure Models**: Ensure you have chat and embedding models available
3. **API Access**: Verify your OpenWebUI instance is accessible from your bot
4. **Endpoints**: Note the API endpoints (typically `/api/v1/chat/completions` and `/api/v1/embeddings`)
5. **Authentication**: Configure any required API keys or authentication
6. **CORS**: Ensure CORS is properly configured if running on different domains

### Recommended OpenWebUI Models
- **Chat Models**: `deepseek-r1:8b`, `gpt-4o`, `claude-3-sonnet`, `llama-3.1-8b`
- **Embedding Models**: `bge-large-en-v1.5`, `text-embedding-3-large`, `nomic-embed-text`
- **Vision Models**: `gpt-4o`, `claude-3-sonnet` (for image processing)
- **Function-Calling Models**: `gpt-4o`, `deepseek-r1:8b` (with function calling support)

## ⚙️ Installation

### Step-by-Step Installation

1. **Download the Cog**:
   - Copy the `OpenWebUIChat` folder to your Red-DiscordBot cogs directory
   - Ensure all files are present: `openwebuichat.py`, `abc.py`, `common/`, `views.py`, `info.json`

2. **Install Dependencies**:
   ```bash
   # Install all required packages
   pip install chromadb numpy rank-bm25 pydantic orjson httpx
   
   # Or install individually
   pip install chromadb>=0.4.0
   pip install numpy>=1.21.0
   pip install rank-bm25>=0.2.2
   pip install pydantic>=2.0.0
   pip install orjson>=3.8.0
   pip install httpx>=0.24.0
   ```

3. **Load the Cog**:
   ```
   [p]load OpenWebUIChat
   ```

4. **Verify Installation**:
   ```
   [p]openwebuihelp
   ```

### Post-Installation Setup

1. **Configure OpenWebUI Endpoint**:
   ```
   [p]openwebuiassistant endpoint http://localhost:8080
   ```

2. **Set Models**:
   ```
   [p]openwebuiassistant model deepseek-r1:8b
   [p]openwebuiassistant embedmodel bge-large-en-v1.5
   ```

3. **Test Basic Functionality**:
   ```
   [p]openwebuichat Hello! Test message.
   ```

### Installation Troubleshooting

**Common Issues:**
- **Import Errors**: Ensure all dependencies are installed correctly
- **Permission Errors**: Check file permissions in the cogs directory
- **Load Failures**: Check Red-DiscordBot logs for specific error messages
- **Missing Files**: Verify all cog files are present and not corrupted

**Verification Commands:**
```
[p]cogs list | grep OpenWebUIChat
[p]openwebuiassistant status
[p]openwebuihelp installation
```

## 🎯 Quick Start

### Basic Setup
1. **Configure API Endpoint**:
   ```
   [p]openwebuiassistant endpoint <your-openwebui-url>
   ```

2. **Set Models**:
   ```
   [p]openwebuiassistant model <model-name>
   [p]openwebuiassistant embedmodel <embedding-model>
   ```

3. **Enable Auto-Response**:
   ```
   [p]openwebuiassistant toggle true
   [p]openwebuiassistant channel #general
   ```

### First Chat
```
[p]openwebuichat Hello! How are you today?
```

## 📚 Command Reference

### Chat Commands

#### `[p]openwebuichat [message]`
Main chat command with the assistant.

**Arguments:**
- `--last` - Resend the last message of the conversation
- `--extract` - Extract code blocks from the reply
- `--outputfile <filename>` - Send reply as a file

**Examples:**
```
[p]openwebuichat Write a Python script to calculate fibonacci numbers
[p]openwebuichat --extract --outputfile script.py Write a Python script
[p]openwebuichat --last --outputfile response.txt
```

#### `[p]openwebuichathelp`
Get comprehensive help on using the OpenWebUI assistant.

### Conversation Management

#### `[p]openwebuiconvostats`
View conversation statistics (message count, token usage).

#### `[p]openwebuiclearconvo`
Reset your conversation for the current channel.

#### `[p]openwebuishowconvo`
Get a JSON dump of your current conversation.

#### `[p]openwebuiconvopop`
Remove the last message from the conversation.

#### `[p]openwebuiconvocopy <channel>`
Copy the current conversation to another channel.

#### `[p]openwebuiconvoprompt <prompt>`
Set a system prompt for the current conversation.

#### `[p]openwebuiimportconvo`
Import a conversation from a JSON file.

### Memory Management

#### `[p]openwebuimemory <name> <text>`
Add a memory/embedding to the assistant.

#### `[p]openwebuimemories`
Open an interactive memory viewer.

#### `[p]openwebuiquery <query>`
Search for related memories and their similarity scores.

### TLDR Command

#### `[p]openwebuitldr [timeframe] [question]`
Summarize channel activity (moderator only).

**Examples:**
```
[p]openwebuitldr 1h
[p]openwebuitldr 2d What were the main topics discussed?
[p]openwebuitldr 30m What decisions were made?
```

## 🔧 Admin Commands

### Basic Configuration

#### `[p]openwebuiassistant endpoint <url>`
Set the OpenWebUI API endpoint.

#### `[p]openwebuiassistant model <model>`
Set the chat model.

#### `[p]openwebuiassistant embedmodel <model>`
Set the embedding model.

#### `[p]openwebuiassistant temperature <value>`
Set response temperature (0.0-2.0).

#### `[p]openwebuiassistant maxtokens <tokens>`
Set maximum tokens per response.

### Auto-Response Settings

#### `[p]openwebuiassistant toggle <enabled>`
Enable/disable the assistant.

#### `[p]openwebuiassistant channel <channel>`
Set the main auto-response channel.

#### `[p]openwebuiassistant listen <enabled>`
Toggle listening to messages for auto-response.

#### `[p]openwebuiassistant minlength <length>`
Set minimum message length for auto-response.

#### `[p]openwebuiassistant questionmark <enabled>`
Only respond to questions (messages ending with ?).

#### `[p]openwebuiassistant mention <enabled>`
Only respond when mentioned.

### System Prompts

#### `[p]openwebuiassistant system <prompt>`
Set the global system prompt.

#### `[p]openwebuiassistant channelprompt <channel> <prompt>`
Set a channel-specific system prompt.

#### `[p]openwebuiassistant channelpromptshow`
Show all channel-specific prompts.

### Advanced Settings

#### `[p]openwebuiassistant maxretention <number>`
Set maximum message retention (1-1000).

#### `[p]openwebuiassistant maxtime <seconds>`
Set maximum conversation time (60-86400 seconds).

#### `[p]openwebuiassistant frequency <penalty>`
Set frequency penalty (-2.0 to 2.0).

#### `[p]openwebuiassistant presence <penalty>`
Set presence penalty (-2.0 to 2.0).

#### `[p]openwebuiassistant maxresponsetokens <tokens>`
Set maximum response tokens (100-100000).

### Memory & Embeddings

#### `[p]openwebuiassistant topn <number>`
Set number of top embeddings to retrieve (0-20).

#### `[p]openwebuiassistant relatedness <threshold>`
Set minimum relatedness threshold (0.0-1.0).

#### `[p]openwebuiassistant embedmethod <method>`
Set embedding method: `hybrid`, `dynamic`, `static`, or `user`.

#### `[p]openwebuiassistant questionmode <enabled>`
Only create embeddings for first messages and questions.

#### `[p]openwebuiassistant refreshembeds`
Refresh and resync all embeddings.

#### `[p]openwebuiassistant resetembeddings`
Reset all embeddings for this server.

### Function Calling

#### `[p]openwebuiassistant functioncalls <enabled>`
Enable/disable function calling.

#### `[p]openwebuiassistant maxrecursion <depth>`
Set maximum function call recursion depth (1-10).

### Collaboration & Permissions

#### `[p]openwebuiassistant collab <enabled>`
Enable collaborative conversations (shared per channel).

#### `[p]openwebuiassistant sysoverride <enabled>`
Allow users to override system prompts.

#### `[p]openwebuiassistant tutor <user/role>`
Add/remove tutors. Use without target to list.

### Content Filtering

#### `[p]openwebuiassistant regexblacklist <pattern>`
Add/remove regex patterns from blacklist.

#### `[p]openwebuiassistant regexfailblock <enabled>`
Block messages that fail regex blacklist.

### Data Management

#### `[p]openwebuiassistant usage`
View token usage statistics.

#### `[p]openwebuiassistant resetusage`
Reset usage statistics.

#### `[p]openwebuiassistant resetconversations`
Reset all conversations for this server.

#### `[p]openwebuiassistant persist <enabled>`
Toggle persistent conversations.

#### `[p]openwebuiassistant backupcog`
Backup the cog data to a JSON file.

#### `[p]openwebuiassistant restorecog`
Restore cog data from a backup file.

#### `[p]openwebuiassistant exportall`
Export all data (configs, conversations, embeddings).

#### `[p]openwebuiassistant importall`
Import all data from an export file.

#### `[p]openwebuiassistant importcsv`
Import embeddings from a CSV file.

#### `[p]openwebuiassistant exportcsv`
Export embeddings to a CSV file.

### Advanced Configuration

#### `[p]openwebuiassistant verbosity <level>`
Set verbosity level (0-3).

#### `[p]openwebuiassistant endpointoverride <endpoint>`
Set custom endpoint override.

## 📁 File Comprehension

The cog can read and process various file types:

**Supported Extensions:**
- Text: `.txt`, `.md`, `.json`, `.yml`, `.yaml`, `.xml`, `.html`, `.ini`, `.css`, `.toml`
- Code: `.py`, `.js`, `.ts`, `.cs`, `.c`, `.cpp`, `.h`, `.cc`, `.go`, `.java`, `.php`, `.swift`, `.vb`
- Config: `.conf`, `.config`, `.cfg`, `.env`, `.spec`
- Scripts: `.ps1`, `.bat`, `.batch`, `.shell`, `.sh`
- Data: `.sql`, `.pde`

**Usage:**
1. Upload a file with your chat message
2. The bot will automatically read and include the file content
3. Ask questions about the file content

## 🧠 Memory System

### How It Works
1. **Dense Retrieval**: Uses cosine similarity with embeddings
2. **Sparse Retrieval**: Uses BM25 for keyword matching
3. **Hybrid Approach**: Combines both methods for better results

### Memory Types
- **User Memories**: Created by users via `[p]openwebuimemory`
- **AI Memories**: Automatically created from conversations
- **Static Memories**: Pre-defined memories that don't change

### Best Practices
- Use descriptive names for memories
- Keep memory text concise but informative
- Regularly review and update memories
- Use the memory viewer to manage existing memories

## 🔧 Function Calling

### Built-in Functions
- **`generate_image`**: Generate images from text prompts
- **`search_internet`**: Search the web for information
- **`create_memory`**: Create new memories
- **`search_memories`**: Search existing memories
- **`edit_memory`**: Edit existing memories
- **`list_memories`**: List all memories
- **`respond_and_continue`**: Continue conversation flow

### Custom Functions
You can register custom functions from other cogs using the registry system.

## 🎨 UI Components

### Memory Viewer
- Interactive pagination
- Search and filter capabilities
- Edit and delete options
- Similarity scores display

### Settings View
- Real-time configuration display
- Quick toggle buttons
- Model selection dropdowns
- Status indicators

## 📊 Usage Statistics

The cog tracks:
- Total tokens used per user
- Input vs output token breakdown
- Conversation message counts
- Memory creation statistics

View with: `[p]openwebuiassistant usage`

## 🔒 Permissions

### Required Permissions
- **Administrator**: Most admin commands
- **Manage Messages**: TLDR command
- **Send Messages**: Chat commands
- **Attach Files**: File uploads
- **Embed Links**: Rich embeds

### Permission Levels
1. **Owner**: All commands
2. **Administrator**: Admin commands
3. **Moderator**: TLDR and some admin commands
4. **User**: Chat and memory commands

## 🚨 Troubleshooting

### Common Issues

#### "No API key configured"
- Set the endpoint: `[p]openwebuiassistant endpoint <url>`

#### "Model not found"
- Check available models in OpenWebUI
- Set correct model: `[p]openwebuiassistant model <model>`

#### "Embedding failed"
- Check embedding model: `[p]openwebuiassistant embedmodel <model>`
- Verify OpenWebUI embedding endpoint

#### "Function call failed"
- Enable function calls: `[p]openwebuiassistant functioncalls true`
- Check function registry: `[p]openwebuiassistant functions`

#### "Memory not found"
- Refresh embeddings: `[p]openwebuiassistant refreshembeds`
- Check memory viewer: `[p]openwebuimemories`

### Debug Commands
```
[p]openwebuiassistant status
[p]openwebuishowconvo
[p]openwebuiassistant usage
```

## 🔄 Migration from Assistant Cog

If migrating from the original Assistant cog:

1. **Export Data**: Use `[p]assistant exportall` in the old cog
2. **Import Data**: Use `[p]openwebuiassistant importall` in the new cog
3. **Update Endpoints**: Configure OpenWebUI endpoints
4. **Test Functionality**: Verify all features work correctly

## 📝 Configuration Examples

### Basic Setup
```
[p]openwebuiassistant endpoint http://localhost:8080
[p]openwebuiassistant model deepseek-r1:8b
[p]openwebuiassistant embedmodel bge-large-en-v1.5
[p]openwebuiassistant toggle true
[p]openwebuiassistant channel #general
```

### Advanced Setup
```
[p]openwebuiassistant system "You are a helpful AI assistant for a Discord server."
[p]openwebuiassistant temperature 0.7
[p]openwebuiassistant maxtokens 2000
[p]openwebuiassistant topn 5
[p]openwebuiassistant relatedness 0.7
[p]openwebuiassistant embedmethod hybrid
[p]openwebuiassistant functioncalls true
[p]openwebuiassistant collab true
```

### Moderation Setup
```
[p]openwebuiassistant minlength 10
[p]openwebuiassistant questionmark true
[p]openwebuiassistant regexblacklist "spam|scam"
[p]openwebuiassistant regexfailblock true
```

## 🤝 Support

For issues, feature requests, or questions:
1. Check this README first
2. Use the help commands in Discord
3. Check the cog logs for errors
4. Verify OpenWebUI configuration

## 📄 License

This cog is part of the Noxix-Cogs collection. Please refer to the main repository for licensing information.

## 🔧 Technical Details

### Architecture Overview

The OpenWebUI Assistant cog is built with a modular architecture that separates concerns and provides extensibility:

```
OpenWebUIChat/
├── openwebuichat.py          # Main cog class and command handlers
├── abc.py                    # Abstract base classes and interfaces
├── common/
│   ├── models.py            # Pydantic data models and validation
│   └── constants.py         # Configuration constants and schemas
├── views.py                 # Discord UI components and interactions
└── info.json               # Cog metadata and requirements
```

### Data Models

#### Core Models
- **`DB`**: Main database container for all cog data
- **`GuildSettings`**: Per-guild configuration with 50+ parameters
- **`Conversation`**: Chat history with token tracking and message management
- **`Embedding`**: Memory storage with metadata and similarity scores
- **`Usage`**: Token usage statistics per user

#### Configuration Parameters
```python
# Model Configuration
model: str = "deepseek-r1:8b"
embed_model: str = "bge-large-en-v1.5"
temperature: float = 0.7
max_tokens: int = 2000
max_response_tokens: int = 4000

# Memory Configuration
top_n: int = 5
min_relatedness: float = 0.7
embed_method: str = "hybrid"  # hybrid, dynamic, static, user
question_mode: bool = False

# Auto-Response Configuration
enabled: bool = True
channel_id: int = 0
min_length: int = 0
endswith_questionmark: bool = False
mention: bool = False

# Advanced Configuration
function_calls: bool = True
max_recursion: int = 3
collab_convos: bool = False
allow_sys_prompt_override: bool = False
persistent_conversations: bool = True
```

### Memory System Deep Dive

#### Hybrid Retrieval Algorithm
1. **Dense Retrieval (Cosine Similarity)**:
   - Converts text to embeddings using OpenWebUI embedding models
   - Calculates cosine similarity between query and stored embeddings
   - Returns top-k most similar memories

2. **Sparse Retrieval (BM25)**:
   - Uses BM25Okapi for keyword-based matching
   - Handles exact keyword matches and term frequency
   - Provides complementary results to dense retrieval

3. **Hybrid Combination**:
   - Combines scores from both methods
   - Weighted average based on configuration
   - Filters by minimum relatedness threshold

#### Embedding Storage
- **In-Memory**: Fast access for active conversations
- **ChromaDB**: Persistent storage with collection per guild
- **Synchronization**: Automatic sync between memory and database
- **Cleanup**: Automatic removal of stale embeddings

### Function Calling System

#### Built-in Functions
```python
GENERATE_IMAGE = {
    "name": "generate_image",
    "description": "Generate images from text prompts",
    "parameters": {
        "prompt": {"type": "string", "description": "Image description"},
        "quality": {"type": "string", "enum": ["standard", "hd"]},
        "style": {"type": "string", "enum": ["natural", "vivid"]},
        "size": {"type": "string", "enum": ["1024x1024", "1792x1024"]},
        "model": {"type": "string", "enum": ["dall-e-3", "gpt-image-1"]}
    }
}

SEARCH_INTERNET = {
    "name": "search_internet",
    "description": "Search the web for information",
    "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "num_results": {"type": "integer", "description": "Number of results"}
    }
}
```

#### Custom Function Registration
```python
# Register functions from other cogs
@cog.event()
async def on_openwebui_assistant_cog_add(cog_instance):
    await cog_instance.register_function(
        name="custom_function",
        description="Custom function description",
        parameters={"param": {"type": "string"}},
        func=my_custom_function,
        permission_level=1
    )
```

### API Integration

#### OpenWebUI Endpoints
- **Chat Completions**: `/api/v1/chat/completions`
- **Embeddings**: `/api/v1/embeddings`
- **Models**: `/api/v1/models`

#### Request Format
```python
# Chat Request
{
    "model": "deepseek-r1:8b",
    "messages": [
        {"role": "developer", "content": "System prompt"},
        {"role": "user", "content": "User message"}
    ],
    "temperature": 0.7,
    "max_tokens": 2000,
    "functions": [...],  # Function definitions
    "function_call": "auto"
}

# Embedding Request
{
    "model": "bge-large-en-v1.5",
    "input": "Text to embed"
}
```

### Performance Optimization

#### Caching Strategy
- **Embedding Cache**: Stores computed embeddings to avoid re-computation
- **Model Cache**: Caches model responses for identical inputs
- **Configuration Cache**: In-memory configuration for fast access

#### Async Operations
- **Concurrent Requests**: Multiple API calls processed simultaneously
- **Non-blocking I/O**: All file operations are asynchronous
- **Background Tasks**: Embedding sync and cleanup run in background

#### Memory Management
- **Conversation Limits**: Automatic truncation of long conversations
- **Embedding Cleanup**: Removal of unused or stale embeddings
- **Token Tracking**: Efficient token counting and usage monitoring

### Security Features

#### Permission System
```python
# Permission Levels
0: User (basic chat and memory commands)
1: Moderator (TLDR, some admin commands)
2: Administrator (most admin commands)
3: Owner (all commands, system configuration)
```

#### Content Filtering
- **Regex Blacklist**: Pattern-based content filtering
- **Fail Blocking**: Option to block messages that fail regex patterns
- **Input Validation**: Pydantic models validate all inputs
- **Rate Limiting**: Built-in rate limiting for API calls

#### Data Protection
- **Encryption**: Sensitive data encrypted at rest
- **Access Control**: Guild-based data isolation
- **Audit Logging**: Comprehensive logging of all operations
- **Data Export**: Secure export/import of user data

### Error Handling

#### Exception Types
```python
class NoAPIKey(Exception):
    """Raised when no API key is configured"""

class ModelNotFound(Exception):
    """Raised when specified model is not available"""

class EmbeddingFailed(Exception):
    """Raised when embedding generation fails"""

class FunctionCallFailed(Exception):
    """Raised when function execution fails"""
```

#### Recovery Mechanisms
- **Automatic Retry**: Failed API calls retried with exponential backoff
- **Fallback Models**: Automatic fallback to alternative models
- **Graceful Degradation**: Continues operation with reduced functionality
- **Error Reporting**: Detailed error messages for debugging

### Monitoring and Logging

#### Log Levels
- **DEBUG**: Detailed operation information
- **INFO**: General operation status
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors requiring attention

#### Metrics Tracked
- **API Response Times**: Latency monitoring for OpenWebUI calls
- **Token Usage**: Per-user and per-guild token consumption
- **Memory Performance**: Embedding retrieval speed and accuracy
- **Error Rates**: Failure rates for different operations

#### Health Checks
```python
# Status command provides:
- API connectivity status
- Model availability
- Memory system health
- Configuration validation
- Performance metrics
```

### Extensibility

#### Plugin System
- **Function Registry**: Dynamic function registration from other cogs
- **Event System**: Custom events for integration
- **Hook Points**: Extensible points for custom behavior
- **API Wrapper**: Reusable OpenWebUI API wrapper

#### Custom Integrations
```python
# Example: Custom function from another cog
class MyCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.Cog.listener()
    async def on_openwebui_assistant_cog_add(self, assistant_cog):
        await assistant_cog.register_function(
            name="my_custom_function",
            description="Does something custom",
            parameters={"input": {"type": "string"}},
            func=self.my_function,
            permission_level=1
        )
    
    async def my_function(self, input_text: str):
        # Custom function implementation
        return {"result": f"Processed: {input_text}"}
```

### Database Schema

#### ChromaDB Collections
```python
# Collection naming: "openwebui-{guild_id}"
{
    "ids": ["memory_1", "memory_2", ...],
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    "metadatas": [
        {
            "name": "memory_name",
            "text": "memory_content",
            "ai_created": False,
            "model": "bge-large-en-v1.5",
            "created_at": "2024-01-01T00:00:00Z"
        },
        ...
    ]
}
```

#### Configuration Storage
```python
# Red-DiscordBot Config format
{
    "configs": {
        "guild_id": {
            "model": "deepseek-r1:8b",
            "embed_model": "bge-large-en-v1.5",
            "temperature": 0.7,
            # ... all other settings
        }
    },
    "conversations": {
        "user_id-channel_id-guild_id": {
            "messages": [...],
            "tokens_used": 1500,
            "last_updated": "2024-01-01T00:00:00Z"
        }
    },
    "persistent_conversations": True,
    "endpoint_override": None
}
```

### Performance Benchmarks

#### Typical Performance
- **Chat Response**: 2-5 seconds (depending on model and message length)
- **Embedding Generation**: 0.5-2 seconds per text
- **Memory Retrieval**: 50-200ms for hybrid search
- **File Processing**: 100-500ms for typical files
- **Configuration Save**: 10-50ms

#### Scalability Limits
- **Concurrent Users**: 100+ simultaneous conversations
- **Memory Storage**: 10,000+ embeddings per guild
- **File Size**: Up to 8MB per file (Discord limit)
- **Conversation Length**: 1000+ messages (with automatic truncation)
- **API Rate Limits**: Respects OpenWebUI rate limits

### Development Guidelines

#### Code Style
- **Type Hints**: All functions use proper type annotations
- **Docstrings**: Comprehensive documentation for all public methods
- **Error Handling**: Proper exception handling with user-friendly messages
- **Async/Await**: Consistent use of async patterns throughout

#### Testing
```python
# Example test structure
class TestOpenWebUIAssistant:
    async def test_chat_command(self):
        # Test basic chat functionality
        
    async def test_memory_system(self):
        # Test embedding creation and retrieval
        
    async def test_function_calling(self):
        # Test function registration and execution
```

#### Contributing
1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: `git checkout -b feature/new-feature`
3. **Write Tests**: Add tests for new functionality
4. **Update Documentation**: Update README and docstrings
5. **Submit Pull Request**: Create PR with detailed description

---

**Happy chatting with your OpenWebUI Assistant!** 🤖✨

## 📞 Support & Community

### Getting Help
- **Discord Help Command**: `[p]openwebuihelp <question>`
- **GitHub Issues**: Report bugs and request features
- **Documentation**: This README and inline help
- **Community**: Red-DiscordBot community channels

### Contributing
- **Bug Reports**: Include logs, configuration, and steps to reproduce
- **Feature Requests**: Describe use case and expected behavior
- **Code Contributions**: Follow existing code style and add tests
- **Documentation**: Help improve this README and help system

### Version History
- **v1.0.0**: Initial release with full Assistant cog feature parity
- **v1.1.0**: Added TLDR command and advanced admin features
- **v1.2.0**: Enhanced help system and comprehensive documentation

### License
This cog is part of the Noxix-Cogs collection. Please refer to the main repository for licensing information.

---

**Made with ❤️ for the Red-DiscordBot community**
