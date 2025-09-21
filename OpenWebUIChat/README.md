# 🤖 OpenWebUIChat - Ultimate AI Assistant

The most advanced AI assistant for Discord with OpenWebUI integration! Features modern UI, slash commands, auto-responses, conversation memory, and much more.

## ✨ Features

### 🎨 Modern UI & Experience
- **Beautiful Embeds**: All responses use rich Discord embeds with colors and formatting
- **Interactive Buttons**: Regenerate responses, save memories, clear chat history
- **Modal Forms**: Easy setup and memory management with interactive forms
- **Slash Commands**: Modern `/chat` and `/ask` commands with autocomplete
- **Typing Indicators**: Shows when the AI is thinking

### 🧠 Advanced AI Capabilities
- **Conversation Memory**: Maintains context across messages in a conversation
- **Knowledge Base**: Store and retrieve information using hybrid search (dense + sparse)
- **Smart Response Filtering**: Automatically removes unwanted roleplay elements and `<think>` tags
- **Custom System Prompts**: Per-guild customizable AI personality
- **Multiple Models**: Support for any OpenWebUI-compatible chat and embedding models

### 🚀 Auto-Response System
- **Channel Auto-Response**: Automatically respond in configured channels
- **Smart Triggers**: Responds to mentions, greetings, and questions
- **Configurable**: Easy setup and management of auto-response channels

### ⚙️ Admin & Management
- **Rate Limiting**: Configurable per-user rate limits to prevent spam
- **Guild Management**: Enable/disable AI per server
- **Conversation Management**: Clear chat history, manage conversations
- **Memory Management**: Add, search, and delete knowledge base entries
- **Status Monitoring**: View configuration and usage statistics

### 🔧 Advanced Configuration
- **Per-Guild Settings**: Customize AI behavior per server
- **User Preferences**: Individual user settings and preferences
- **Flexible Models**: Support for any OpenWebUI model
- **Error Handling**: Graceful error handling with user-friendly messages

## 📋 Requirements

### Dependencies
- `httpx>=0.25.0` - HTTP client for OpenWebUI API calls
- `numpy>=1.24.0` - Numerical computations for embeddings
- `rank-bm25>=0.2.2` - BM25 sparse retrieval algorithm
- `discord.py>=2.3.0` - Discord.py for modern UI features

### OpenWebUI Setup
1. Install and run OpenWebUI
2. Ensure you have chat and embedding models available
3. Note your OpenWebUI API endpoint (typically `http://localhost:8080`)

## ⚙️ Installation

1. **Copy the cog** to your Red-DiscordBot cogs directory
2. **Install dependencies**:
   ```bash
   pip install httpx numpy rank-bm25 discord.py
   ```
3. **Load the cog**:
   ```
   [p]load OpenWebUIChat
   ```

## 🎯 Quick Start

### Interactive Setup
1. **Use the setup modal**:
   ```
   /ai setup
   ```
   This opens an interactive form to configure everything at once!

### Manual Setup
1. **Configure OpenWebUI endpoint**:
   ```
   /ai url http://localhost:8080
   ```

2. **Set your API key** (if required):
   ```
   /ai key your-api-key
   ```

3. **Configure models**:
   ```
   /ai models deepseek-r1:8b bge-large-en-v1.5
   ```

4. **Start chatting**:
   ```
   /chat Hello! How are you today?
   ```

## 📚 Commands

### 🤖 Chat Commands

#### `/chat [message]` or `/ask [question]`
Chat with the AI assistant using modern slash commands.

**Examples:**
```
/chat What's the weather like?
/ask Write a Python function to sort a list
/chat Explain quantum computing in simple terms
```

**Features:**
- Beautiful embed responses
- Interactive buttons (regenerate, save memory, clear chat)
- Conversation memory
- Automatic response cleaning

### ⚙️ Configuration Commands (Owner Only)

#### `/ai setup`
Interactive setup modal for quick configuration.

#### `/ai url <url>`
Set the OpenWebUI API endpoint.

#### `/ai key <key>`
Set the API key for authentication.

#### `/ai models <chat_model> <embed_model>`
Set the chat and embedding models.

#### `/ai status`
Show current configuration and status.

### 📢 Auto-Response Management

#### `/ai auto <channel>`
Add a channel for auto-responses. The AI will automatically respond to:
- Bot mentions
- Greetings (hey, hi, hello)
- Questions (what, how, why, when, where, who)

#### `/ai remove <channel>`
Remove a channel from auto-responses.

### 🧠 Memory Management (Owner Only)

#### `/memory add <name> <content>`
Add a memory to the knowledge base.

**Example:**
```
/memory add "Python basics" "Python is a high-level programming language known for its simplicity and readability."
```

#### `/memory list`
List all memories in the knowledge base with previews.

#### `/memory delete <name>`
Delete a memory from the knowledge base.

#### `/memory search <query>`
Search memories using semantic similarity.

### 👑 Admin Commands (Manage Guild Permission)

#### `/aiadmin enable`
Enable AI assistant in this guild.

#### `/aiadmin disable`
Disable AI assistant in this guild.

#### `/aiadmin ratelimit <number>`
Set rate limit (1-60 messages per minute per user).

#### `/aiadmin prompt <text>`
Set a custom system prompt for this guild.

#### `/aiadmin clear`
Clear all conversation history in this guild.

## 🎨 UI Features

### Interactive Response Buttons
Every AI response includes:
- **🔄 Regenerate**: Generate a new response to the same question
- **💾 Save Memory**: Save the response as a memory (requires Manage Messages permission)
- **🗑️ Clear Chat**: Clear the conversation history

### Beautiful Embeds
- **Color-coded responses**: Blue for normal responses, red for errors, green for success
- **Rich formatting**: Proper markdown support and formatting
- **User attribution**: Shows who requested the response
- **Timestamps**: When the response was generated

### Modal Forms
- **Setup Modal**: Interactive configuration with all settings
- **Memory Modal**: Easy memory creation with name and content fields
- **Validation**: Proper input validation and error handling

## 🧠 How It Works

### Conversation Memory
- Each conversation thread maintains its own history
- Context is preserved across multiple messages
- Automatic cleanup of old conversations (24+ hours)
- Configurable maximum conversation length

### Knowledge Base System
- **Hybrid Retrieval**: Combines dense (cosine similarity) and sparse (BM25) search
- **Semantic Search**: Find relevant information even with different wording
- **Automatic Integration**: Relevant memories are automatically included in responses
- **Fallback**: Works perfectly even without any memories

### Response Processing
- **Smart Filtering**: Removes unwanted roleplay elements and `<think>` tags
- **Content Cleaning**: Handles multiple spaces, excessive newlines
- **Length Management**: Automatically handles Discord's message limits
- **Error Recovery**: Graceful handling of API errors

### Rate Limiting
- **Per-user limits**: Configurable messages per minute
- **Automatic reset**: Rate limits reset every minute
- **Guild-specific**: Different limits per server
- **Admin override**: Admins can adjust limits as needed

## 🔧 Configuration

### Recommended Models
- **Chat Models**: `deepseek-r1:8b`, `gpt-4o`, `claude-3-sonnet`, `llama-3.1-8b`
- **Embedding Models**: `bge-large-en-v1.5`, `text-embedding-3-large`, `nomic-embed-text`

### Memory Settings
- **Similarity Threshold**: 0.7 (configurable in code)
- **Top K Results**: 5 memories maximum per query
- **Hybrid Weights**: 70% dense, 30% sparse retrieval

### Rate Limiting
- **Default**: 5 messages per minute per user
- **Range**: 1-60 messages per minute
- **Reset**: Every 60 seconds

## 🚨 Troubleshooting

### Common Issues

#### "OpenWebUI URL or key not set"
- Use `/ai setup` for interactive configuration
- Or manually set: `/ai url <your-openwebui-url>`
- Set API key if required: `/ai key <your-key>`

#### "Rate Limited"
- You're sending messages too quickly
- Wait a moment before trying again
- Admins can adjust limits with `/aiadmin ratelimit <number>`

#### "Failed to retrieve memories"
- Check your embedding model: `/ai models <chat> <embed>`
- Verify OpenWebUI is running and accessible
- The bot continues with general chat even if memory retrieval fails

#### No response from chat
- Verify OpenWebUI is running
- Check your chat model: `/ai models <chat> <embed>`
- Ensure the API endpoint is correct
- Check if AI is enabled: `/aiadmin enable`

### Debug Commands
```
/ai status                    # Check current configuration
/memory list                  # Check if memories exist
/aiadmin ratelimit 10         # Increase rate limit if needed
```

## 🎯 Use Cases

### General Chat
- Casual conversation with AI
- Questions and answers
- Creative writing assistance
- Code help and debugging

### Knowledge Base
- Store server rules and information
- FAQ responses
- Technical documentation
- Custom server knowledge

### Auto-Response Channels
- General help channels
- Q&A channels
- Support channels
- Community interaction

### Admin Management
- Server-specific AI personalities
- Rate limiting for busy servers
- Conversation management
- Usage monitoring

## 📁 File Structure

```
OpenWebUIChat/
├── __init__.py          # Cog setup
├── info.json           # Cog metadata with modern features
├── openwebuichat.py    # Main implementation with all features
└── README.md           # This comprehensive guide
```

## 🔮 Future Features

Planned enhancements include:
- Image generation integration
- Code execution capabilities
- Web search integration
- Voice message support
- Multi-language support
- Advanced analytics
- Custom function calling
- Plugin system

## 🤝 Support

For issues or questions:
1. Check this README first
2. Use `/ai status` to verify configuration
3. Check the bot logs for error messages
4. Ensure all dependencies are installed
5. Verify OpenWebUI is running and accessible

## 📄 License

This cog is part of the Noxix-Cogs collection. Please refer to the main repository for licensing information.

---

**🚀 Ready to experience the ultimate AI assistant for Discord!** 

Start with `/ai setup` and begin chatting with `/chat` - your AI assistant is ready to help! 🤖✨