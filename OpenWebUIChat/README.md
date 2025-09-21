# OpenWebUI Chat Bot

A simple Discord bot cog that integrates with OpenWebUI to provide AI chat capabilities. This bot can engage in general conversation and optionally use a knowledge base for enhanced responses.

## 🚀 Features

- **General AI Chat**: Chat with AI models through OpenWebUI
- **Optional Memory System**: Store and retrieve knowledge using embeddings
- **Simple Setup**: Easy configuration with minimal dependencies
- **Hybrid Retrieval**: Combines dense (cosine similarity) and sparse (BM25) search for memory recall

## 📋 Requirements

### Dependencies
- `httpx` - HTTP client for OpenWebUI API calls
- `numpy` - Numerical computations for embeddings
- `rank-bm25` - BM25 sparse retrieval algorithm

### OpenWebUI Setup
1. Install and run OpenWebUI
2. Ensure you have chat and embedding models available
3. Note your OpenWebUI API endpoint (typically `http://localhost:8080`)

## ⚙️ Installation

1. **Copy the cog** to your Red-DiscordBot cogs directory
2. **Install dependencies**:
   ```bash
   pip install httpx numpy rank-bm25
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

4. **Start chatting**:
   ```
   [p]llmchat Hello! How are you today?
   ```

## 📚 Commands

### Chat Commands

#### `[p]llmchat [message]`
Chat with the AI assistant.

**Examples:**
```
[p]llmchat What's the weather like?
[p]llmchat Write a Python function to sort a list
[p]llmchat Explain quantum computing in simple terms
```

### Configuration Commands (Owner Only)

#### `[p]setopenwebui url <url>`
Set the OpenWebUI API endpoint.

#### `[p]setopenwebui key <key>`
Set the API key for authentication.

#### `[p]setopenwebui chatmodel <model>`
Set the chat model to use.

#### `[p]setopenwebui embedmodel <model>`
Set the embedding model for memory system.

### Memory Management (Owner Only)

#### `[p]openwebuimemory add <name> <text>`
Add a memory to the knowledge base.

**Example:**
```
[p]openwebuimemory add "Python basics" "Python is a high-level programming language known for its simplicity and readability."
```

#### `[p]openwebuimemory list`
List all memories in the knowledge base.

#### `[p]openwebuimemory del <name>`
Delete a memory from the knowledge base.

## 🧠 How It Works

### General Chat
- The bot responds to any query as a helpful AI assistant
- Uses a general system prompt for friendly, informative responses
- Works even without any memories stored

### Memory Enhancement
- If memories are available, the bot searches for relevant ones
- Uses hybrid retrieval (dense + sparse search) for better results
- Enhances responses with relevant knowledge when found
- Continues working as general chat if no relevant memories are found

### Hybrid Retrieval System
1. **Dense Retrieval**: Uses cosine similarity with embeddings
2. **Sparse Retrieval**: Uses BM25 for keyword matching  
3. **Combined Results**: Merges both approaches for optimal recall

## 🔧 Configuration

### Recommended Models
- **Chat Models**: `deepseek-r1:8b`, `gpt-4o`, `claude-3-sonnet`, `llama-3.1-8b`
- **Embedding Models**: `bge-large-en-v1.5`, `text-embedding-3-large`, `nomic-embed-text`

### Memory Settings
- **Similarity Threshold**: 0.8 (configurable in code)
- **Top K Results**: 9 memories maximum per query
- **Hybrid Weights**: 70% dense, 30% sparse retrieval

## 🚨 Troubleshooting

### Common Issues

#### "OpenWebUI URL or key not set"
- Set the endpoint: `[p]setopenwebui url <your-openwebui-url>`
- Set the API key if required: `[p]setopenwebui key <your-key>`

#### "Failed to retrieve memories"
- Check your embedding model: `[p]setopenwebui embedmodel <model>`
- Verify OpenWebUI is running and accessible
- The bot will continue with general chat even if memory retrieval fails

#### No response from chat
- Verify OpenWebUI is running
- Check your chat model: `[p]setopenwebui chatmodel <model>`
- Ensure the API endpoint is correct

### Debug Commands
```
[p]setopenwebui url http://localhost:8080  # Check current URL
[p]openwebuimemory list                    # Check if memories exist
```

## 📁 File Structure

```
OpenWebUIChat/
├── __init__.py          # Cog setup
├── info.json           # Cog metadata  
├── openwebuichat.py    # Main implementation
└── README.md           # This file
```

## 🤝 Support

For issues or questions:
1. Check this README first
2. Verify your OpenWebUI setup
3. Check the bot logs for error messages
4. Ensure all dependencies are installed

## 📄 License

This cog is part of the Noxix-Cogs collection. Please refer to the main repository for licensing information.

---

**Happy chatting with your AI assistant!** 🤖✨