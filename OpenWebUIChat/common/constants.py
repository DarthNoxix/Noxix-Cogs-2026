# Constants for OpenWebUI Assistant

# Common OpenWebUI models and their context limits
MODELS = {
    # OpenAI models (if using OpenAI API through OpenWebUI)
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 400000,
    
    # Common local models
    "deepseek-r1:8b": 32768,
    "deepseek-r1:32b": 32768,
    "llama3.1:8b": 8192,
    "llama3.1:70b": 8192,
    "qwen2.5:7b": 32768,
    "qwen2.5:14b": 32768,
    "qwen2.5:32b": 32768,
    "mistral:7b": 32768,
    "mixtral:8x7b": 32768,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
    "phi3:3.8b": 128000,
    "phi3:14b": 128000,
    "gemma2:9b": 8192,
    "gemma2:27b": 8192,
}

# Common embedding models
EMBEDDING_MODELS = [
    "bge-large-en-v1.5",
    "bge-large-en-v1.5:1024",
    "bge-m3",
    "nomic-embed-text",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

# Vision models (models that support image input)
VISION_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision-preview",
    "llava:7b",
    "llava:13b",
    "llava:34b",
    "bakllava:7b",
    "bakllava:13b",
    "moondream:1.8b",
    "moondream:2b",
]

# Models that support function calling
FUNCTION_CALLING_MODELS = [
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-r1:8b",
    "deepseek-r1:32b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
]

# File extensions that can be read
READ_EXTENSIONS = [
    ".txt", ".py", ".json", ".yml", ".yaml", ".xml", ".html", ".ini", ".css", ".toml", ".md",
    ".conf", ".config", ".cfg", ".go", ".java", ".c", ".php", ".swift", ".vb", ".xhtml", ".rss",
    ".js", ".ts", ".cs", ".c++", ".cpp", ".cbp", ".h", ".cc", ".ps1", ".bat", ".batch", ".shell",
    ".env", ".sh", ".pde", ".spec", ".sql"
]

# Loading GIF URL
LOADING = "https://i.imgur.com/l3p6EMX.gif"

# Built-in function schemas
GENERATE_IMAGE = {
    "name": "generate_image",
    "description": "Use this to generate an image from a text prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "A sentence or phrase that describes what you want to visualize, must be less than 1000 characters",
            },
            "quality": {
                "type": "string",
                "enum": ["standard", "hd", "low", "medium", "high"],
                "description": "The quality of the image. Defaults to 'medium'.",
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1792x1024", "1024x1792", "1024x1536", "1536x1024"],
                "description": "The size of the image, defaults to 1024x1024",
            },
        },
        "required": ["prompt"],
    },
}

SEARCH_INTERNET = {
    "name": "search_web_brave",
    "description": "Search the web for current information on a topic using the Brave Search API.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, can be a question or topic",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

CREATE_MEMORY = {
    "name": "create_memory",
    "description": "Use this to remember information that you normally wouldnt have access to. Useful when someone corrects you, tells you something new, or tells you to remember something. Use the search_memories function first to ensure no duplicates are created.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {
                "type": "string",
                "description": "A short name to describe the memory, perferrably less than 50 characters or 3 words tops",
            },
            "memory_text": {
                "type": "string",
                "description": "The information to remember, write as if you are informing yourself of the thing to remember, Make sure to include the context of the conversation as well as the answer or important information to be retained",
            },
        },
        "required": ["memory_name", "memory_text"],
    },
}

SEARCH_MEMORIES = {
    "name": "search_memories",
    "description": "Use this to find information about something, always use this if you are unsure about the answer to a question.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_query": {
                "type": "string",
                "description": "a sentence or phrase that describes what you are looking for, this should be as specific as possible, it will be tokenized to find the best match with related embeddings.",
            },
            "amount": {
                "type": "integer",
                "description": "Max amount of memories to fetch. Defaults to 2",
            },
        },
        "required": ["search_query"],
    },
}

EDIT_MEMORY = {
    "name": "edit_memory",
    "description": "Use this to edit existing memories, useful for correcting inaccurate memories after making them. Use search_memories first if the memory you need to edit is not in the conversation.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {
                "type": "string",
                "description": "The name of the memory entry, case sensitive",
            },
            "memory_text": {
                "type": "string",
                "description": "The new text that will replace the current content of the memory, this should reflect the old memory with the corrections",
            },
        },
        "required": ["memory_name", "memory_text"],
    },
}

LIST_MEMORIES = {
    "name": "list_memories",
    "description": "Get a list of all your available memories",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

DO_NOT_RESPOND_SCHEMA = {
    "name": "do_not_respond",
    "description": "Call this function if you do not want to or do not need to respond to the user.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

RESPOND_AND_CONTINUE = {
    "name": "respond_and_continue",
    "description": "Call this function if you want to respond to the user but also continue working on the task at hand.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message to send to the user, this can be something like 'I will continue working on this task, please wait.' or 'I will get back to you shortly.'",
            },
        },
        "required": ["content"],
    },
}
