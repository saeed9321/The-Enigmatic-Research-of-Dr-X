"""
Configuration settings for the RAG (Retrieval-Augmented Generation) system.

This module contains settings related to:
    - Document processing
    - Loader
    - Chunking
    - Clustering
    - Embedding
    - Vector database
    - LLM
    - Prompt strategies
    - RAG
    - Translation
    - Evaluation
    - Summarizer
    - UI
    - System prompt template
"""
import os
from typing import Dict, Any

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Document processing settings
DOCUMENT_SETTINGS = {
    # Directory containing the documents to process
    "documents_dir": os.path.join(BASE_DIR, "Dr.X Files"),
}
# Loader settings
LOADER_SETTINGS = {
    # Log file path
    "log_file": os.path.join(BASE_DIR, "logs", "loader.log")
}
# Chunking settings
CHUNKING_SETTINGS = {
    # Default chunking strategy
    "chunk_strategy": "recursive",

    # Maximum number of tokens per chunk
    "max_tokens_per_chunk": 512,
    
    # Chunk overlap to maintain context between chunks
    "chunk_overlap": 50,

    # Log file path for chunking
    "log_file": os.path.join(BASE_DIR, "logs", "chunking.log")
}
# Clustering settings
CLUSTERING_SETTINGS = {

    # Number of clusters for document grouping
    "n_clusters": 4,

    # Log file path for clustering
    "log_file": os.path.join(BASE_DIR, "logs", "clustering.log")
}
# Embedding settings
EMBEDDING_SETTINGS = {
    # Embedding model to use
    "model_name": "nomic-ai/nomic-embed-text-v1",

    # Log file path for embedding
    "log_file": os.path.join(BASE_DIR, "logs", "embedding.log")
}
# Vector database settings
VECTOR_DB_SETTINGS = {
    # Directory to store the vector database
    "persist_dir": os.path.join(BASE_DIR, "chroma_store"),
    
    # Name of the collection in the vector database
    "collection_name": "documents",

    # Log file path for vector database
    "log_file": os.path.join(BASE_DIR, "logs", "vector_db.log")
}
# LLM settings
LLM_SETTINGS = {
    # Default model to use
    "default_model": "llama3.1",
    
    # Context window size
    "num_ctx": 4096,
    
    # Maximum number of tokens to generate in the output
    "max_output_tokens": 1024,
    
    # Temperature for LLM (0 = deterministic, 1 = creative)
    "temperature": 0.2
}
# Define different prompt strategies
PROMPT_STRATEGIES = {
    "default": {
        "map_template": """Write a concise summary of the following text:
        {text}
        CONCISE SUMMARY:""",

        "combine_template": """Write a comprehensive summary of the following text that combines the key points:
        {text}
        COMPREHENSIVE SUMMARY:"""
    },

    "analytical": {
        "map_template": """Analyze the following text and extract the main concepts, arguments, and findings:
        {text}
        ANALYTICAL SUMMARY:""",

        "combine_template": """Synthesize the following summaries into a cohesive analysis that highlights key themes and relationships:
        {text}
        SYNTHESIZED ANALYSIS:"""
    },
    "extractive": {
        "map_template": """Identify and extract the most important sentences and key information from the following text:
        {text}
        KEY EXTRACTS:""",

        "combine_template": """Combine the following key extracts into a flowing narrative that preserves the most important information:
        {text}
        NARRATIVE SUMMARY:"""
    }
}
# RAG settings
RAG_SETTINGS = {
    # Number of documents to retrieve from the vector database
    "n_results": 5,
    
    # Whether to include chat history in the context
    "use_history": True,
    
    # Maximum number of messages to keep in the chat history
    "max_history_messages": 2,
    
    # Log file path
    "log_file": os.path.join(BASE_DIR, "logs", "rag_pipeline.log")
}
# Translation settings
TRANSLATION_SETTINGS = {    
    # Log file path
    "log_file": os.path.join(BASE_DIR, "logs", "translation.log"),

    # Default model to use
    "default_model": "llama3.1" # you can use also (prakasharyan/qwen-arabic) or any other model available in Ollama
}
# evaluation settings
EVALUATION_SETTINGS = {    
    # Log file path
    "log_file": os.path.join(BASE_DIR, "logs", "evaluation.log"),

    # Default model to use
    "default_model": "llama3.1"
}
# Summarizer settings
SUMMARIZER_SETTINGS = {    
    # Log file path
    "log_file": os.path.join(BASE_DIR, "logs", "summarizer.log"),

    # Enable/Disable clustering (to find the main ideas using K-means clustering)
    "with_clustering": True,
}
# UI settings
UI_SETTINGS = {
    # Welcome message title
    "welcome_title": "Dr. X's Investigation Assistant",
    
    # Welcome message subtitle
    "welcome_subtitle": "Help investigate Dr. X's mysterious disappearance through the documents left behind\nType 'exit' to end the conversation",
    # Exit commands
    "exit_commands": ["exit", "quit", "bye"]
}
# System prompt template
SYSTEM_PROMPT_TEMPLATE = """
You are Dr. X's AI assistant, designed to help users understand Dr. X's left behind documents. Your responses should be:

1. Conversational - speak naturally as if having a dialogue
2. Focused specifically on answering the user's question
3. Based on the provided context, not speculation
4. Clear and concise while maintaining accuracy

When answering:
- If the question is about Dr. X's disappearance or documents:
  * Make your best guess based on the context provided
  * Draw directly from the context provided
  * Look for clues or patterns that might explain the disappearance
  * Provide detailed analytical answers with specific references
  * Cite specific documents or sections when relevant
  * Highlight any suspicious information or inconsistencies
  * If the context doesn't contain enough information, say so
- If the question is unrelated to Dr. X's documents:
  * Respond in a conversational manner
  * Answer normally as a helpful assistant
  * Stay within your general knowledge boundaries

Context from Dr. X's documents:
{context}

Please answer this question in a conversational way:
{question}
"""

def get_config() -> Dict[str, Any]:
    """
    Get combined configuration.
    
    Returns:
        Combined configuration dictionary
    """
    # Base configuration for all environments
    config = {
        "document": DOCUMENT_SETTINGS,
        "loader": LOADER_SETTINGS,
        "chunking": CHUNKING_SETTINGS,
        "clustering": CLUSTERING_SETTINGS,
        "summarizer": SUMMARIZER_SETTINGS,
        "embedding": EMBEDDING_SETTINGS,
        "vector_db": VECTOR_DB_SETTINGS,
        "translation":TRANSLATION_SETTINGS,
        "evaluation":EVALUATION_SETTINGS,
        "llm": LLM_SETTINGS,
        "prompt_strategies": PROMPT_STRATEGIES,
        "rag": RAG_SETTINGS,
        "ui": UI_SETTINGS,
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
        "base_dir": BASE_DIR,
    }
    
    # Create required directories
    os.makedirs(os.path.dirname(config["rag"]["log_file"]), exist_ok=True)
    os.makedirs(config["vector_db"]["persist_dir"], exist_ok=True)
    
    return config 