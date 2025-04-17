# Dr. X Document Analysis Suite

## Overview

An NLP system designed to analyze Dr. X's mysteriously left-behind research publications in various formats (.pdf, .docx, .csv, .xlsx). The suite extracts, processes, and analyzes text to uncover insights about Dr. X's work and potential clues about their disappearance.

## Core Components

1. **Document Loading:** Multi-format text extraction with intelligent table handling
2. **Text Chunking:** Breaking documents into manageable pieces using recursive or agent-based methods
3. **Vector Embedding:** Creating semantic text representations using `nomic-ai/nomic-embed-text-v1`
4. **Vector Database:** Storing and retrieving embeddings with ChromaDB
5. **RAG Q&A System:** Context-aware answers to user questions about the documents
6. **Translation:** English/Arabic document translation
7. **Summarization:** Document condensation using various LLM strategies
8. **Evaluation:** Quality and performance measurement

## Key Features

-   **Multi-Format Support:** Processes PDF, DOCX, CSV, XLSX, XLS, XLSM files
-   **Smart Text Processing:** Extracts and formats tables into readable text
-   **Flexible Chunking:** `recursive` character splitting or `agented` semantic chunking
-   **Local Vector Storage:** Persistent ChromaDB for efficient semantic search
-   **Interactive QA:** Context-aware responses with conversational memory
-   **Advanced Summarization:**
    -   Multiple methods: `map_reduce`, `refine`, `rerank`
    -   Various prompt strategies: `default`, `analytical`, `extractive`
    -   Optional K-Means clustering with t-SNE visualization
-   **Quality Metrics:** ROUGE-based summary evaluation
-   **Performance Tracking:** Tokens/second monitoring for LLM operations

## Technology

-   **Core:** Python, LangChain, sentence-transformers, ChromaDB, Ollama
-   **Models:** Local LLMs (llama3.1 default) and embedding model
-   **Document Processing:** PyMuPDF, python-docx, pandas, openpyxl
-   **UI/Logging:** rich, inquirer, colorama, matplotlib

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Install Ollama and pull model: `ollama pull llama3.1`
3. Place documents in `Dr.X Files/` directory
4. Run: `python main.py`

## Usage

The interactive CLI offers four main functions:

1. **RAG QA:** Ask questions about the documents
2. **Translation:** Convert documents between English and Arabic
3. **Main Ideas Extraction:** Generate document summaries using various methods
4. **Summary Evaluation:** Compare generated summaries against references

## Configuration

Customize settings in `config.py`:

-   Document paths
-   Chunking strategies and sizes
-   Model selection
-   Vector database settings
-   Prompt templates
-   Performance parameters
