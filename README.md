# Local Brain

A high-performance, privacy-focused Retrieval-Augmented Generation (RAG) system designed to run 100% locally. This project allows users to chat with personal documents (PDFs, Markdown, Text) without any data leaving the machine.

## Project Overview

*   **Objective**: Local semantic search and generation.
*   **Privacy**: Zero external API calls (unless explicitly configured).
*   **Hardware Target**: Optimized for NVIDIA RTX 4060 (8GB VRAM).
*   **Tech Stack**: Python, LangChain, ChromaDB, Ollama.

## Architecture

The system follows a modular ETL + Retrieval pattern:
1.  **Ingestion**: Loads and chunks documents.
2.  **Indexing**: Embeds text using local models and stores vectors in ChromaDB.
3.  **Retrieval**: Semantic search with optional re-ranking.
4.  **Generation**: Answers queries using a quantized LLM (via Ollama).

## Prerequisites

*   **Python 3.10+**
*   **Ollama**: Must be installed and running.
    *   `ollama pull llama3` (Logic)
    *   `ollama pull nomic-embed-text` (Embeddings)
*   **CUDA Toolkit**: Recommended for GPU acceleration.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**
    Review `config/config.yaml` to adjust model preferences and paths.

## Usage

**Ingest Documents**
```bash
python main.py ingest --source ./documents/
```

**Start Chat**
```bash
python main.py chat
```

## Hardware Limitations

This system is strictly tuned for **8GB VRAM** budgets.
*   Retriever: ~1GB overhead
*   LLM (4-bit): ~5-6GB
*   Leaves ~1GB buffer for system operations.

Do not attempt to run larger unquantized models without upgrading hardware or offloading to CPU.