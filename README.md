# Local Brain RAG

A high-performance, privacy-focused Retrieval-Augmented Generation (RAG) system designed to run 100% locally. This project allows users to chat with personal documents (PDFs, Markdown, Text) without any data leaving the machine.

## Project Overview

-   **Objective**: Local semantic search and generation.
-   **Privacy**: Zero external API calls.
-   **Hardware Target**: Optimized for NVIDIA RTX 4060 (8GB VRAM).
-   **Tech Stack**: Python, LangChain, ChromaDB, Ollama.

## Prerequisites

-   **Python 3.11 or 3.12**
-   **Ollama**: Must be installed and running.
    -   `ollama pull llama3`
    -   `ollama pull nomic-embed-text`
-   **CUDA Toolkit**: Required for GPU acceleration.

## Quick Start

### 1. Set Up Environment

Run the automated setup script:
```powershell
.\setup_env.ps1
```

### 2. Manual Setup

```powershell
# Create venv
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Ingest Documents

```powershell
python main.py ingest ./data/documents
```

### 4. Start Chat

```powershell
python main.py chat
```

## Hardware Configuration

This system is tuned for **8GB VRAM**:
-   Retriever: ~1GB overhead
-   LLM (4-bit): ~5-6GB
-   Buffer: ~1GB

## Troubleshooting

-   **CUDA Not Available**: Ensure NVIDIA drivers and CUDA toolkit are installed.
-   **Python Version**: Use Python 3.11 or 3.12. Python 3.13 is not supported.