# Local Brain

A high-performance, privacy-focused Retrieval-Augmented Generation (RAG) system designed to run 100% locally. This project allows users to chat with personal documents (PDFs, Markdown, Text) without any data leaving the machine.

## Project Overview

*   **Objective**: Local semantic search and generation.
*   **Privacy**: Zero external API calls (unless explicitly configured).
*   **Hardware Target**: Optimized for NVIDIA RTX 4060 (8GB VRAM).
*   **Tech Stack**: Python, LangChain, ChromaDB, Ollama.

## Prerequisites

*   **Python 3.11 or 3.12** (Python 3.13 is NOT supported due to ML ecosystem limitations)
*   **Ollama**: Must be installed and running.
    *   `ollama pull llama3` (Logic)
    *   `ollama pull nomic-embed-text` (Embeddings)
*   **CUDA Toolkit**: Recommended for GPU acceleration (RTX 4060).

## Quick Start

### 1. Set Up Environment

Run the automated setup script:
```powershell
.\setup_env.ps1
```

This will:
- Check for Python 3.11/3.12 (or prompt you to install it)
- Create a virtual environment
- Install all dependencies

### 2. Manual Setup (Alternative)

If you prefer manual installation:

```powershell
# Create venv with Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install from: https://ollama.com/download

```powershell
ollama pull llama3
```

### 4. Ingest Documents

```powershell
python main.py ingest ./data/documents
```

### 5. Start Chat

```powershell
python main.py chat
```

## Hardware Limitations

This system is strictly tuned for **8GB VRAM** budgets:
*   Retriever: ~1GB overhead
*   LLM (4-bit): ~5-6GB
*   Leaves ~1GB buffer for system operations

Do not attempt to run larger unquantized models without upgrading hardware.

## Troubleshooting

### Python Version Issues
If you see compilation errors (`Microsoft Visual C++ required`, `Rust compiler missing`), you're likely using Python 3.13. **Downgrade to Python 3.11 or 3.12**.

### CUDA Not Available
Ensure NVIDIA drivers and CUDA toolkit are installed. Verify with:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```