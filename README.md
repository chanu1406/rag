<p align="center">
  <h1 align="center">🧠 Local Brain RAG</h1>
  <p align="center">
    <strong>Privacy-First Document Intelligence</strong>
  </p>
  <p align="center">
    A high-performance Retrieval-Augmented Generation system that runs 100% locally.<br/>
    Chat with your documents without any data leaving your machine.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/LangChain-0.3+-green?style=flat-square" alt="LangChain"/>
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-orange?style=flat-square" alt="Ollama"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-purple?style=flat-square" alt="ChromaDB"/>
</p>

---

## ✨ Features

- **🔒 Privacy First** — Zero external API calls. All processing happens locally.
- **⚡ GPU Accelerated** — CUDA-powered embeddings for fast vectorization.
- **🔍 Hybrid Search** — Combines semantic (vector) + keyword (BM25) retrieval with Reciprocal Rank Fusion.
- **🎯 Re-ranking** — FlashRank cross-encoder for improved relevance.
- **📄 Multi-Format** — Supports PDF, Markdown, Text, and Word documents.
- **🛠️ Fully Configurable** — YAML-based configuration for all parameters.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11 or 3.12** (3.13 not yet supported)
- **NVIDIA GPU with CUDA** (recommended for performance)
- **[Ollama](https://ollama.ai/)** installed and running

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/local-brain-rag.git
cd local-brain-rag

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 3. Setup Ollama

```bash
# Pull the LLM model
ollama pull llama3

# Start Ollama server (if not running)
ollama serve
```

### 4. Ingest Documents

```bash
# Add your documents to data/documents/ then run:
python main.py ingest ./data/documents

# Or ingest with reset (clears existing vectors)
python main.py ingest ./data/documents --reset
```

### 5. Start Chatting

```bash
python main.py chat
```

---

## 📖 Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py ingest <path>` | Ingest documents from a file or directory |
| `python main.py ingest <path> --reset` | Clear vector store and re-ingest |
| `python main.py chat` | Start interactive chat session |

### Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the chat |

---

## ⚙️ Configuration

All settings are managed in `config/config.yaml`:

```yaml
llm:
  model: "llama3"              # Ollama model to use
  temperature: 0.0             # 0 = deterministic responses

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"               # Use GPU for embeddings

retrieval:
  chunk_size: 500              # Characters per chunk
  k_retrieved: 20              # Candidates for re-ranking
  k_final: 5                   # Final context to LLM
  use_reranker: true           # Enable FlashRank
```

---

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Documents  │───▶│  Ingestion  │───▶│   Chunks    │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────────────────▼──────────────────────────┐
                   │                    Indexing                        │
                   │  ┌─────────────┐              ┌─────────────────┐  │
                   │  │  Embedding  │──────────────▶│    ChromaDB    │  │
                   │  │   (CUDA)    │              │  (Vector Store) │  │
                   │  └─────────────┘              └─────────────────┘  │
                   └────────────────────────────────────────────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐    ┌─────────────┐
│    User     │───▶│   Query     │───▶│  Retrieval  │───▶│   Answer    │
│   Question  │    │  Embedding  │    │ (Hybrid+RRF)│    │ + Sources   │
└─────────────┘    └─────────────┘    └──────┬──────┘    └─────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Ollama (LLM)   │
                                    │   Generation    │
                                    └─────────────────┘
```

---

## 📁 Project Structure

```
local-brain-rag/
├── main.py                 # CLI entry point
├── config/
│   └── config.yaml         # Configuration file
├── src/
│   ├── config.py           # Config validation (Pydantic)
│   ├── ingest.py           # Document loading & chunking
│   ├── vectorstore.py      # ChromaDB + embeddings + retrieval
│   ├── rag.py              # RAG orchestration
│   └── utils.py            # VRAM monitoring utilities
├── data/
│   ├── documents/          # Your documents go here
│   └── chroma_db/          # Persisted vector store
├── tests/                  # Test suite
└── requirements.txt        # Dependencies
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA not available** | Ensure NVIDIA drivers and CUDA toolkit are installed |
| **Ollama connection failed** | Run `ollama serve` in a separate terminal |
| **Python version error** | Use Python 3.11 or 3.12 |
| **Out of memory** | Reduce `k_retrieved` in config or use smaller embedding model |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for privacy-conscious AI
</p>