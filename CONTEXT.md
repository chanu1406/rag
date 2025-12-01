# PROJECT IDENTITY: LOCAL BRAIN (RAG PIPELINE)

## 1. CORE OBJECTIVE
To build a high-performance, privacy-focused **Retrieval-Augmented Generation (RAG)** system that runs 100% locally. The system allows the user to chat with their personal documents (PDFs, Markdown, Text) without data leaving the machine.

## 2. HARDWARE CONSTRAINTS (CRITICAL)
**Target GPU:** NVIDIA RTX 4060
**VRAM Buffer:** 8GB (Strict Limit)
**Compute Capability:** Ada Lovelace Architecture (Supports FP8, but limited by memory)

**Implications for Code Generation:**
* **LLM Strategy:** We cannot load full precision 7B+ models in raw PyTorch without OOM (Out of Memory) errors. We **MUST** utilize `Ollama` (running externally) or 4-bit quantization (`bitsandbytes`/`Unsloth`) to fit the LLM into ~4-5GB VRAM.
* **Embedding Strategy:** The Embedding model (e.g., `all-MiniLM-L6-v2`) is small (~500MB). This **MUST** run on `device='cuda'` inside PyTorch to utilize the RTX 4060's tensor cores for fast vectorization.
* **VRAM Budgeting:**
    * LLM (Quantized): ~5.0 GB
    * Embedding Model + Vectors: ~1.0 GB
    * System Overhead/Display: ~1.5 GB
    * *Buffer:* ~0.5 GB

## 3. TECHNICAL STACK
* **Language:** Python 3.10+
* **Orchestration:** LangChain (or LangGraph for complex flows)
* **LLM Backend:** Ollama (via `langchain-community.llms`) OR HuggingFace `bitsandbytes` (if strictly internal). *Preference: Ollama for stability.*
* **Vector Database:** ChromaDB (Persistent local storage)
* **Embeddings:** `sentence-transformers` (HuggingFace)
* **Hardware Acceleration:** CUDA 12.x (PyTorch)

## 4. ARCHITECTURE OVERVIEW
The project follows a modular "ETL" (Extract, Transform, Load) + Retrieval pattern:

1.  **Ingestion (`src/ingest.py`):**
    * Loaders for PDF, TXT, MD.
    * RecursiveCharacterTextSplitter for chunking.
2.  **Indexing (`src/vectorstore.py`):**
    * Initialize ChromaDB client.
    * Embed chunks using GPU.
    * Upsert vectors to disk.
3.  **Retrieval (`src/rag.py`):**
    * Accept user query.
    * Semantic search against ChromaDB.
    * Construct prompt (System Prompt + Context + Question).
    * Stream response from LLM.
4.  **Interface (`main.py`):**
    * CLI (Command Line Interface) initially.
    * Streamlit/Gradio dashboard (Future capability).

## 5. CODING STANDARDS & STYLE
* **Type Hinting:** Mandatory for all function signatures (e.g., `def process(text: str) -> List[float]:`).
* **Docstrings:** Google Style or NumPy Style. Explain *args*, *returns*, and *raises*.
* **Error Handling:** specific `try/except` blocks (avoid bare `except Exception`).
* **Modularity:** No monolithic scripts. Functions should do one thing.
* **Config Driven:** All magic numbers (chunk sizes, model names, paths) must be pulled from `config.yaml`, not hardcoded.

## 6. CURRENT PROJECT STATUS
* [ ] **Phase 1:** Skeleton & Structure Setup (Current)
* [ ] **Phase 2:** Ingestion Logic Implementation
* [ ] **Phase 3:** Vector Store & Embedding Implementation
* [ ] **Phase 4:** Retrieval Chain & LLM Connection
* [ ] **Phase 5:** Optimization (VRAM tuning)

## 7. INSTRUCTIONS FOR AI AGENTS
When assisting with this project:
1.  **Always** verify VRAM usage before suggesting a new library or model.
2.  If suggesting a PyTorch operation, ensure `.to('cuda')` is used where appropriate.
3.  Assume the user has `Ollama` installed and running in the background for LLM inference.
4.  Reference the specific file paths (`src/...`) defined in the architecture.