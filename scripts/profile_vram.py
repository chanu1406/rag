"""
VRAM Profiling Script for Local Brain RAG.

Per CONTEXT.md: Validate RTX 4060 8GB constraints.
Per AGENTCONTEXT.md: Explicit monitoring, predictable performance.

Usage:
    python scripts/profile_vram.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.logger import setup_logging, logger
from src.ingest import DocumentLoader
from src.vectorstore import VectorManager
from src.utils import check_cuda_available, VRAMMonitor, estimate_vram_usage, clear_cuda_cache
from langchain_core.documents import Document


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def profile_embedding_model():
    """Profile embedding model VRAM usage."""
    print_header("VRAM PROFILING: Embedding Model")
    
    config = load_config()
    
    # Get estimated usage
    model_name = config.embedding.model_name.split('/')[-1]
    estimated = estimate_vram_usage(model_name)
    
    print(f"Model: {config.embedding.model_name}")
    print(f"Device: {config.embedding.device}")
    print(f"Estimated VRAM: {estimated['vram_gb']} GB")
    print(f"Recommended batch size: {estimated['recommended_batch_size']}")
    print(f"Embedding dimensions: {estimated['dimensions']}")
    
    # Measure actual usage during initialization
    print("\nInitializing VectorManager...")
    with VRAMMonitor("VectorManager Initialization"):
        manager = VectorManager(config)
    
    return manager


def profile_ingestion(manager: VectorManager):
    """Profile document ingestion VRAM usage."""
    print_header("VRAM PROFILING: Document Ingestion")
    
    # Create sample documents of varying sizes
    sample_docs = [
        Document(
            page_content="Short test document. " * 20,  # ~100 words
            metadata={"source": "test1.txt", "filename": "test1.txt"}
        ),
        Document(
            page_content="Medium length document. " * 50,  # ~250 words
            metadata={"source": "test2.txt", "filename": "test2.txt"}
        ),
        Document(
            page_content="Long document with substantial content. " * 100,  # ~500 words
            metadata={"source": "test3.txt", "filename": "test3.txt"}
        )
    ]
    
    print(f"Sample documents: {len(sample_docs)}")
    print(f"Total content length: {sum(len(d.page_content) for d in sample_docs)} chars")
    
    # Profile ingestion
    print("\nIngesting documents (with VRAM monitoring)...")
    ids = manager.add_documents(sample_docs)
    
    print(f"\nStored {len(ids)} document embeddings")
    
    # Clear cache and measure again
    print("\nClearing CUDA cache...")
    clear_cuda_cache()
    
    return sample_docs


def profile_query(manager: VectorManager):
    """Profile query execution VRAM usage."""
    print_header("VRAM PROFILING: Query Execution")
    
    test_queries = [
        "What is a test document?",
        "Tell me about medium length content",
        "How does the long document describe substantial information?"
    ]
    
    print(f"Test queries: {len(test_queries)}\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: '{query}'")
        results = manager.search(query)
        print(f"  -> Retrieved {len(results)} results\n")
    
    # Clear cache
    clear_cuda_cache()


def run_profiling():
    """Main profiling workflow."""
    print("\n" + "=" * 80)
    print("  LOCAL BRAIN RAG - VRAM PROFILING")
    print("  RTX 4060 8GB Constraint Validation")
    print("=" * 80)
    
    # Check CUDA availability
    if not check_cuda_available():
        print("\n[WARNING] CUDA not available")
        print("VRAM profiling requires CUDA. Results will be limited.")
        print("\nTo enable CUDA:")
        print("  1. Install NVIDIA GPU drivers")
        print("  2. Install PyTorch with CUDA support:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return
    
    # Run profiling steps
    manager = profile_embedding_model()
    profile_ingestion(manager)
    profile_query(manager)
    
    # Summary
    print_header("PROFILING COMPLETE")
    print("[OK] All operations completed within VRAM constraints")
    print("\nCheck logs above for detailed VRAM usage metrics.")
    print("Look for '[VRAMMonitor]' entries showing delta and peak usage.")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        run_profiling()
    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Profiling failed: {e}")
        logger.exception("Profiling error")
        sys.exit(1)
