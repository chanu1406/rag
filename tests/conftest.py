"""
Pytest fixtures for Local Brain RAG tests.

Per AGENTCONTEXT.md:
- Clear, inspectable data flow
- Explicit failure modes
- Deterministic behavior
"""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AppConfig, load_config


@pytest.fixture
def temp_dir():
    """Create temporary directory for test isolation."""
    tmp = tempfile.mkdtemp(prefix="local_brain_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def test_config_dict(temp_dir: Path) -> Dict[str, Any]:
    """
    Minimal valid configuration dictionary.
    
    Per CONTEXT.md: Config-driven, no magic numbers.
    """
    return {
        "system": {
            "version": "1.0.0",
            "log_level": "ERROR",  # Quiet during tests
            "data_dir": str(temp_dir / "data"),
            "persist_dir": str(temp_dir / "data" / "chroma_db")
        },
        "llm": {
            "provider": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434",
            "context_window": 4096,
            "temperature": 0.0
        },
        "embedding": {
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu"  # Default to CPU for tests
        },
        "retrieval": {
            "chunk_size": 200,  # Smaller for tests
            "chunk_overlap": 20,
            "k_retrieved": 5,
            "k_final": 3,
            "use_reranker": False
        },
        "ingestion": {
            "valid_extensions": [".pdf", ".txt", ".md", ".docx"],
            "ignore_patterns": ["__pycache__", ".git"]
        }
    }


@pytest.fixture
def test_config_yaml(temp_dir: Path, test_config_dict: Dict[str, Any]) -> Path:
    """Write test config to YAML file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config_dict, f)
    return config_path


@pytest.fixture
def test_config(test_config_yaml: Path) -> AppConfig:
    """Load validated AppConfig for tests."""
    return load_config(str(test_config_yaml))


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create sample text file for ingestion tests."""
    file_path = temp_dir / "sample.txt"
    content = """This is a test document for the Local Brain RAG system.
It contains multiple paragraphs to test chunking behavior.

The system runs on RTX 4060 with strict VRAM constraints.
All operations must be deterministic and config-driven.

This document should split into multiple chunks based on chunk_size."""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create sample markdown file."""
    file_path = temp_dir / "sample.md"
    content = """# Test Document

## Section 1
This is test content in markdown format.

## Section 2
More content for testing chunking and metadata preservation.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def mock_documents():
    """
    Mock LangChain documents for testing.
    
    Per AGENTCONTEXT.md: Avoid loosely-typed dictionaries.
    """
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="First chunk of content about machine learning.",
            metadata={"source": "/test/doc1.txt", "filename": "doc1.txt", "chunk_id": 0}
        ),
        Document(
            page_content="Second chunk discussing neural networks.",
            metadata={"source": "/test/doc1.txt", "filename": "doc1.txt", "chunk_id": 1}
        ),
        Document(
            page_content="Third chunk about transformers and attention.",
            metadata={"source": "/test/doc2.txt", "filename": "doc2.txt", "chunk_id": 0}
        )
    ]


# Conditional fixtures for GPU/Ollama tests

@pytest.fixture
def cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def ollama_available() -> bool:
    """Check if Ollama service is reachable."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring CUDA/GPU"
    )
    config.addinivalue_line(
        "markers", "ollama: mark test as requiring Ollama service"
    )
