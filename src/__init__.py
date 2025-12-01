"""
Local Brain RAG - Main Package
Optimized for NVIDIA RTX 4060 (8GB VRAM)
"""

__version__ = "0.1.0"
__author__ = "Local Brain Team"

from .ingest import DocumentLoader
from .vectorstore import VectorManager
from .rag import RAGEngine

__all__ = [
    "DocumentLoader",
    "VectorManager",
    "RAGEngine",
]
