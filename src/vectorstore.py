from typing import List, Any, Optional, Dict
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.config import AppConfig
from src.logger import logger
from src.utils import VRAMMonitor
import shutil

class VectorManager:
    """
    Manages ChromaDB vector store and embedding operations.
    
    Features:
    - Enforces CUDA usage for embeddings.
    - Uses explicit configuration.
    """
 
    def __init__(self, config: AppConfig): 
        self.config = config
        self.persist_dir = str(config.system.persist_dir)
        
        # Initialize Embeddings (Force CUDA)
        logger.info(f"Initializing embeddings: {config.embedding.model_name} on {config.embedding.device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding.model_name,
            model_kwargs={"device": config.embedding.device},
            encode_kwargs={"normalize_embeddings": True} # Better cosine similarity
        )
        
        # Initialize Vector Store
        self._init_chroma()
        
        # Keyword Retriever (BM25) - Initialized on demand or after ingestion
        self.bm25: Optional[BM25Retriever] = None

    def _init_chroma(self):
        """Initialize ChromaDB client."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.debug(f"ChromaDB initialized at {self.persist_dir}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Embed and store documents in ChromaDB with VRAM monitoring.
        
        Tracks VRAM usage for hardware constraints.
        """
        if not documents:
            logger.warning("No documents to add.")
            return []

        logger.info(f"Adding {len(documents)} document chunks to vector store...")
        
        # Monitor VRAM during embedding (most intensive operation)
        with VRAMMonitor("Document Embedding"):
            try:
                ids = self.vectorstore.add_documents(documents)
                self.vectors = self.vectorstore
                logger.success(f"Successfully stored {len(ids)} vectors.")
                return ids
            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}")
                raise e

    def build_keyword_index(self, documents: List[Document]):
        """
        Builds in-memory BM25 index for Hybrid Search (Phase 2).
        """
        logger.info("Building BM25 keyword index...")
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = self.config.retrieval.k_retrieved

    def search(self, query: str) -> List[Document]:
        """
        Primary search interface with VRAM monitoring.
        Currently semantic search, prepared for hybrid.
        """
        k = self.config.retrieval.k_retrieved
        
        logger.debug(f"Executing semantic search for: '{query}' (k={k})")
        
        # Monitor VRAM during query embedding
        with VRAMMonitor("Query Embedding"):
            results = self.vectorstore.similarity_search(query, k=k)
        
        # Note: Hybrid search (BM25 + vector) and re-ranking not yet implemented
        # See implementation_plan.md for future enhancements
        
        return results

    def clear(self):
        """Clear the vector store and recreate directory."""
        logger.warning(f"Deleting vector store at {self.persist_dir}")
        
        # Delete entire directory to fully clear
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        
        # Recreate directory
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Reinitialize ChromaDB with fresh instance
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.debug(f"ChromaDB reinitialized at {self.persist_dir}")
