from typing import List, Any, Optional, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.config import AppConfig
from src.logger import logger
import shutil

class VectorManager:
    """
    Manages ChromaDB vector store and embedding operations.
    
    Adheres to AGENTCONTEXT.md:
    - Enforces CUDA usage for embeddings (RTX 4060).
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
        Embed and store documents in ChromaDB.
        """
        if not documents:
            logger.warning("No documents to add.")
            return []

        logger.info(f"Adding {len(documents)} document chunks to vector store...")
        
        # Add to Chroma (processing happens here via HuggingFaceEmbeddings)
        try:
            ids = self.vectorstore.add_documents(documents)
            self.vectors = self.vectorstore # storage persists automatically in newer versions
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
        Primary search interface. 
        Currently mostly semantic, prepared for Hybrid.
        """
        k = self.config.retrieval.k_retrieved
        
        # Semantic Search (Vector)
        logger.debug(f"Executing semantic search for: '{query}' (k={k})")
        results = self.vectorstore.similarity_search(query, k=k)
        
        # TODO: Phase 2 - Merge with self.bm25.get_relevant_documents(query)
        # TODO: Phase 2 - Apply FlashRank re-ranking here
        
        return results

    def clear(self):
        """Nu-uke the vector store."""
        logger.warning(f"Deleting vector store at {self.persist_dir}")
        self.vectorstore = None
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self.config.system.validate_paths(self.persist_dir) # Recreate dir
        self._init_chroma()
