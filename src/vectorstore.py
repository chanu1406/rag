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

# Conditional import for re-ranking
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    logger.warning("FlashRank not available. Install with: pip install flashrank")

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
        
        # Re-ranker (FlashRank) - Initialized if enabled and available
        self.reranker = None
        if config.retrieval.use_reranker:
            if FLASHRANK_AVAILABLE:
                logger.info("Initializing FlashRank re-ranker...")
                self.reranker = Ranker()
                logger.debug("FlashRank re-ranker initialized")
            else:
                logger.warning("Re-ranking enabled in config but FlashRank not installed")

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
        Also builds BM25 index for hybrid search.
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
                
                # Build BM25 index for hybrid search
                self.build_keyword_index(documents)
                
                logger.success(f"Successfully stored {len(ids)} vectors.")
                return ids
            except Exception as e:
                logger.error(f"Failed to add documents to ChromaDB: {e}")
                raise e

    def build_keyword_index(self, documents: List[Document]):
        """
        Builds in-memory BM25 index for Hybrid Search.
        """
        if not documents:
            return
            
        logger.info("Building BM25 keyword index...")
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = self.config.retrieval.k_retrieved
        logger.debug(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str) -> List[Document]:
        """
        Hybrid search interface with VRAM monitoring.
        Combines BM25 keyword search + vector semantic search using RRF.
        """
        k = self.config.retrieval.k_retrieved
        
        logger.debug(f"Executing hybrid search for: '{query}' (k={k})")
        
        # Get candidates from both retrievers
        with VRAMMonitor("Query Embedding"):
            vector_results = self.vectorstore.similarity_search(query, k=k)
        
        # If BM25 index exists, do hybrid retrieval
        if self.bm25:
            bm25_results = self.bm25.invoke(query)
            
            # Merge using Reciprocal Rank Fusion
            merged = self._reciprocal_rank_fusion(
                {"vector": vector_results, "bm25": bm25_results},
                k=k
            )
            logger.debug(f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(merged)} merged")
            
            # Apply re-ranking if available
            if self.reranker:
                merged = self._rerank(query, merged)
            
            return merged
        else:
            logger.debug("BM25 not available, using vector-only search")
            return vector_results
    
    def _reciprocal_rank_fusion(self, retriever_results: Dict[str, List[Document]], k: int = 60) -> List[Document]:
        """
        Merge results from multiple retrievers using Reciprocal Rank Fusion.
        
        RRF score for document d: sum over all retrievers of 1/(rank(d) + k)
        where k is a constant (default 60).
        """
        doc_scores = {}
        
        for retriever_name, docs in retriever_results.items():
            for rank, doc in enumerate(docs):
                # Use page_content as unique key
                doc_key = doc.page_content
                
                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {"doc": doc, "score": 0.0}
                
                # RRF formula
                doc_scores[doc_key]["score"] += 1.0 / (rank + k)
        
        # Sort by score descending
        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        
        return [item["doc"] for item in ranked][:self.config.retrieval.k_retrieved]
    
    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents using FlashRank cross-encoder.
        
        FlashRank uses a cross-encoder model to score query-document pairs,
        providing more accurate relevance than embedding similarity alone.
        """
        if not documents:
            return documents
        
        logger.debug(f"Re-ranking {len(documents)} documents with FlashRank...")
        
        # Prepare passages for FlashRank
        passages = [
            {"text": doc.page_content, "meta": doc.metadata}
            for doc in documents
        ]
        
        # Create rerank request
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # Get reranked results
        results = self.reranker.rerank(rerank_request)
        
        # Reconstruct documents in new order
        reranked_docs = []
        for result in results[:self.config.retrieval.k_final]:
            # Find original document by matching corpus_id
            orig_doc = documents[result["corpus_id"]]
            reranked_docs.append(orig_doc)
        
        logger.debug(f"Re-ranking complete: Top {len(reranked_docs)} kept")
        return reranked_docs

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
