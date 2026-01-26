"""
Tests for rag.py module.

Per AGENTCONTEXT.md:
- Explicit failure modes (Ollama check)
- Clear prompt construction
- Source citation
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.rag import RAGEngine
from src.vectorstore import VectorManager


class TestRAGEngineInit:
    """Test RAG engine initialization."""
    
    @pytest.mark.ollama
    @pytest.mark.unit
    def test_initialization_with_ollama(self, test_config, ollama_available):
        """Initialize RAG engine when Ollama is running."""
        if not ollama_available:
            pytest.skip("Ollama service not running")
        
        vector_manager = VectorManager(test_config)
        rag_engine = RAGEngine(test_config, vector_manager)
        
        assert rag_engine.config == test_config
        assert rag_engine.vector_manager == vector_manager
        assert rag_engine.llm is not None
        assert rag_engine.prompt_template is not None
    
    @pytest.mark.unit
    def test_initialization_without_ollama(self, test_config):
        """Failure path: Ollama not running should raise RuntimeError."""
        vector_manager = VectorManager(test_config)
        
        # Mock Ollama reachability check to return False
        with patch.object(RAGEngine, '_is_ollama_reachable', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                RAGEngine(test_config, vector_manager)
            
            assert "Ollama service not available" in str(exc_info.value)


class TestPromptConstruction:
    """Test prompt assembly."""
    
    @pytest.mark.unit
    def test_format_context(self, test_config, ollama_available):
        """Format retrieved documents into context string."""
        if not ollama_available:
            pytest.skip("Ollama service not running")
        
        vector_manager = VectorManager(test_config)
        rag_engine = RAGEngine(test_config, vector_manager)
        
        docs = [
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"filename": "ml_intro.txt", "chunk_id": 0}
            ),
            Document(
                page_content="Neural networks are inspired by biology.",
                metadata={"filename": "nn_basics.txt", "chunk_id": 0}
            )
        ]
        
        context = rag_engine._format_context(docs)
        
        # Check formatting
        assert "ml_intro.txt" in context
        assert "nn_basics.txt" in context
        assert "Machine learning" in context
        assert "Neural networks" in context
        assert "[Source 1:" in context or "Source 1:" in context


class TestSourceExtraction:
    """Test citation metadata extraction."""
    
    @pytest.mark.unit
    def test_extract_sources(self, test_config, ollama_available):
        """Extract source metadata from documents."""
        if not ollama_available:
            pytest.skip("Ollama service not running")
        
        vector_manager = VectorManager(test_config)
        rag_engine = RAGEngine(test_config, vector_manager)
        
        docs = [
            Document(
                page_content="Content",
                metadata={"filename": "doc1.txt", "chunk_id": 0}
            ),
            Document(
                page_content="More content",
                metadata={"filename": "doc2.txt", "chunk_id": 1}
            )
        ]
        
        sources = rag_engine._extract_sources(docs)
        
        assert len(sources) == 2
        assert sources[0]["filename"] == "doc1.txt"
        assert sources[0]["chunk_id"] == 0
        assert sources[1]["filename"] == "doc2.txt"
        assert sources[1]["chunk_id"] == 1


class TestQueryExecution:
    """Test full query execution (integration-style)."""
    
    @pytest.mark.ollama
    @pytest.mark.integration
    def test_query_with_documents(self, test_config, mock_documents, ollama_available):
        """Execute query with retrieved documents."""
        if not ollama_available:
            pytest.skip("Ollama service not running")
        
        vector_manager = VectorManager(test_config)
        vector_manager.add_documents(mock_documents)
        
        rag_engine = RAGEngine(test_config, vector_manager)
        result = rag_engine.query("What is machine learning?")
        
        # Check result structure
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert len(result["answer"]) > 0
    
    @pytest.mark.ollama
    @pytest.mark.integration
    def test_query_no_documents(self, test_config, ollama_available):
        """Query with empty vector store."""
        if not ollama_available:
            pytest.skip("Ollama service not running")
        
        vector_manager = VectorManager(test_config)
        rag_engine = RAGEngine(test_config, vector_manager)
        
        result = rag_engine.query("Test question")
        
        # Should return graceful message
        assert "answer" in result
        assert "No relevant documents" in result["answer"]
        assert result["sources"] == []


class TestErrorHandling:
    """Test error handling in RAG pipeline."""
    
    @pytest.mark.unit
    def test_ollama_check_explicit_error(self, test_config):
        """Ollama check should provide actionable error."""
        vector_manager = VectorManager(test_config)
        
        with patch.object(RAGEngine, '_is_ollama_reachable', return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                RAGEngine(test_config, vector_manager)
            
            error_msg = str(exc_info.value)
            # Per AGENTCONTEXT.md: errors must explain what, why, how to fix
            assert "Ollama" in error_msg
            assert "not available" in error_msg or "service" in error_msg
