"""
Tests for ingest.py module.

Per AGENTCONTEXT.md:
- Deterministic chunking (same input → same output)
- Metadata preservation
- Explicit error handling
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingest import DocumentLoader


class TestDocumentLoader:
    """Test DocumentLoader initialization and configuration."""
    
    @pytest.mark.unit
    def test_initialization(self, test_config):
        """Loader should initialize with config parameters."""
        loader = DocumentLoader(test_config)
        
        assert loader.chunk_size == test_config.retrieval.chunk_size
        assert loader.chunk_overlap == test_config.retrieval.chunk_overlap
        assert loader.text_splitter is not None
        assert ".pdf" in loader.valid_extensions


class TestFileLoading:
    """Test loading various file formats."""
    
    @pytest.mark.unit
    def test_load_text_file(self, test_config, sample_text_file):
        """Load a simple text file with metadata."""
        loader = DocumentLoader(test_config)
        documents = loader.load_file(sample_text_file)
        
        assert len(documents) > 0
        assert isinstance(documents[0], Document)
        
        # Check metadata preservation
        doc = documents[0]
        assert "source" in doc.metadata
        assert "filename" in doc.metadata
        assert doc.metadata["filename"] == "sample.txt"
        assert str(sample_text_file) in doc.metadata["source"]
    
    @pytest.mark.unit
    def test_load_markdown_file(self, test_config, sample_markdown_file):
        """Load a markdown file with metadata."""
        loader = DocumentLoader(test_config)
        documents = loader.load_file(sample_markdown_file)
        
        assert len(documents) > 0
        doc = documents[0]
        assert doc.metadata["filename"] == "sample.md"
        assert "# Test Document" in doc.page_content or "Section" in doc.page_content
    
    @pytest.mark.unit
    def test_load_nonexistent_file(self, test_config, temp_dir):
        """Failure path: File does not exist."""
        loader = DocumentLoader(test_config)
        fake_file = temp_dir / "doesnotexist.txt"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_file(fake_file)
        
        assert "File not found" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_unsupported_extension(self, test_config, temp_dir):
        """Return empty list for unsupported file types."""
        loader = DocumentLoader(test_config)
        
        # Create file with unsupported extension
        unsupported = temp_dir / "test.xyz"
        unsupported.write_text("content")
        
        documents = loader.load_file(unsupported)
        assert documents == []


class TestChunking:
    """Test deterministic text chunking."""
    
    @pytest.mark.unit
    def test_chunk_documents(self, test_config, sample_text_file):
        """Chunk documents with metadata assignment."""
        loader = DocumentLoader(test_config)
        documents = loader.load_file(sample_text_file)
        chunks = loader.chunk_documents(documents)
        
        assert len(chunks) > 0
        
        # Check chunk metadata
        for i, chunk in enumerate(chunks):
            assert "chunk_id" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["chunk_id"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
    
    @pytest.mark.unit
    def test_chunking_determinism(self, test_config, sample_text_file):
        """Same input must produce same chunks (determinism requirement)."""
        loader = DocumentLoader(test_config)
        documents = loader.load_file(sample_text_file)
        
        # Chunk twice
        chunks1 = loader.chunk_documents(documents)
        chunks2 = loader.chunk_documents(documents)
        
        assert len(chunks1) == len(chunks2)
        
        # Compare content and metadata
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.page_content == c2.page_content
            assert c1.metadata["chunk_id"] == c2.metadata["chunk_id"]
    
    @pytest.mark.unit
    def test_chunk_empty_documents(self, test_config):
        """Handle empty document list gracefully."""
        loader = DocumentLoader(test_config)
        chunks = loader.chunk_documents([])
        
        assert chunks == []
    
    @pytest.mark.unit
    def test_chunk_size_respected(self, test_config, temp_dir):
        """Chunks should approximately respect configured size."""
        loader = DocumentLoader(test_config)
        
        # Create long document
        long_text = "word " * 1000  # Much longer than chunk_size
        long_file = temp_dir / "long.txt"
        long_file.write_text(long_text)
        
        documents = loader.load_file(long_file)
        chunks = loader.chunk_documents(documents)
        
        # Should split into multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be roughly <= chunk_size (allowing some overlap)
        max_allowed = test_config.retrieval.chunk_size + test_config.retrieval.chunk_overlap
        for chunk in chunks:
            assert len(chunk.page_content) <= max_allowed + 100  # Small buffer for word boundaries


class TestMetadataPreservation:
    """Test that metadata flows through the pipeline."""
    
    @pytest.mark.unit
    def test_source_metadata_in_chunks(self, test_config, sample_text_file):
        """Chunks should preserve source file metadata."""
        loader = DocumentLoader(test_config)
        documents = loader.load_file(sample_text_file)
        chunks = loader.chunk_documents(documents)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "filename" in chunk.metadata
            assert chunk.metadata["filename"] == "sample.txt"
            assert "chunk_id" in chunk.metadata
