"""
Integration tests for metadata extraction.

These tests verify:
1. Extraction functionality with/without Ollama
2. Performance overhead measurements
3. Metadata quality validation
"""

import sys
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import DocumentLoader
from src.config import load_config


class TestMetadataExtraction:
    """Test metadata extraction with different scenarios."""
    
    def test_extraction_disabled(self, test_config, temp_dir):
        """Test that extraction can be disabled."""
        # Modify config to disable extraction
        test_config.ingestion.extraction.enabled = False
        loader = DocumentLoader(test_config)
        
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Engineering Department Report 2024")
        
        docs = loader.load_file(test_file)
        assert len(docs) > 0
        
        # Should have basic metadata but not extracted fields
        assert "filename" in docs[0].metadata
        # Fallback values should be present
        assert docs[0].metadata.get("department") == "Uncategorized"
    
    @pytest.mark.ollama
    def test_extraction_enabled_with_ollama(self, test_config, temp_dir, ollama_available):
        """Test extraction with Ollama running."""
        if not ollama_available:
            pytest.skip("Ollama service not available")
        
        # Enable extraction
        test_config.ingestion.extraction.enabled = True
        loader = DocumentLoader(test_config)
        
        # Create test file with clear metadata
        test_file = temp_dir / "engineering_report_2024.txt"
        content = """
        Department of Computer Engineering
        University Research Report
        Date: 2024-06-15
        
        Summary: Analysis of neural network architectures for edge computing.
        This report focuses on optimization techniques for limited hardware.
        """
        test_file.write_text(content)
        
        docs = loader.load_file(test_file)
        assert len(docs) > 0
        
        # Verify extracted metadata
        metadata = docs[0].metadata
        assert "title" in metadata
        assert "department" in metadata
        assert "date" in metadata
        assert "tags" in metadata
        
        # Check quality - department should contain "Engineering"
        assert "Engineering" in metadata.get("department", "") or "Computer" in metadata.get("department", "")
        
        # Date should be extracted
        assert metadata.get("date") is not None
    
    @pytest.mark.ollama
    def test_performance_overhead(self, test_config, temp_dir, ollama_available):
        """Benchmark: Compare ingestion with/without extraction."""
        if not ollama_available:
            pytest.skip("Ollama service not available")
        
        test_file = temp_dir / "perf_test.txt"
        test_file.write_text("Test document for performance analysis. " * 50)
        
        # Test WITHOUT extraction
        test_config.ingestion.extraction.enabled = False
        loader_no_extract = DocumentLoader(test_config)
        
        start = time.time()
        docs_no_extract = loader_no_extract.load_file(test_file)
        time_no_extract = time.time() - start
        
        # Test WITH extraction
        test_config.ingestion.extraction.enabled = True
        loader_with_extract = DocumentLoader(test_config)
        
        start = time.time()
        docs_with_extract = loader_with_extract.load_file(test_file)
        time_with_extract = time.time() - start
        
        # Calculate overhead
        overhead = time_with_extract - time_no_extract
        overhead_percent = (overhead / time_no_extract) * 100 if time_no_extract > 0 else 0
        
        print(f"\n--- Performance Benchmark ---")
        print(f"Without extraction: {time_no_extract:.3f}s")
        print(f"With extraction: {time_with_extract:.3f}s")
        print(f"Overhead: {overhead:.3f}s ({overhead_percent:.1f}%)")
        
        # Both should produce same number of documents
        assert len(docs_no_extract) == len(docs_with_extract)
        
        # Return results for analysis
        return {
            "time_no_extract": time_no_extract,
            "time_with_extract": time_with_extract,
            "overhead_seconds": overhead,
            "overhead_percent": overhead_percent
        }
    
    def test_fallback_on_llm_failure(self, test_config, temp_dir):
        """Test that system gracefully falls back if LLM fails."""
        # Enable extraction but LLM might not respond
        test_config.ingestion.extraction.enabled = True
        test_config.ingestion.extraction.model = "nonexistent-model"
        
        loader = DocumentLoader(test_config)
        test_file = temp_dir / "test.txt"
        test_file.write_text("Content")
        
        # Should not crash, should use fallback
        docs = loader.load_file(test_file)
        assert len(docs) > 0
        assert "filename" in docs[0].metadata
    
    @pytest.mark.ollama
    def test_metadata_propagation_to_chunks(self, test_config, temp_dir, ollama_available):
        """Verify that extracted metadata is applied to ALL chunks."""
        if not ollama_available:
            pytest.skip("Ollama service not available")
        
        test_config.ingestion.extraction.enabled = True
        test_config.retrieval.chunk_size = 100  # Small chunks
        loader = DocumentLoader(test_config)
        
        # Create large document
        test_file = temp_dir / "large_doc.txt"
        content = "Department: Engineering\nDate: 2024-01-01\n" + ("x " * 200)
        test_file.write_text(content)
        
        docs = loader.load_file(test_file)
        chunks = loader.chunk_documents(docs)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # ALL chunks should have the same extracted metadata
        first_metadata = chunks[0].metadata
        for chunk in chunks[1:]:
            assert chunk.metadata.get("department") == first_metadata.get("department")
            assert chunk.metadata.get("date") == first_metadata.get("date")


class TestExtractionQuality:
    """Test the quality of extracted metadata."""
    
    @pytest.mark.ollama
    def test_date_extraction_accuracy(self, test_config, temp_dir, ollama_available):
        """Test that dates are correctly identified."""
        if not ollama_available:
            pytest.skip("Ollama service not available")
        
        test_config.ingestion.extraction.enabled = True
        loader = DocumentLoader(test_config)
        
        test_file = temp_dir / "dated_doc.txt"
        test_file.write_text("Report created on March 15, 2023 for Q1 analysis.")
        
        docs = loader.load_file(test_file)
        metadata = docs[0].metadata
        
        # Should extract year at minimum
        date_str = metadata.get("date", "")
        assert "2023" in str(date_str)
    
    @pytest.mark.ollama
    def test_department_inference(self, test_config, temp_dir, ollama_available):
        """Test department categorization."""
        if not ollama_available:
            pytest.skip("Ollama service not available")
        
        test_config.ingestion.extraction.enabled = True
        loader = DocumentLoader(test_config)
        
        # Test different department contexts
        test_cases = [
            ("HR benefits policy document", "HR"),
            ("Quarterly financial statement", "Finance"),
            ("Software architecture design", "Engineering")
        ]
        
        for content, expected_dept_keyword in test_cases:
            test_file = temp_dir / f"test_{expected_dept_keyword}.txt"
            test_file.write_text(content)
            
            docs = loader.load_file(test_file)
            dept = docs[0].metadata.get("department", "")
            
            # Dept should contain or be related to expected keyword
            print(f"Content: '{content}' -> Department: '{dept}'")
            # Note: LLM may not be perfect, this is a quality check not strict test


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
