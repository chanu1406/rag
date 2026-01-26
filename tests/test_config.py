"""
Tests for config.py module.

Per AGENTCONTEXT.md:
- Test happy paths AND failure paths
- Explicit, actionable errors
- Deterministic behavior
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.config import (
    AppConfig,
    SystemConfig,
    LLMConfig,
    EmbeddingConfig,
    RetrievalConfig,
    IngestionConfig,
    load_config
)


class TestConfigLoading:
    """Test configuration loading from YAML files."""
    
    @pytest.mark.unit
    def test_load_valid_config(self, test_config_yaml):
        """Happy path: Load valid configuration file."""
        config = load_config(str(test_config_yaml))
        
        assert isinstance(config, AppConfig)
        assert config.system.version == "1.0.0"
        assert config.llm.model == "llama3"
        assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.retrieval.chunk_size == 200
    
    @pytest.mark.unit
    def test_load_missing_config_file(self, temp_dir):
        """Failure path: Config file does not exist."""
        missing_path = temp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(str(missing_path))
        
        assert "Config file missing" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_load_invalid_yaml_syntax(self, temp_dir):
        """Failure path: Malformed YAML syntax."""
        bad_yaml = temp_dir / "bad.yaml"
        bad_yaml.write_text("system:\n  version: [invalid yaml syntax}}")
        
        # Should raise yaml.YAMLError or similar
        with pytest.raises(Exception):
            load_config(str(bad_yaml))
    
    @pytest.mark.unit
    def test_load_missing_required_fields(self, temp_dir):
        """Failure path: Config missing required top-level sections."""
        partial_config = {
            "system": {
                "version": "1.0.0",
                "log_level": "INFO",
                "data_dir": str(temp_dir / "data"),
                "persist_dir": str(temp_dir / "data" / "chroma_db")
            }
            # Missing: llm, embedding, retrieval, ingestion
        }
        
        config_path = temp_dir / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(partial_config, f)
        
        # load_config wraps ValidationError in ValueError
        with pytest.raises((ValueError, ValidationError)):
            load_config(str(config_path))


class TestSystemConfig:
    """Test SystemConfig validation."""
    
    @pytest.mark.unit
    def test_create_directories_on_validation(self, temp_dir):
        """Path validation should create missing directories."""
        data_dir = temp_dir / "new_data"
        persist_dir = temp_dir / "new_data" / "chroma"
        
        config = SystemConfig(
            version="1.0.0",
            log_level="INFO",
            data_dir=data_dir,
            persist_dir=persist_dir
        )
        
        # Directories should be created by validator
        assert config.data_dir.exists()
        assert config.persist_dir.exists()


class TestLLMConfig:
    """Test LLM configuration validation."""
    
    @pytest.mark.unit
    def test_context_window_constraint(self):
        """VRAM constraint: context_window must be <= 8192."""
        # Valid: within constraint
        valid_config = LLMConfig(
            provider="ollama",
            model="llama3",
            base_url="http://localhost:11434",
            context_window=4096,
            temperature=0.5
        )
        assert valid_config.context_window == 4096
        
        # Invalid: exceeds RTX 4060 limit
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="ollama",
                model="llama3",
                base_url="http://localhost:11434",
                context_window=16384,  # Too large!
                temperature=0.5
            )
        
        error_msg = str(exc_info.value)
        assert "8192" in error_msg or "less than or equal" in error_msg
    
    @pytest.mark.unit
    def test_temperature_range(self):
        """Temperature must be between 0.0 and 1.0."""
        # Valid temperatures
        for temp in [0.0, 0.5, 1.0]:
            config = LLMConfig(
                provider="ollama",
                model="llama3",
                base_url="http://localhost:11434",
                context_window=4096,
                temperature=temp
            )
            assert config.temperature == temp
        
        # Invalid: negative
        with pytest.raises(ValidationError):
            LLMConfig(
                provider="ollama",
                model="llama3",
                base_url="http://localhost:11434",
                context_window=4096,
                temperature=-0.1
            )
        
        # Invalid: > 1.0
        with pytest.raises(ValidationError):
            LLMConfig(
                provider="ollama",
                model="llama3",
                base_url="http://localhost:11434",
                context_window=4096,
                temperature=1.5
            )


class TestEmbeddingConfig:
    """Test embedding configuration."""
    
    @pytest.mark.unit
    def test_valid_embedding_config(self):
        """Create valid embedding configuration."""
        config = EmbeddingConfig(
            provider="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda"
        )
        
        assert config.provider == "huggingface"
        assert config.device == "cuda"


class TestRetrievalConfig:
    """Test retrieval configuration."""
    
    @pytest.mark.unit
    def test_chunk_overlap_less_than_size(self):
        """Chunk overlap should be less than chunk size for correctness."""
        config = RetrievalConfig(
            chunk_size=500,
            chunk_overlap=50,
            k_retrieved=20,
            k_final=5,
            use_reranker=True
        )
        
        # This is a sanity check, not enforced by Pydantic
        assert config.chunk_overlap < config.chunk_size


class TestIngestionConfig:
    """Test ingestion configuration."""
    
    @pytest.mark.unit
    def test_valid_extensions_list(self):
        """Valid extensions should be a list of strings."""
        config = IngestionConfig(
            valid_extensions=[".pdf", ".txt", ".md"],
            ignore_patterns=["__pycache__", ".git"]
        )
        
        assert ".pdf" in config.valid_extensions
        assert len(config.ignore_patterns) == 2
