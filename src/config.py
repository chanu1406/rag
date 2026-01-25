import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, validator
from loguru import logger

class SystemConfig(BaseModel):
    version: str
    log_level: str = "INFO"
    data_dir: Path
    persist_dir: Path

    @validator("data_dir", "persist_dir")
    def validate_paths(cls, v):
        path = Path(v)
        if not path.exists():
            logger.info(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        return path

class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    context_window: int = Field(..., le=8192) # Hardware constraint safeguard
    temperature: float = Field(..., ge=0.0, le=1.0)

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str
    device: str

class RetrievalConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    k_retrieved: int
    k_final: int
    use_reranker: bool

class IngestionConfig(BaseModel):
    valid_extensions: List[str]
    ignore_patterns: List[str]

class AppConfig(BaseModel):
    system: SystemConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    ingestion: IngestionConfig

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """
    Load and validate configuration from YAML file.
    Fails hard if configuration is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found at: {path.absolute()}")
        raise FileNotFoundError(f"Config file missing: {path}")

    try:
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        
        config = AppConfig(**raw_config)
        logger.debug("Configuration loaded and validated successfully.")
        return config
    
    except ValidationError as e:
        logger.error("Configuration validation failed!")
        for error in e.errors():
            logger.error(f"Field: {error['loc']} - Error: {error['msg']}")
        raise ValueError("Invalid configuration. Check logs for details.")
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        raise
