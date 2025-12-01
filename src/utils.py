"""
Utility Functions for Local Brain RAG
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def load_config(config_path: str = "./config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    # TODO: Implement config loading logic
    # TODO: Add validation for required fields
    # TODO: Merge with environment variables if needed
    pass


def check_cuda_available() -> bool:
    """
    Check if CUDA is available and print GPU information.

    Returns:
        True if CUDA is available, False otherwise
    """
    # TODO: Check torch.cuda.is_available()
    # TODO: Print GPU name, memory, CUDA version
    # TODO: Warn if CUDA is not available
    # TODO: Return availability status
    pass


def get_device(force_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device (CUDA/CPU) for model execution.

    Args:
        force_cuda: If True, raise error when CUDA is unavailable

    Returns:
        torch.device object

    Raises:
        RuntimeError: If force_cuda=True and CUDA is unavailable
    """
    # TODO: Check CUDA availability
    # TODO: If force_cuda and no CUDA, raise RuntimeError
    # TODO: Log selected device
    # TODO: Return torch.device('cuda') or torch.device('cpu')
    pass


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories if they don't exist.

    Args:
        config: Configuration dictionary containing paths
    """
    # TODO: Extract paths from config
    # TODO: Create directories: data/raw, data/processed, data/vectorstore, logs
    # TODO: Create .gitkeep files in data directories
    # TODO: Log created directories
    pass


def clear_cuda_cache() -> None:
    """
    Clear PyTorch CUDA cache to free up VRAM.
    Critical for 8GB VRAM management.
    """
    # TODO: Check if CUDA is available
    # TODO: Call torch.cuda.empty_cache()
    # TODO: Optionally log freed memory
    pass


def estimate_vram_usage(model_name: str) -> Dict[str, float]:
    """
    Estimate VRAM usage for embedding models.
    Helpful for 8GB VRAM planning.

    Args:
        model_name: Name of the sentence-transformer model

    Returns:
        Dictionary with estimated VRAM in GB and recommended batch size
    """
    # TODO: Create lookup table for common models
    # TODO: Return estimated VRAM and recommended batch size
    # Example:
    # {
    #     'vram_gb': 1.2,
    #     'recommended_batch_size': 32
    # }
    pass


def format_documents_for_context(documents: list) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Args:
        documents: List of Document objects from vector store

    Returns:
        Formatted string containing document contents
    """
    # TODO: Extract page_content from each document
    # TODO: Add document metadata (filename, page number)
    # TODO: Format as numbered list or separated sections
    # TODO: Return formatted string
    pass


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure loguru logger based on config settings.

    Args:
        config: Configuration dictionary with logging parameters
    """
    # TODO: Remove default logger
    # TODO: Add console logger with custom format
    # TODO: Add file logger with rotation
    # TODO: Set log level from config
    pass


class VRAMMonitor:
    """
    Context manager for monitoring VRAM usage during operations.
    Useful for profiling and optimization on RTX 4060.
    """

    def __init__(self, operation_name: str):
        """
        Initialize VRAM monitor.

        Args:
            operation_name: Name of the operation being monitored
        """
        self.operation_name = operation_name
        self.start_memory = 0
        self.end_memory = 0

    def __enter__(self):
        """Start monitoring VRAM usage."""
        # TODO: Record current VRAM usage with torch.cuda.memory_allocated()
        # TODO: Log start of operation
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and report VRAM usage."""
        # TODO: Record final VRAM usage
        # TODO: Calculate delta
        # TODO: Log VRAM used by operation
        # TODO: Clear cache if configured
        pass

    def get_usage_mb(self) -> float:
        """Get VRAM usage in MB."""
        # TODO: Return (end_memory - start_memory) in MB
        pass
