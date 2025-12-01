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
    config_file = Path(config_path)

    # Check if config file exists
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config.yaml exists in the config/ directory."
        )

    # Load YAML configuration
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML configuration file: {config_path}\n"
            f"Details: {str(e)}"
        )

    # Validate required top-level keys
    required_keys = ['llm', 'embeddings', 'vectorstore', 'document_processing', 'rag']
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(
            f"Configuration file is missing required keys: {', '.join(missing_keys)}\n"
            f"Please check your config.yaml file."
        )

    # Merge with environment variables if they exist
    config = _merge_env_variables(config)

    return config


def check_cuda_available() -> bool:
    """
    Check if CUDA is available and print GPU information.

    Returns:
        True if CUDA is available, False otherwise
    """
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA is not available!")
        print("   The application will run on CPU, which will be significantly slower.")
        print("   Please ensure:")
        print("   1. NVIDIA GPU drivers are installed")
        print("   2. PyTorch with CUDA support is installed")
        print("   3. CUDA toolkit is properly configured")
        return False

    # CUDA is available - print detailed information
    print("✓ CUDA is available")
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")

    # Get VRAM information
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    print(f"  Total VRAM: {total_memory:.2f} GB")

    # Check if this is the expected RTX 4060
    gpu_name = torch.cuda.get_device_name(0)
    if "4060" in gpu_name:
        print(f"  ✓ Detected RTX 4060 - optimal configuration loaded")
    else:
        print(f"  ⚠️  Expected RTX 4060, but detected: {gpu_name}")
        print(f"     The configuration is optimized for 8GB VRAM.")

    # Get compute capability
    capability = torch.cuda.get_device_capability(0)
    print(f"  Compute Capability: {capability[0]}.{capability[1]}")

    return True


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
    cuda_available = torch.cuda.is_available()

    if force_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required but not available!\n"
            "This application is optimized for NVIDIA RTX 4060 (8GB VRAM).\n"
            "Please ensure:\n"
            "  1. NVIDIA GPU drivers are installed\n"
            "  2. PyTorch with CUDA support is installed:\n"
            "     pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "  3. CUDA toolkit is properly configured\n"
            "\nAlternatively, set force_cuda=False to run on CPU (not recommended)."
        )

    if cuda_available:
        device = torch.device('cuda')
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        logger.warning("Using device: CPU (this will be significantly slower)")

    return device


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories if they don't exist.

    Args:
        config: Configuration dictionary containing paths
    """
    # Extract paths from config, with defaults
    paths_config = config.get('paths', {})
    data_dir = Path(paths_config.get('data_dir', './data'))
    raw_dir = Path(paths_config.get('raw_documents', './data/raw'))
    processed_dir = Path(paths_config.get('processed_documents', './data/processed'))
    vectorstore_dir = Path(paths_config.get('vectorstore', './data/vectorstore'))
    logs_dir = Path(paths_config.get('logs', './logs'))

    # Also get vectorstore persist directory (might be different)
    vectorstore_persist = Path(config.get('vectorstore', {}).get('persist_directory', './data/vectorstore'))

    # List of directories to create
    directories = [
        data_dir,
        raw_dir,
        processed_dir,
        vectorstore_dir,
        vectorstore_persist,
        logs_dir
    ]

    # Create directories
    created_dirs = []
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(directory))

    # Create .gitkeep files in data directories to track empty folders in git
    gitkeep_dirs = [raw_dir, processed_dir, vectorstore_dir]
    for directory in gitkeep_dirs:
        gitkeep_file = directory / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()

    # Log created directories
    if created_dirs:
        logger.info(f"Created directories: {', '.join(created_dirs)}")
    else:
        logger.debug("All required directories already exist")


def clear_cuda_cache() -> None:
    """
    Clear PyTorch CUDA cache to free up VRAM.
    Critical for 8GB VRAM management.
    """
    if not torch.cuda.is_available():
        return

    # Get memory before clearing
    allocated_before = torch.cuda.memory_allocated() / (1024**2)  # MB

    # Clear cache
    torch.cuda.empty_cache()

    # Get memory after clearing
    allocated_after = torch.cuda.memory_allocated() / (1024**2)  # MB
    freed = allocated_before - allocated_after

    if freed > 0:
        logger.debug(f"Cleared CUDA cache: freed {freed:.2f} MB")


def estimate_vram_usage(model_name: str) -> Dict[str, float]:
    """
    Estimate VRAM usage for embedding models.
    Helpful for 8GB VRAM planning.

    Args:
        model_name: Name of the sentence-transformer model

    Returns:
        Dictionary with estimated VRAM in GB and recommended batch size
    """
    # Lookup table for common embedding models
    # Values are empirically measured on RTX 4060
    model_specs = {
        'all-MiniLM-L6-v2': {
            'vram_gb': 0.5,
            'recommended_batch_size': 64,
            'dimensions': 384
        },
        'all-MiniLM-L12-v2': {
            'vram_gb': 0.8,
            'recommended_batch_size': 48,
            'dimensions': 384
        },
        'all-mpnet-base-v2': {
            'vram_gb': 1.2,
            'recommended_batch_size': 32,
            'dimensions': 768
        },
        'multi-qa-MiniLM-L6-cos-v1': {
            'vram_gb': 0.5,
            'recommended_batch_size': 64,
            'dimensions': 384
        },
        'paraphrase-MiniLM-L3-v2': {
            'vram_gb': 0.3,
            'recommended_batch_size': 96,
            'dimensions': 384
        }
    }

    # Return specs or default estimate
    if model_name in model_specs:
        return model_specs[model_name]
    else:
        # Default conservative estimate
        logger.warning(f"Unknown model '{model_name}', using default VRAM estimate")
        return {
            'vram_gb': 1.0,
            'recommended_batch_size': 32,
            'dimensions': 768
        }


def format_documents_for_context(documents: list) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Args:
        documents: List of Document objects from vector store

    Returns:
        Formatted string containing document contents
    """
    if not documents:
        return "No relevant documents found."

    formatted_parts = []

    for i, doc in enumerate(documents, 1):
        # Extract content and metadata
        content = doc.page_content
        metadata = doc.metadata

        # Get source information
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')

        # Format document section
        section = f"[Document {i}]\n"
        section += f"Source: {source} (Page {page})\n"
        section += f"Content:\n{content}\n"
        section += "-" * 80

        formatted_parts.append(section)

    return "\n\n".join(formatted_parts)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure loguru logger based on config settings.

    Args:
        config: Configuration dictionary with logging parameters
    """
    from loguru import logger as loguru_logger

    # Get logging configuration
    logging_config = config.get('logging', {})
    log_level = logging_config.get('level', 'INFO')
    log_format = logging_config.get('format',
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )
    log_file = logging_config.get('log_file', './logs/local_brain.log')
    rotation = logging_config.get('rotation', '10 MB')
    retention = logging_config.get('retention', '1 week')

    # Remove default logger
    loguru_logger.remove()

    # Add console logger with colors
    loguru_logger.add(
        sink=lambda msg: print(msg, end=''),
        format=log_format,
        level=log_level,
        colorize=True
    )

    # Ensure log directory exists
    log_path = Path(log_file).parent
    log_path.mkdir(parents=True, exist_ok=True)

    # Add file logger with rotation
    loguru_logger.add(
        sink=log_file,
        format=log_format,
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True  # Async logging
    )

    loguru_logger.info(f"Logging initialized: level={log_level}, file={log_file}")


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
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all ops to complete
            self.start_memory = torch.cuda.memory_allocated()
            logger.debug(f"[{self.operation_name}] Started - VRAM: {self.start_memory / (1024**2):.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and report VRAM usage."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all ops to complete
            self.end_memory = torch.cuda.memory_allocated()

            # Calculate usage
            delta_mb = (self.end_memory - self.start_memory) / (1024**2)
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

            # Log results
            logger.info(
                f"[{self.operation_name}] Completed - "
                f"VRAM used: {delta_mb:+.2f} MB, "
                f"Peak: {peak_mb:.2f} MB"
            )

            # Reset peak memory stats for next operation
            torch.cuda.reset_peak_memory_stats()

        return False  # Don't suppress exceptions

    def get_usage_mb(self) -> float:
        """Get VRAM usage in MB."""
        return (self.end_memory - self.start_memory) / (1024**2)


# ============================================================================
# Private Helper Functions
# ============================================================================

def _merge_env_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge environment variables into configuration.
    Environment variables take precedence over config file values.

    Args:
        config: Configuration dictionary from YAML

    Returns:
        Updated configuration with environment variables merged
    """
    import os

    # LLM configuration overrides
    if os.getenv('OLLAMA_HOST'):
        config['llm']['base_url'] = os.getenv('OLLAMA_HOST')
    if os.getenv('OLLAMA_MODEL'):
        config['llm']['model_name'] = os.getenv('OLLAMA_MODEL')

    # Embedding configuration overrides
    if os.getenv('EMBEDDING_MODEL'):
        config['embeddings']['model_name'] = os.getenv('EMBEDDING_MODEL')
    if os.getenv('EMBEDDING_DEVICE'):
        config['embeddings']['device'] = os.getenv('EMBEDDING_DEVICE')

    # ChromaDB configuration overrides
    if os.getenv('CHROMA_PERSIST_DIRECTORY'):
        config['vectorstore']['persist_directory'] = os.getenv('CHROMA_PERSIST_DIRECTORY')
    if os.getenv('CHROMA_COLLECTION_NAME'):
        config['vectorstore']['collection_name'] = os.getenv('CHROMA_COLLECTION_NAME')

    # CUDA configuration
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        # This is handled by PyTorch automatically, but we can log it
        pass

    return config
