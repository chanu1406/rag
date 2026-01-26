"""
VRAM Utilities for Local Brain RAG.

Optimized for RTX 4060 (8GB VRAM) constraints.
Provides explicit monitoring and predictable performance.
"""

import torch
from pathlib import Path
from typing import Dict
from loguru import logger


def check_cuda_available() -> bool:
    """
    Check if CUDA is available and print GPU information.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available")
        logger.warning("The application will run on CPU (significantly slower)")
        logger.warning("Ensure: 1) NVIDIA GPU drivers installed, 2) PyTorch with CUDA support")
        return False
    
    # CUDA is available - print detailed information
    logger.info("CUDA is available")
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    # Get VRAM information
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"Total VRAM: {total_memory:.2f} GB")
    
    # Check if this is the expected RTX 4060
    gpu_name = torch.cuda.get_device_name(0)
    if "4060" in gpu_name:
        logger.success("Detected RTX 4060 - optimal configuration")
    else:
        logger.warning(f"Expected RTX 4060, but detected: {gpu_name}")
        logger.warning("Configuration is optimized for 8GB VRAM")
    
    # Get compute capability
    capability = torch.cuda.get_device_capability(0)
    logger.info(f"Compute Capability: {capability[0]}.{capability[1]}")
    
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


def clear_cuda_cache() -> None:
    """
    Clear PyTorch CUDA cache to free up VRAM.
    Critical for 8GB VRAM management on RTX 4060.
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
    Helpful for 8GB VRAM planning on RTX 4060.
    
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


class VRAMMonitor:
    """
    Context manager for monitoring VRAM usage during operations.
    Useful for profiling and optimization on RTX 4060.
    
    Provides explicit, inspectable performance metrics.
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
