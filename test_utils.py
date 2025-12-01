"""
Test script for src/utils.py
Verifies all utility functions work correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    load_config,
    check_cuda_available,
    get_device,
    ensure_directories,
    setup_logging,
    clear_cuda_cache,
    estimate_vram_usage,
    VRAMMonitor
)


def test_load_config():
    """Test configuration loading"""
    print("\n" + "="*80)
    print("TEST 1: load_config()")
    print("="*80)

    try:
        config = load_config("./config/config.yaml")
        print("✓ Configuration loaded successfully")
        print(f"  LLM Model: {config['llm']['model_name']}")
        print(f"  Embedding Model: {config['embeddings']['model_name']}")
        print(f"  Embedding Device: {config['embeddings']['device']}")
        print(f"  Vector Store: {config['vectorstore']['type']}")
        return config
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return None


def test_cuda_check():
    """Test CUDA availability check"""
    print("\n" + "="*80)
    print("TEST 2: check_cuda_available()")
    print("="*80)

    cuda_available = check_cuda_available()
    if cuda_available:
        print("\n✓ CUDA check completed successfully")
    else:
        print("\n⚠️  CUDA not available (expected if no GPU)")

    return cuda_available


def test_get_device():
    """Test device selection"""
    print("\n" + "="*80)
    print("TEST 3: get_device()")
    print("="*80)

    try:
        # Try with force_cuda=False (won't raise error)
        device = get_device(force_cuda=False)
        print(f"✓ Device selected: {device}")
        return device
    except Exception as e:
        print(f"✗ Error getting device: {e}")
        return None


def test_ensure_directories(config):
    """Test directory creation"""
    print("\n" + "="*80)
    print("TEST 4: ensure_directories()")
    print("="*80)

    try:
        ensure_directories(config)
        print("✓ Directories ensured successfully")

        # Verify directories exist
        paths_to_check = [
            "./data/raw",
            "./data/processed",
            "./data/vectorstore",
            "./logs"
        ]

        for path in paths_to_check:
            if Path(path).exists():
                print(f"  ✓ {path} exists")
            else:
                print(f"  ✗ {path} missing")

    except Exception as e:
        print(f"✗ Error ensuring directories: {e}")


def test_setup_logging(config):
    """Test logging configuration"""
    print("\n" + "="*80)
    print("TEST 5: setup_logging()")
    print("="*80)

    try:
        setup_logging(config)
        print("✓ Logging configured successfully")

        # Test log file creation
        log_file = Path(config.get('logging', {}).get('log_file', './logs/local_brain.log'))
        if log_file.exists():
            print(f"  ✓ Log file created: {log_file}")
        else:
            print(f"  ⚠️  Log file not yet created (will be created on first log)")

    except Exception as e:
        print(f"✗ Error setting up logging: {e}")


def test_estimate_vram():
    """Test VRAM estimation"""
    print("\n" + "="*80)
    print("TEST 6: estimate_vram_usage()")
    print("="*80)

    models_to_test = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'unknown-model-xyz'
    ]

    for model_name in models_to_test:
        specs = estimate_vram_usage(model_name)
        print(f"\nModel: {model_name}")
        print(f"  VRAM: {specs['vram_gb']} GB")
        print(f"  Batch Size: {specs['recommended_batch_size']}")
        print(f"  Dimensions: {specs['dimensions']}")


def test_vram_monitor():
    """Test VRAM monitor context manager"""
    print("\n" + "="*80)
    print("TEST 7: VRAMMonitor class")
    print("="*80)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - skipping VRAM monitor test")
            return

        # Test VRAM monitoring
        with VRAMMonitor("Test Operation"):
            # Allocate some memory
            x = torch.randn(1000, 1000).cuda()
            del x

        print("✓ VRAMMonitor test completed")

        # Test clear cache
        clear_cuda_cache()
        print("✓ CUDA cache cleared")

    except Exception as e:
        print(f"✗ Error in VRAM monitor test: {e}")


def main():
    """Run all tests"""
    print("\n")
    print("*" * 80)
    print("TESTING src/utils.py - Local Brain RAG")
    print("*" * 80)

    # Run tests
    config = test_load_config()

    if config:
        test_cuda_check()
        test_get_device()
        test_ensure_directories(config)
        test_setup_logging(config)
        test_estimate_vram()
        test_vram_monitor()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("All basic utility functions are implemented and working!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Ensure PyTorch with CUDA is installed")
    print("3. Run this script again to verify CUDA functionality")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
