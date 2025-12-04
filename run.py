# run.py
import os
import sys
import logging

# Check for torch distributed availability and patch if missing
# This is necessary for Jetson devices with PyTorch builds that lack distributed support
try:
    import torch
    if not torch.distributed.is_available():
        # Create a dummy module to satisfy imports
        from unittest.mock import MagicMock
        import types
        
        # Mock torch.distributed.fsdp
        fsdp_mock = types.ModuleType("torch.distributed.fsdp")
        sys.modules["torch.distributed.fsdp"] = fsdp_mock
        torch.distributed.fsdp = fsdp_mock
        
        # Mock FullyShardedDataParallel class which is checked by transformers
        class MockFSDP:
            pass
        fsdp_mock.FullyShardedDataParallel = MockFSDP
        
        # Ensure other distributed functions exist to avoid AttributeErrors
        if not hasattr(torch.distributed, 'get_world_size'):
            torch.distributed.get_world_size = lambda: 1
        if not hasattr(torch.distributed, 'is_initialized'):
            torch.distributed.is_initialized = lambda: False
        
        print("[WARNING] torch.distributed is not available. Mocking torch.distributed.fsdp to prevent transformers ImportError.")
except ImportError:
    pass

import hydra
from omegaconf import DictConfig
from core.runner import run_benchmark
from core.logging_setup import setup_logging

# Set tokenizers parallelism to false to avoid fork warnings when subprocesses are created
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg) 
    run_benchmark(cfg)

if __name__ == "__main__":
    main()
