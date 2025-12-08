# run.py
import os
import sys
import logging

# Check for torch distributed availability and patch if missing
# This is necessary for Jetson devices with PyTorch builds that lack distributed support
# We'll check if we are on a Jetson platform first (using jtop availability or platform checks)
try:
    try:
        # Check if we are on a Jetson using jetson-stats
        from jtop import jtop
        is_jetson = True
    except ImportError:
        is_jetson = False

    if is_jetson:
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
            
            print("[WARNING] torch.distributed is not available. Mocking torch.distributed.fsdp to prevent transformers ImportError on Jetson.")
except ImportError:
    pass

    pass

# --- Dynamic Transformers Version Management ---
import subprocess
import pkg_resources

def _check_and_install_transformers():
    """
    Checks if the requested model requires a specific transformers version 
    and installs/restarts if necessary.
    """
    # Define rules: model_keyword -> required_version_spec
    # 'moment' models require transformers < 4.41 (e.g., 4.33.3 or similar old stable) due to legacy code
    # 'chronos', 'moirai', 'qwen', 'llama-vision' usually benefit from newer transformers
    
    # Defaults
    NEW_TRANSFORMERS = "transformers>=4.41.0" 
    OLD_TRANSFORMERS = "transformers==4.33.3" # Known working for Moment/Legacy
    
    # Heuristic: Check args for model name
    args_str = " ".join(sys.argv).lower()
    
    target_spec = None
    reason = None
    
    if "moment" in args_str:
        target_spec = OLD_TRANSFORMERS
        reason = "Legacy model (Moment) detected"
    elif any(x in args_str for x in ["chronos", "moirai", "qwen", "llama", "molmo"]):
        target_spec = NEW_TRANSFORMERS
        reason = "Modern model detected"
        
    if not target_spec:
        return # No specific requirement detected, keep current

    try:
        # Check current version
        current_dist = pkg_resources.get_distribution("transformers")
        current_version = current_dist.version
        
        # Check if satisfied
        # pkg_resources.Requirement.parse("transformers>=4.41.0")
        req = pkg_resources.Requirement.parse(target_spec)
        
        if current_version not in req:
            print(f"[AUTO-INSTALL] {reason}. Current transformers={current_version} does not match {target_spec}.")
            print(f"[AUTO-INSTALL] Installing {target_spec}...")
            
            # Install
            subprocess.check_call([sys.executable, "-m", "pip", "install", target_spec])
            
            print("[AUTO-INSTALL] Installation complete. Restarting process...")
            print("-" * 50)
            
            # Restart script with same args
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            # Version OK
            pass
            
    except pkg_resources.DistributionNotFound:
        # Transformers not installed? Install default new.
        print("[AUTO-INSTALL] transformers not found. Installing default...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", NEW_TRANSFORMERS])
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[AUTO-INSTALL] Warning: Failed to check/update transformers: {e}")

# Run check before importing core modules
_check_and_install_transformers()

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
