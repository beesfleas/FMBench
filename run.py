#!/usr/bin/env python3
"""
FMBench - Foundation Model Benchmarking Tool

Main entry point for running benchmarks. Handles platform-specific setup
(Jetson distributed patching, transformer version management) before
delegating to the core benchmark runner.
"""

import os
import sys
import subprocess
from importlib.metadata import version, PackageNotFoundError

# =============================================================================
# Constants
# =============================================================================

NEW_TRANSFORMERS = "transformers>=4.41.0"
OLD_TRANSFORMERS = "transformers==4.33.3"  # Known working for Moment/Legacy

MODERN_MODEL_KEYWORDS = ["chronos", "moirai", "qwen", "llama", "molmo"]
LEGACY_MODEL_KEYWORDS = ["moment"]


# =============================================================================
# Platform-Specific Setup
# =============================================================================

def _is_jetson_platform() -> bool:
    """Check if running on an NVIDIA Jetson device."""
    try:
        from jtop import jtop
        return True
    except ImportError:
        return False


def _patch_torch_distributed_for_jetson() -> None:
    """
    Mock torch.distributed.fsdp for Jetson devices lacking distributed support.
    
    Some Jetson PyTorch builds don't include torch.distributed, which causes
    ImportErrors when loading transformers. This patches the missing modules.
    """
    import torch
    
    if torch.distributed.is_available():
        return
    
    import types
    
    # Mock torch.distributed.fsdp module
    fsdp_mock = types.ModuleType("torch.distributed.fsdp")
    sys.modules["torch.distributed.fsdp"] = fsdp_mock
    torch.distributed.fsdp = fsdp_mock
    
    # Mock FullyShardedDataParallel class (checked by transformers)
    class MockFSDP:
        pass
    fsdp_mock.FullyShardedDataParallel = MockFSDP
    
    # Mock distributed functions to avoid AttributeErrors
    if not hasattr(torch.distributed, 'get_world_size'):
        torch.distributed.get_world_size = lambda: 1
    if not hasattr(torch.distributed, 'is_initialized'):
        torch.distributed.is_initialized = lambda: False
    
    print("[WARNING] torch.distributed not available. "
          "Mocked torch.distributed.fsdp for Jetson compatibility.")


def _setup_jetson_compatibility() -> None:
    """Apply Jetson-specific patches if running on Jetson hardware."""
    try:
        if _is_jetson_platform():
            _patch_torch_distributed_for_jetson()
    except ImportError:
        pass


# =============================================================================
# Transformers Version Management
# =============================================================================

def _detect_required_transformers_version() -> tuple[str | None, str | None]:
    """
    Detect required transformers version based on model name in CLI args.
    
    Returns:
        Tuple of (version_spec, reason) or (None, None) if no specific version needed.
    """
    args_str = " ".join(sys.argv).lower()
    
    if any(keyword in args_str for keyword in LEGACY_MODEL_KEYWORDS):
        return OLD_TRANSFORMERS, "Legacy model (Moment) detected"
    
    if any(keyword in args_str for keyword in MODERN_MODEL_KEYWORDS):
        return NEW_TRANSFORMERS, "Modern model detected"
    
    return None, None


def _install_and_restart(version_spec: str) -> None:
    """Install a specific transformers version and restart the process."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", version_spec])
    print("[AUTO-INSTALL] Installation complete. Restarting process...")
    print("-" * 50)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _ensure_correct_transformers_version() -> None:
    """
    Ensure the correct transformers version is installed for the target model.
    
    Legacy models (e.g., Moment) require older transformers versions,
    while modern models benefit from the latest version.
    """
    target_spec, reason = _detect_required_transformers_version()
    
    if target_spec is None:
        return
    
    try:
        current_dist = pkg_resources.get_distribution("transformers")
        current_version = current_dist.version
        req = pkg_resources.Requirement.parse(target_spec)
        
        if current_version not in req:
            print(f"[AUTO-INSTALL] {reason}. "
                  f"Current transformers={current_version} does not match {target_spec}.")
            print(f"[AUTO-INSTALL] Installing {target_spec}...")
            _install_and_restart(target_spec)
            
    except pkg_resources.DistributionNotFound:
        print("[AUTO-INSTALL] transformers not found. Installing default...")
        _install_and_restart(NEW_TRANSFORMERS)
    except Exception as e:
        print(f"[AUTO-INSTALL] Warning: Failed to check/update transformers: {e}")


# =============================================================================
# Initialization
# =============================================================================

# Apply platform-specific patches
_setup_jetson_compatibility()

# Ensure correct transformers version before importing dependent modules
_ensure_correct_transformers_version()

# Prevent tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# =============================================================================
# Main Entry Point
# =============================================================================

import hydra
from omegaconf import DictConfig
from core.runner import run_benchmark
from core.logging_setup import setup_logging


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the benchmark with the provided Hydra configuration."""
    setup_logging(cfg)
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
