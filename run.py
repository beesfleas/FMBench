#!/usr/bin/env python3
"""
FMBench - Foundation Model Benchmarking Tool

Main entry point for running benchmarks. Handles platform-specific setup
(Jetson distributed patching, transformer version management) before
delegating to the core benchmark runner.
"""

import os
import sys
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


class TransformersVersionError(Exception):
    """Raised when the installed transformers version doesn't match requirements."""
    pass


def _parse_version_spec(spec: str) -> tuple[str, str, str]:
    """
    Parse a version specification like 'transformers>=4.41.0' or 'transformers==4.33.3'.
    
    Returns:
        Tuple of (package_name, operator, version)
    """
    import re
    match = re.match(r'(\w+)(>=|==|<=|>|<)(.+)', spec)
    if match:
        return match.groups()
    return spec, ">=", "0.0.0"


def _version_matches(current: str, operator: str, required: str) -> bool:
    """Check if current version matches the requirement."""
    from packaging.version import Version
    current_v = Version(current)
    required_v = Version(required)
    
    if operator == ">=":
        return current_v >= required_v
    elif operator == "==":
        return current_v == required_v
    elif operator == "<=":
        return current_v <= required_v
    elif operator == ">":
        return current_v > required_v
    elif operator == "<":
        return current_v < required_v
    return True


def _check_transformers_version() -> None:
    """
    Check that the correct transformers version is installed for the target model.
    
    Raises TransformersVersionError if the installed version doesn't match requirements.
    Legacy models (e.g., Moment) require older transformers versions,
    while modern models benefit from the latest version.
    """
    target_spec, reason = _detect_required_transformers_version()
    
    if target_spec is None:
        return
    
    try:
        current_version = version("transformers")
        _, operator, required_version = _parse_version_spec(target_spec)
        
        if not _version_matches(current_version, operator, required_version):
            raise TransformersVersionError(
                f"\n{'='*60}\n"
                f"TRANSFORMERS VERSION MISMATCH\n"
                f"{'='*60}\n"
                f"Reason: {reason}\n"
                f"Current version: transformers=={current_version}\n"
                f"Required version: {target_spec}\n"
                f"\n"
                f"Please install the correct version:\n"
                f"  pip install '{target_spec}'\n"
                f"{'='*60}"
            )
            
    except PackageNotFoundError:
        raise TransformersVersionError(
            f"\n{'='*60}\n"
            f"TRANSFORMERS NOT INSTALLED\n"
            f"{'='*60}\n"
            f"The 'transformers' package is required but not installed.\n"
            f"\n"
            f"Please install it:\n"
            f"  pip install '{NEW_TRANSFORMERS}'\n"
            f"{'='*60}"
        )


# =============================================================================
# Initialization
# =============================================================================

# Apply platform-specific patches
_setup_jetson_compatibility()

# Check transformers version before importing dependent modules
_check_transformers_version()

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
