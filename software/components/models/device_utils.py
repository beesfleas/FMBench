"""Device detection and configuration utilities for model loaders."""
import torch
import logging
from functools import wraps
from typing import Optional, Tuple, Union

log = logging.getLogger(__name__)

# MPS (Apple Silicon) tensor size limit ~1B parameters
MPS_MAX_PARAMETERS = 1_000_000_000


class MPSMemoryError(RuntimeError):
    """Raised when a model is too large for MPS or MPS runs out of memory."""
    pass


def estimate_model_size_from_hub(model_id: str) -> Optional[int]:
    """
    Estimate model parameter count from HuggingFace Hub before loading.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Estimated parameter count, or None if cannot be determined
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        
        # Try safetensors metadata first (most accurate)
        if hasattr(info, 'safetensors') and info.safetensors:
            total_params = info.safetensors.get('total', 0)
            if total_params > 0:
                log.debug("Estimated %s parameters from safetensors metadata", f"{total_params:,}")
                return total_params
        
        # Fallback: estimate from model file sizes (rough: ~2 bytes per param for fp16)
        if hasattr(info, 'siblings') and info.siblings:
            total_size = sum(
                getattr(s, 'size', 0) or 0 
                for s in info.siblings 
                if s.rfilename.endswith(('.safetensors', '.bin'))
            )
            if total_size > 0:
                # Assume fp16 (2 bytes per param) as typical
                estimated_params = total_size // 2
                log.debug("Estimated %s parameters from file size", f"{estimated_params:,}")
                return estimated_params
                
        return None
    except Exception as e:
        log.debug("Could not estimate model size from Hub: %s", e)
        return None


def check_mps_compatibility(model_id: str, allow_fallback: bool = True) -> Tuple[bool, str]:
    """
    Pre-flight check if a model is likely compatible with MPS.
    
    Args:
        model_id: HuggingFace model identifier
        allow_fallback: Whether CPU fallback is allowed
        
    Returns:
        tuple: (should_use_mps, reason)
    """
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return False, "MPS not available"
    
    estimated_params = estimate_model_size_from_hub(model_id)
    
    if estimated_params is None:
        # Can't estimate - load to CPU first to be safe
        log.warning("Cannot estimate model size for %s. Will load to CPU first.", model_id)
        return False, "size_unknown"
    
    if estimated_params > MPS_MAX_PARAMETERS:
        if allow_fallback:
            log.warning(
                "Model %s has ~%s parameters, exceeds MPS limit of %s. Using CPU.",
                model_id, f"{estimated_params:,}", f"{MPS_MAX_PARAMETERS:,}"
            )
            return False, f"too_large ({estimated_params:,} params)"
        else:
            raise MPSMemoryError(
                f"Model {model_id} has ~{estimated_params:,} parameters, "
                f"exceeding MPS limit of {MPS_MAX_PARAMETERS:,}. "
                f"Use quantization or set device_preference=cpu."
            )
    
    log.debug("Model %s with ~%s params should fit on MPS", model_id, f"{estimated_params:,}")
    return True, "compatible"


def handle_mps_errors(func):
    """
    Decorator to catch MPS-specific errors and provide actionable messages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Check for common MPS OOM patterns
            if any(pattern in error_msg for pattern in [
                'mps backend', 'mps_', 'out of memory', 
                'insufficient memory', 'metal', 'ane'
            ]):
                raise MPSMemoryError(
                    f"MPS memory error during model operation. "
                    f"Try: 1) Use quantization (model.quantization=4), "
                    f"2) Set device_preference=cpu, or "
                    f"3) Use a smaller model variant. "
                    f"Original error: {e}"
                ) from e
            raise
    return wrapper

def get_device_config(config):
    """
    Determine device configuration from config.
    
    Args:
        config: Model configuration dict
        
    Returns:
        tuple: (use_cuda, use_mps, device_name)
    """
    preference = config.get("device_preference", "auto")
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if preference == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return True, False, "CUDA"
    elif preference == "mps":
        if not has_mps:
            raise RuntimeError("MPS requested but MPS is not available (requires Apple Silicon).")
        return False, True, "MPS"
    elif preference == "cpu":
        return False, False, "CPU"
    else:  # 'auto'
        use_cuda = has_cuda
        use_mps = not has_cuda and has_mps
        device_name = "CUDA" if use_cuda else ("MPS" if use_mps else "CPU")
        return use_cuda, use_mps, device_name

def get_quantization_config(config, use_cuda):
    """
    Get quantization configuration if requested.
    
    Args:
        config: Model configuration dict
        use_cuda: Whether CUDA is being used
        
    Returns:
        BitsAndBytesConfig or None
    """
    quantize_bits = config.get("quantization", None)
    
    if quantize_bits is None:
        return None
    
    if quantize_bits not in [4, 8]:
        raise ValueError(
            f"Invalid quantization value: {quantize_bits}. "
            f"Must be 4 or 8. Set via config or command line: model.quantization=4"
        )
    
    try:
        from transformers import BitsAndBytesConfig
        if quantize_bits == 4:
            config_obj = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if use_cuda else torch.float32
            )
            log.info("Using 4-bit quantization (reduces model size by ~4x)")
        else:  # 8-bit
            config_obj = BitsAndBytesConfig(load_in_8bit=True)
            log.info("Using 8-bit quantization (reduces model size by ~2x)")
        return config_obj
    except ImportError:
        raise RuntimeError(
            "Quantization requested but bitsandbytes is not installed. "
            "Install with: pip install bitsandbytes"
        )

def get_load_kwargs(use_cuda, use_mps, quantization_config):
    """
    Get keyword arguments for model loading.
    
    Args:
        use_cuda: Whether to use CUDA
        use_mps: Whether to use MPS
        quantization_config: Quantization config or None
        
    Returns:
        dict: Load kwargs
    """
    if use_cuda:
        dtype = torch.float16
        device_map = auto
    elif use_mps:
        # Use float16 on MPS for better performance (Apple Silicon supports it well)
        dtype = torch.float16
        # Load directly to MPS device instead of CPU-first (major perf improvement)
        device_map = "mps"
    else:
        dtype = torch.float32
        device_map = None
    
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True
    }
    
    if quantization_config:
        kwargs["quantization_config"] = quantization_config
    
    # Always set device_map when specified - this is needed for:
    # 1. CUDA: device_map="auto" ensures all model parts go to GPU
    # 2. Pre-quantized models (compressed_tensors): need explicit device mapping
    # 3. VLMs: multi-part architectures need proper device distribution
    if device_map is not None:
        kwargs["device_map"] = device_map
    
    return kwargs


def get_mps_safe_load_kwargs(config, model_id: str) -> Tuple[dict, bool, bool, str]:
    """
    Get safe loading configuration with pre-flight MPS compatibility check.
    
    This is the recommended function for model loaders. It:
    1. Checks device preference from config
    2. If MPS is requested/auto-selected, validates model will fit
    3. Falls back to CPU if model is too large or size unknown
    4. Returns appropriate load kwargs
    
    Args:
        config: Model configuration dict
        model_id: HuggingFace model identifier
        
    Returns:
        tuple: (load_kwargs, use_cuda, use_mps, device_name)
    """
    use_cuda, use_mps, device_name = get_device_config(config)
    quantization_config = get_quantization_config(config, use_cuda)
    
    # Pre-flight MPS compatibility check (skip if using quantization - that handles size)
    if use_mps and not quantization_config:
        allow_fallback = config.get("allow_mps_fallback", True)
        mps_ok, reason = check_mps_compatibility(model_id, allow_fallback)
        
        if not mps_ok:
            if reason == "size_unknown":
                # Load to CPU first, will check after loading
                log.info("Loading %s to CPU first (MPS size unknown)", model_id)
                use_mps = False
                device_name = "CPU (pending MPS check)"
            else:
                # Too large or MPS not available
                use_mps = False
                device_name = "CPU"
    
    load_kwargs = get_load_kwargs(use_cuda, use_mps, quantization_config)
    
    return load_kwargs, use_cuda, use_mps, device_name, quantization_config

def check_mps_model_size(model, model_id):
    """
    Check if model is too large for MPS and raise error if so.
    
    Args:
        model: Loaded model
        model_id: Model identifier for error message
    """
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > MPS_MAX_PARAMETERS:
        raise RuntimeError(
            f"Model too large for MPS ({num_params:,} parameters). "
            f"MPS has a 2^32 byte tensor size limit that large models exceed. "
            f"Use quantization (4-bit or 8-bit) or CPU device instead."
        )

def move_to_device(model, use_mps, quantization_config):
    """
    Move model to appropriate device.
    
    Note: With device_map="mps" in get_load_kwargs(), models are now loaded
    directly to MPS. This function handles fallback cases and returns the
    current device.
    
    Args:
        model: Model to move
        use_mps: Whether MPS should be used
        quantization_config: Quantization config or None
        
    Returns:
        torch.device: Device the model is on
    """
    current_device = next(model.parameters()).device
    
    # If model is already on the correct device, just return it
    if use_mps and current_device.type == "mps":
        log.debug("Model already on MPS device")
        return current_device
    elif use_mps and not quantization_config:
        # Fallback: move to MPS if not already there
        device = torch.device("mps")
        log.debug("Moving model to MPS device")
        model.to(device)
        return device
    elif torch.cuda.is_available() and current_device.type == "cuda":
        log.debug("Model already on CUDA device")
        return current_device
    elif torch.cuda.is_available() and not quantization_config:
        device = torch.device("cuda")
        log.debug("Moving model to CUDA device")
        model.to(device)
        return device
    else:
        return current_device

def clear_device_cache():
    """Clear CUDA and MPS caches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def try_move_to_mps(model, model_id: str, config: dict) -> Tuple[bool, str]:
    """
    Attempt to move a CPU-loaded model to MPS after verifying size.
    
    Use this when pre-flight size check couldn't determine model size
    (e.g., model not on Hub, private model, etc).
    
    Args:
        model: Model loaded on CPU
        model_id: Model identifier
        config: Model configuration dict
        
    Returns:
        tuple: (success, device_name)
    """
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return False, "CPU"
    
    allow_fallback = config.get("allow_mps_fallback", True)
    num_params = sum(p.numel() for p in model.parameters())
    
    if num_params > MPS_MAX_PARAMETERS:
        if allow_fallback:
            log.warning(
                "Model %s has %s parameters, exceeds MPS limit. Staying on CPU.",
                model_id, f"{num_params:,}"
            )
            return False, "CPU"
        else:
            raise MPSMemoryError(
                f"Model {model_id} has {num_params:,} parameters, "
                f"exceeding MPS limit of {MPS_MAX_PARAMETERS:,}. "
                f"Use quantization or set device_preference=cpu."
            )
    
    # Safe to move to MPS
    try:
        log.info("Model %s (%s params) fits MPS limit. Moving to MPS.", model_id, f"{num_params:,}")
        model.to(torch.device("mps"))
        return True, "MPS"
    except RuntimeError as e:
        error_msg = str(e).lower()
        if any(p in error_msg for p in ['mps', 'metal', 'memory']):
            if allow_fallback:
                log.warning("Failed to move to MPS: %s. Staying on CPU.", e)
                return False, "CPU"
            else:
                raise MPSMemoryError(f"Failed to move model to MPS: {e}") from e
        raise
