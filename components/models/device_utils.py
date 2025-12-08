"""Device detection and configuration utilities for model loaders."""
import torch
import logging

log = logging.getLogger(__name__)

# MPS (Apple Silicon) tensor size limit ~1B parameters
MPS_MAX_PARAMETERS = 1_000_000_000

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
        device_map = "auto"  # Required for VLMs with multi-part architectures (vision encoder + LLM)
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
    elif device_map is not None:
        kwargs["device_map"] = device_map
    
    return kwargs

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

