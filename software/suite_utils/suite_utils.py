import re
from pathlib import Path
from typing import Optional, Dict
from omegaconf import OmegaConf
from suite_config import DEVICE_LIMITS, KNOWN_MODELS

def load_model_config(model_name: str) -> Optional[Dict]:
    """Load model config file, return None if not found."""
    # Assuming this is run from within software package or handled correctly.
    # We use Path(__file__).parent.parent to go up one level if we were in a subpackage,
    # but here utils is in software/, same as benchmark_suite.py.
    # software/suite_utils/suite_utils.py
    # conf is in software/conf
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "conf/model" / f"{model_name}.yaml"
    if not config_path.exists():
        return None
    return OmegaConf.load(config_path)

def get_model_category(model_name: str) -> str:
    """Get model category from config file, default to LLM."""
    config = load_model_config(model_name)
    return config.get("model_category", "LLM") if config else "LLM"

def extract_param_count_from_text(text: str) -> Optional[float]:
    """Extract parameter count (in billions) from text using regex."""
    match = re.search(r'(\d+(?:\.\d+)?)b', text.lower())
    return float(match.group(1)) if match else None

def get_model_parameter_count(model_name: str) -> Optional[float]:
    """Get model parameter count from name, config file, or known models."""
    # Try filename first
    if count := extract_param_count_from_text(model_name):
        return count
    
    # Try model_id in config file
    config = load_model_config(model_name)
    if config and (model_id := config.get("model_id")):
        if count := extract_param_count_from_text(model_id):
            return count
    
    # Check known models
    model_lower = model_name.lower()
    for key, params in KNOWN_MODELS.items():
        if key in model_lower:
            return params
    
    # For quantized models, check base model
    if "-quantized" in model_name:
        return get_model_parameter_count(model_name.replace("-quantized", ""))
    
    return None

def is_model_allowed_for_device(model_name: str, device_level: str) -> bool:
    """Check if model can run on given device level."""
    limit = DEVICE_LIMITS.get(device_level, float("inf"))
    if limit == float("inf"):
        return True  # Server: no limit
    
    param_count = get_model_parameter_count(model_name)
    return param_count is not None and param_count <= limit

def format_time(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
