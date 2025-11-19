from .huggingface_llm import HuggingFaceLLMLoader
from .huggingface_vlm import HuggingFaceVLMLoader
from .huggingface_timeseries import HuggingFaceTimeSeriesLoader
from .base import BaseModelLoader

# Model category to loader class mapping
_MODEL_LOADERS = {
    "LLM": HuggingFaceLLMLoader,
    "VLM": HuggingFaceVLMLoader,
    "TIME_SERIES": HuggingFaceTimeSeriesLoader,
}

def get_model_loader(model_config) -> BaseModelLoader:
    """
    Factory function to instantiate the correct model loader based on configuration.
    
    Args:
        model_config (dict): Configuration dictionary containing at least 
                             "model_category".
    
    Returns:
        BaseModelLoader: An instantiated concrete model loader object.
    
    Raises:
        ValueError: If an unknown model category is specified.
    """
    model_category = model_config.get("model_category", "LLM")
    
    loader_class = _MODEL_LOADERS.get(model_category)
    if loader_class is None:
        raise ValueError(
            f"Unknown model category: {model_category}. "
            f"Available categories: {list(_MODEL_LOADERS.keys())}"
        )
    
    return loader_class()
