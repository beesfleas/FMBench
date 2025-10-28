from .huggingface_llm import HuggingFaceLLMLoader
from .huggingface_vlm import HuggingFaceVLMLoader
from .huggingface_timeseries import HuggingFaceTimeSeriesLoader
from .base import BaseModelLoader

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
    
    if model_category == "VLM":
        return HuggingFaceVLMLoader()
    elif model_category == "TIME_SERIES":
        return HuggingFaceTimeSeriesLoader()
    elif model_category == "LLM":
        return HuggingFaceLLMLoader()
    else:
        raise ValueError(f"Unknown model category specified: {model_category}")
