from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from PIL import Image


class BaseModelLoader(ABC):
    """Base class for all model loaders."""
    
    @abstractmethod
    def load_model(self, config: Dict[str, Any]) -> None:
        """
        Load model with config.
        
        Args:
            config: Configuration dictionary containing model_id, device settings, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, 
        prompt: Optional[str] = None, 
        image: Optional[Union[str, Image.Image]] = None, 
        time_series_data: Optional[Any] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Run model inference.
        
        Args:
            prompt: Optional text prompt for the model
            image: Optional image input (path string or PIL Image object)
            time_series_data: Optional time series data tensor
        
        Returns:
            Model output - either a string or dict with 'output' key and metrics
        """
        raise NotImplementedError

    def unload_model(self) -> None:
        """Release all model-related resources from VRAM and RAM."""
        pass

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity for a given text.
        
        Args:
            text: The input text to compute perplexity for.
            
        Returns:
            Perplexity score as a float.
        """
        raise NotImplementedError
