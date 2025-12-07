class BaseModelLoader:
    """Base class for all model loaders."""
    
    def load_model(self, config):
        """Load model with config (model id, device, etc.)"""
        raise NotImplementedError

    def predict(self, prompt=None, image=None, time_series_data=None):
        """
        Run model inference.
        
        Args:
            prompt: Optional text prompt for the model
            image: Optional image input (path string or PIL Image object)
            time_series_data: Optional time series data tensor
        
        Returns:
            Model output (varies by model type)
        """
        raise NotImplementedError

    def unload_model(self):
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
