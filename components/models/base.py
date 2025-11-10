# components/models/base.py
class BaseModelLoader:
    def load_model(self, config):
        """Load model with config (model id, device, etc.)"""
        raise NotImplementedError

    def predict(self, prompt, image=None):
        """Run model inference for the given input prompt and optional image
        
        Args:
            prompt: Text prompt for the model
            image: Optional image input (path string or PIL Image object)
        
        Returns:
            Generated text response
        """
        raise NotImplementedError

    def unload_model(self):
        """
        Releases all model-related resources from VRAM and RAM.
        """
        pass
