# components/models/base.py
class BaseModelLoader:
    def load_model(self, config):
        """Load model with config (model id, device, etc.)"""
        raise NotImplementedError

    def predict(self, prompt):
        """Run model inference for the given input prompt"""
        raise NotImplementedError
