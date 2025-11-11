from transformers import AutoModel
from .base import BaseModelLoader
import torch
import gc

class HuggingFaceTimeSeriesLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.config = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        try:
            self.model = AutoModel.from_pretrained(model_id)
            print(f"Loaded Time Series model: {model_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Time Series model {model_id}: {e}") from e

    def predict(self, prompt=None, image=None, time_series_data=None):
        if time_series_data is None:
            raise ValueError("time_series_data must be provided for Time Series model")

        with torch.no_grad():
            outputs = self.model(time_series_data)
        # Add post-processing as needed here
        return outputs

    def unload_model(self):
        model_id = self.config.get("model_id", "Unknown model")

        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()

        print(f"Unloaded model: {model_id}")