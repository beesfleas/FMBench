from transformers import AutoModel
from .base import BaseModelLoader
from .device_utils import (
    get_device_config, get_load_kwargs, move_to_device, clear_device_cache
)
import torch
import gc
import logging

log = logging.getLogger(__name__)

class HuggingFaceTimeSeriesLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.config = None
        self.device = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        
        # Device configuration
        use_cuda, use_mps, device_name = get_device_config(config)
        log.debug("Device: %s", device_name)
        
        load_kwargs = get_load_kwargs(use_cuda, use_mps, None)
        log.debug("Loading model: device_map=%s, dtype=%s",
                  load_kwargs.get("device_map"), load_kwargs.get("dtype"))
        
        try:
            self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
            self.device = move_to_device(self.model, use_mps, None)
            log.info("Loaded Time Series model: %s on %s", model_id, device_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load Time Series model {model_id}: {e}") from e

    def predict(self, prompt=None, image=None, time_series_data=None):
        if time_series_data is None:
            raise ValueError("time_series_data must be provided for Time Series model")

        device = self.device if self.device is not None else next(self.model.parameters()).device
        if isinstance(time_series_data, torch.Tensor):
            time_series_data = time_series_data.to(device)

        with torch.no_grad():
            outputs = self.model(time_series_data)
        return outputs

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.device = None
        clear_device_cache()
        gc.collect()
        log.debug("Model unloaded")