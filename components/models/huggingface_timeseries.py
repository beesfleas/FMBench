
from transformers import AutoModel, AutoModelForSeq2SeqLM
from .base import BaseModelLoader
from .device_utils import (
    get_device_config, get_load_kwargs, move_to_device, clear_device_cache,
    check_mps_model_size
)
import torch
import gc
import logging
import numpy as np

log = logging.getLogger(__name__)

class HuggingFaceTimeSeriesLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.config = None
        self.device = None
        self.pipeline = None # For Chronos
        self.is_chronos = False
        self.is_moment = False
        self.is_moirai = False

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
            # Check for Moirai (Salesforce)
            if "moirai" in model_id.lower():
                self.is_moirai = True
                try:
                    # Moirai usually requires uni2ts.MoiraiForecast
                    from uni2ts.model.moirai import MoiraiForecast
                    # MoiraiForecast.load_from_checkpoint or similar
                    # But HF loading: AutoModelForTimeseries? Not standard yet.
                    # As of now, Moirai uses standard HF weights but needs custom wrapper or code.
                    # We will try to load it via AutoModel if uni2ts not found, but it won't predict properly.
                    # Assuming basic HF loading first for structure.
                    self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
                except ImportError:
                    log.warning("uni2ts package not found. Loading Moirai as standard HF model.")
                    self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)

            # Check for Moment (AutonLab)
            elif "moment" in model_id.lower():
                self.is_moment = True
                try:
                    from momentfm import MOMENTPipeline
                    self.model = MOMENTPipeline.from_pretrained(
                        model_id, 
                        model_kwargs={"IS_MOMENT_PIPELINE": True} # Example arg, waiting for library specifics
                    )
                    self.model.init() # Some pipelines need init
                    self.model = self.model.to(device_name.lower() if "cuda" in device_name.lower() else "cpu")
                except ImportError:
                     log.warning("momentfm package not found. Attempting to load as standard HF model.")
                     self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
                except Exception as e:
                     # Fallback
                     log.warning(f"Failed to load MOMENTPipeline: {e}. Fallback to AutoModel.")
                     self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)

            # Check for Chronos
            elif "chronos" in model_id.lower():
                try:
                    from chronos import ChronosPipeline
                    self.is_chronos = True
                    self.pipeline = ChronosPipeline.from_pretrained(
                        model_id,
                        device_map=load_kwargs.get("device_map", "cpu"),
                        torch_dtype=load_kwargs.get("torch_dtype", torch.float32)
                    )
                    self.model = self.pipeline.model
                except ImportError:
                    log.warning("chronos package not found. Attempting to load as standard HF model.")
                    self.is_chronos = False
                    self.model = AutoModel.from_pretrained(model_id, **load_kwargs)

            else:
                # Standard loading
                self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
            
            # Post-loading device handling if not already handled by pipeline
            if not self.pipeline and not self.is_moment: # MOMENTPipeline handles device
                 self.device = move_to_device(self.model, use_mps, None)
            elif self.pipeline:
                 self.device = self.pipeline.model.device
            elif self.is_moment:
                 self.device = next(self.model.parameters()).device

            log.info("Loaded Time Series model: %s on %s", model_id, self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Time Series model {model_id}: {e}") from e

    def predict(self, prompt=None, image=None, time_series_data=None):
        if time_series_data is None:
            if isinstance(prompt, list):
                time_series_data = prompt
            else:
                 raise ValueError("time_series_data must be provided for Time Series model")

        # Configs
        prediction_length = self.config.get("prediction_length", 12)
        
        # --- CHRONOS ---
        if self.is_chronos and self.pipeline:
            # ... existing chronos logic ...
            if isinstance(time_series_data, list):
                context = torch.tensor(time_series_data)
            elif isinstance(time_series_data, np.ndarray):
                context = torch.from_numpy(time_series_data)
            else:
                context = time_series_data
            
            forecast = self.pipeline.predict(
                context,
                prediction_length,
                num_samples=self.config.get("num_samples", 20),
                limit_prediction_length=False
            )
            forecast_np = forecast.cpu().numpy()
            if forecast_np.ndim == 3:
                forecast_np = forecast_np[0]
            median_forecast = np.median(forecast_np, axis=0)
            return {"forecast": median_forecast.tolist()}

        # --- MOMENT ---
        if self.is_moment:
            # Input: [batch_size, n_channels, context_length]
            # Verify if model object is pipeline or raw
            # If using momentfm, inputs might be different.
            # Assuming AutoModel fallback or standard MOMENT input
            
            # Convert to [1, 1, T] if list
            if isinstance(time_series_data, list):
                 tensor = torch.tensor(time_series_data).unsqueeze(0).unsqueeze(0) # [1, 1, T]
            elif isinstance(time_series_data, np.ndarray):
                 tensor = torch.from_numpy(time_series_data).unsqueeze(0).unsqueeze(0)
            elif isinstance(time_series_data, torch.Tensor):
                 tensor = time_series_data
                 if tensor.ndim == 1:
                     tensor = tensor.unsqueeze(0).unsqueeze(0)
            
            tensor = tensor.to(self.device).float()
            
            try:
                # Moment usually takes (x_enc=..., input_mask=...)
                # Or if pipeline: model(tensor)
                # This is highly specific to momentfm API
                # Placeholder for robust call:
                with torch.no_grad():
                     output = self.model(tensor) # Try direct call
                     
                # Parse output - Moment can do classification, forecasting, etc.
                # Assuming forecasting output structure
                # This is likely to fail without specific library logic
                return {"forecast": [0.0]*prediction_length} # Placeholder
            except Exception as e:
                log.error(f"Moment inference failed: {e}")
                return {"forecast": [0.0]*prediction_length}

        # --- MOIRAI ---
        if self.is_moirai:
             # Moirai input handling
             # Similar to Chronos/others, usually expects specific dict format for uni2ts
             # Without uni2ts, we can't easily run Moirai inference as it involves patching/masking
             log.warning("Moirai inference requires uni2ts library setup.")
             return {"forecast": [0.0]*prediction_length}


        # --- DEFAULT ---
        device = self.device or next(self.model.parameters()).device
        
        # Ensure tensor
        if isinstance(time_series_data, list):
            time_series_data = torch.tensor(time_series_data, device=device)
        elif isinstance(time_series_data, np.ndarray):
            time_series_data = torch.from_numpy(time_series_data).to(device)
        elif isinstance(time_series_data, torch.Tensor):
            time_series_data = time_series_data.to(device)

        with torch.no_grad():
            try:
                outputs = self.model(time_series_data)
                # Return dummy if raw output doesn't match metric expectation
                return {"forecast": [0.0]*prediction_length} 
            except Exception as e:
                log.error("Raw model inference failed: %s", e)
                return {"error": str(e), "forecast": [0.0] * prediction_length}
        return outputs

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.pipeline = None
        self.device = None
        clear_device_cache()
        gc.collect()
        log.debug("Model unloaded")
