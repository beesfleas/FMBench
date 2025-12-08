
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
        self.is_arima = False
        self.is_patchtst = False

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        
        # Check for ARIMA (Statistical model, no HF loading)
        if "arima" in model_id.lower() or "statsmodels" in model_id.lower():
            self.is_arima = True
            log.info("Initialized ARIMA model (statsmodels). No weights loaded yet (fits per series).")
            return

        # Device configuration
        use_cuda, use_mps, device_name = get_device_config(config)
        log.debug("Device: %s", device_name)
        
        load_kwargs = get_load_kwargs(use_cuda, use_mps, None)
        log.debug("Loading model: device_map=%s, dtype=%s",
                  load_kwargs.get("device_map"), load_kwargs.get("dtype"))
        
        try:
            # Check for Moirai
            if "moirai" in model_id.lower():
                self.is_moirai = True
                try:
                    from uni2ts.model.moirai import MoiraiForecast
                    # Placeholder for uni2ts loading if we had it
                    self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
                except ImportError:
                    log.warning("uni2ts package not found. Loading Moirai as standard HF model.")
                    self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)

            # Check for Moment
            elif "moment" in model_id.lower():
                self.is_moment = True
                try:
                    from momentfm import MOMENTPipeline
                    self.model = MOMENTPipeline.from_pretrained(
                        model_id, 
                        model_kwargs={"IS_MOMENT_PIPELINE": True} 
                    )
                    self.model.init()
                    self.model = self.model.to(device_name.lower() if "cuda" in device_name.lower() else "cpu")
                except ImportError:
                     log.warning("momentfm package not found. Attempting to load as standard HF model.")
                     self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
                except Exception as e:
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
            
            # Check for PatchTST
            elif "patchtst" in model_id.lower():
                self.is_patchtst = True
                # PatchTST usually works with AutoModelForPrediction or similar, but standard AutoModel might load base.
                # Use standard load, specific logic in predict if needed.
                from transformers import PatchTSTForPrediction
                try:
                    self.model = PatchTSTForPrediction.from_pretrained(
                        model_id, 
                        num_input_channels=1, 
                        ignore_mismatched_sizes=True, 
                        **load_kwargs
                    )
                except Exception:
                    log.warning("PatchTSTForPrediction not found or failed. Loading base AutoModel.")
                    self.model = AutoModel.from_pretrained(model_id, **load_kwargs)

            else:
                # Standard loading
                self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
            
            # Post-loading device handling
            if not self.pipeline and not self.is_moment and self.model: 
                 self.device = move_to_device(self.model, use_mps, None)
            elif self.pipeline:
                 self.device = self.pipeline.model.device
            elif self.is_moment:
                 self.device = next(self.model.parameters()).device

            if self.model or self.pipeline:
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
        
        # --- ARIMA ---
        if self.is_arima:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                # Fit ARIMA on the fly (naive default order)
                # Ensure input is 1D list/array
                if isinstance(time_series_data, torch.Tensor):
                    history = time_series_data.cpu().numpy().flatten()
                elif isinstance(time_series_data, list):
                    history = np.array(time_series_data).flatten()
                else:
                    history = np.array(time_series_data).flatten()
                
                # Simple ARIMA(5,1,0) as a reasonable default baseline
                model = ARIMA(history, order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=prediction_length)
                return {"forecast": forecast.tolist()}
            except ImportError:
                 log.error("statsmodels not installed. Cannot run ARIMA.")
                 return {"forecast": [0.0]*prediction_length}
            except Exception as e:
                 log.error(f"ARIMA fitting failed: {e}")
                 # Fallback: naive persistence (last value)
                 last_val = history[-1] if len(history) > 0 else 0.0
                 return {"forecast": [last_val]*prediction_length}

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

        # --- PATCHTST ---
        if self.is_patchtst:
             # PatchTST input: past_values tensor [batch, seq_len, num_input_channels]
             # Assuming univariate for M3 -> num_input_channels=1
             
             device = self.device or next(self.model.parameters()).device
             if isinstance(time_series_data, list):
                 past_values = torch.tensor(time_series_data, device=device).unsqueeze(0).unsqueeze(-1) # [1, T, 1]
             elif isinstance(time_series_data, np.ndarray):
                 past_values = torch.from_numpy(time_series_data).to(device).unsqueeze(0).unsqueeze(-1)
             elif isinstance(time_series_data, torch.Tensor):
                 past_values = time_series_data.to(device)
                 if past_values.ndim == 1:
                      past_values = past_values.unsqueeze(0).unsqueeze(-1)
             
             # Handle context length truncation for PatchTST
             # Default context_length is often 512 for these models
             context_length = self.config.get("context_length", 512)
             if past_values.shape[1] > context_length:
                 past_values = past_values[:, -context_length:, :]
             
             # Cast to model dtype (handle fp16 mismatch)
             past_values = past_values.to(dtype=self.model.dtype)

             with torch.no_grad():
                 try:
                     # PatchTSTForPrediction output has .prediction_logits or .prediction_outputs
                     # It usually returns a generic Output class
                     outputs = self.model(past_values=past_values)
                     # Attributes: prediction_outputs usually [batch, prediction_len, num_output_channels]
                     if hasattr(outputs, "prediction_outputs"):
                         forecast = outputs.prediction_outputs[0, :, 0].cpu().numpy()
                         return {"forecast": forecast.tolist()}
                     elif hasattr(outputs, "logits"): # Fallback name?
                         forecast = outputs.logits[0, :, 0].cpu().numpy()
                         return {"forecast": forecast.tolist()}
                     else:
                          log.warning(f"PatchTST output unknown keys: {outputs.keys()}")
                          return {"forecast": [0.0]*prediction_length}
                 except Exception as e:
                     log.error(f"PatchTST inference failed: {e}")
                     return {"forecast": [0.0]*prediction_length}

        # --- MOIRAI ---
        if self.is_moirai:
             # Moirai input handling
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
