from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from .base import BaseModelLoader
from .streamers import TTFTStreamer
from .device_utils import (
    get_device_config, get_quantization_config, get_load_kwargs,
    check_mps_model_size, move_to_device, clear_device_cache
)
import gc
import logging

log = logging.getLogger(__name__)

class HuggingFaceLLMLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        
        log.debug("Loading tokenizer for %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Device and quantization configuration
        use_cuda, use_mps, device_name = get_device_config(config)
        log.debug("Device: %s", device_name)
        
        quantization_config = get_quantization_config(config, use_cuda)
        load_kwargs = get_load_kwargs(use_cuda, use_mps, quantization_config)
        
        log.debug("Loading model: device_map=%s, dtype=%s, quantization=%s",
                  load_kwargs.get("device_map"), load_kwargs.get("dtype"),
                  "enabled" if quantization_config else "disabled")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        # Move to device and check MPS size limits
        if use_mps and not quantization_config:
            try:
                check_mps_model_size(self.model, model_id)
            except RuntimeError as e:
                if config.get("allow_mps_fallback", True):
                    log.warning("Model too large for MPS: %s. Falling back to CPU.", e)
                    use_mps = False
                    device_name = "CPU"
                else:
                    raise e
        
        move_to_device(self.model, use_mps, quantization_config)
        log.info("Model loaded: %s on %s", model_id, device_name)

    def predict(self, prompt, image=None, time_series_data=None):
        if image is not None:
            log.warning("Image input provided to text-only model, ignoring")

        log.debug("Tokenizing prompt (length: %d)", len(prompt))
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        log.debug("Input shape: %s", inputs['input_ids'].shape)

        # Put tensors on the same device as the model
        device = next(self.model.parameters()).device
        log.debug("Model device: %s", device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_tokens = self.config.get("max_tokens", 64)
        log.debug("Generating (max_new_tokens: %d)", max_tokens)
        
        streamer = TTFTStreamer()
        start_time = time.time()
        output_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer
        )
        end_time = time.time()
        latency = end_time - start_time
        
        log.debug("Generation completed, output shape: %s", output_ids.shape)
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        new_token_ids = output_ids[0][input_length:]
        result = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        log.debug("Decoded result (length: %d)", len(result))
        return {"output": result, "ttft": streamer.ttft, "latency": latency}

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.tokenizer = None
        clear_device_cache()
        gc.collect()
        log.debug("Model unloaded")
