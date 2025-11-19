from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModelLoader
import torch
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

        preference = config.get("device_preference", "auto")  # 'cpu'|'cuda'|'auto'
        has_cuda = torch.cuda.is_available()
        use_cuda = (preference == "cuda") or (preference == "auto" and has_cuda)
        log.debug("Device: %s (CUDA available: %s)", "CUDA" if use_cuda else "CPU", has_cuda)
        if preference == "cuda" and not has_cuda:
            raise RuntimeError("CUDA requested but no CUDA device is available.")

        device_map = "auto" if use_cuda else None
        dtype = torch.float16 if use_cuda else torch.float32
        log.debug("Loading model with device_map=%s, dtype=%s", device_map, dtype)

        # Avoid FP8 by explicitly setting torch_dtype; leave quantization to model choice
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype
            # Optionally enable one of these if bitsandbytes is installed:
            # load_in_8bit=True,
            # load_in_4bit=True,
        )
        log.info("Model loaded: %s", model_id)

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
        output_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id     # Explicitly set pad_token_id (idk why this is needed but it works)
        )
        log.debug("Generation completed, output shape: %s", output_ids.shape)
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        new_token_ids = output_ids[0][input_length:]
        result = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        log.debug("Decoded result (length: %d)", len(result))
        return result

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        log.debug("Model unloaded")
