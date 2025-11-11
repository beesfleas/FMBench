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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        preference = config.get("device_preference", "auto")  # 'cpu'|'cuda'|'auto'
        has_cuda = torch.cuda.is_available()
        use_cuda = (preference == "cuda") or (preference == "auto" and has_cuda)
        if preference == "cuda" and not has_cuda:
            raise RuntimeError("CUDA requested but no CUDA device is available.")

        device_map = "auto" if use_cuda else None
        dtype = torch.float16 if use_cuda else torch.float32

        # Avoid FP8 by explicitly setting torch_dtype; leave quantization to model choice
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype
            # Optionally enable one of these if bitsandbytes is installed:
            # load_in_8bit=True,
            # load_in_4bit=True,
        )

        logging.info(f"Loaded LLM model: {model_id}")

    def predict(self, prompt, image=None, time_series_data=None):
        if image is not None:
            logging.warning("Image input provided to text-only model, ignoring image")

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        # Put tensors on the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=self.config.get("max_tokens", 64),
            pad_token_id=self.tokenizer.pad_token_id     # Explicitly set pad_token_id (idk why this is needed but it works)
        )
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        new_token_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    def unload_model(self):
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
