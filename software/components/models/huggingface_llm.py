from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import threading
from .base import BaseModelLoader
from .streamers import TTFTStreamer
from .device_utils import (
    get_mps_safe_load_kwargs, try_move_to_mps, move_to_device, 
    clear_device_cache, handle_mps_errors, MPSMemoryError
)
import gc
import logging

log = logging.getLogger(__name__)

class HuggingFaceLLMLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None

    @handle_mps_errors
    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        
        trust_remote_code = config.get("trust_remote_code", True)
        log.debug("Loading tokenizer for %s (trust_remote_code=%s)", model_id, trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get safe device configuration with pre-flight MPS check
        load_kwargs, use_cuda, use_mps, device_name, quantization_config = get_mps_safe_load_kwargs(config, model_id)
        log.debug("Device: %s", device_name)
        
        # FlashAttention 2 support
        if config.get("use_flash_attention_2", False):
            load_kwargs["attn_implementation"] = "flash_attention_2"
            log.info("FlashAttention 2 enabled")
        
        log.debug("Loading model: device_map=%s, dtype=%s, quantization=%s",
                  load_kwargs.get("device_map"), load_kwargs.get("torch_dtype"),
                  "enabled" if quantization_config else "disabled")

        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True
            log.info("`trust_remote_code` is enabled (default behavior). This allows the model to execute custom code from the repository.")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        # If we loaded to CPU due to unknown size, try moving to MPS now
        if "pending MPS check" in device_name:
            moved, device_name = try_move_to_mps(self.model, model_id, config)
        elif use_mps and not quantization_config:
            # Ensure model is on the right device
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
        
        # Container to store results from the thread
        thread_results = {}

        def generate_wrapper():
            start_time = time.time()
            try:
                output = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=streamer
                )
                end_time = time.time()
                thread_results['output_ids'] = output
                thread_results['latency'] = end_time - start_time
            except Exception as e:
                thread_results['error'] = e

        # Run generation in a separate thread
        gen_thread = threading.Thread(target=generate_wrapper)
        gen_thread.start()
        gen_thread.join()

        if 'error' in thread_results:
            raise thread_results['error']
            
        output_ids = thread_results['output_ids']
        latency = thread_results['latency']
        
        log.debug("Generation completed, output shape: %s", output_ids.shape)
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        new_token_ids = output_ids[0][input_length:]
        result = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        log.debug("Decoded result (length: %d)", len(result))
        return {"output": result, "ttft": streamer.ttft, "latency": latency, "num_tokens": len(new_token_ids)}

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.tokenizer = None
        clear_device_cache()
        gc.collect()
        log.debug("Model unloaded")

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity using a sliding window approach for long text."""
        log.debug("Computing perplexity for text length: %d", len(text))
        
        encodings = self.tokenizer(text, return_tensors="pt")
        
        # Put tensors on the same device as the model
        device = next(self.model.parameters()).device
        inputs = encodings.input_ids.to(device)
        
        max_length = self.config.get("max_length", self.model.config.max_position_embeddings)
        stride = self.config.get("stride", 512)
        seq_len = inputs.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            
            input_ids = inputs[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        log.debug("Computed perplexity: %s", ppl.item())
        return ppl.item()
