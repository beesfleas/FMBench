# from transformers import AutoProcessor, AutoModelForImageTextToText (Moved to load_model)
from .base import BaseModelLoader
from .streamers import TTFTStreamer
from .device_utils import (
    get_device_config, get_load_kwargs, move_to_device, clear_device_cache,
    check_mps_model_size
)
from PIL import Image
import os
import torch
import gc
import logging

log = logging.getLogger(__name__)

class HuggingFaceVLMLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.processor = None
        self.config = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        
        log.debug("Loading processor for %s", model_id)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Device configuration
        use_cuda, use_mps, device_name = get_device_config(config)
        log.debug("Device: %s", device_name)
        
        # Check for quantization
        from .device_utils import get_quantization_config
        quantization_config = get_quantization_config(config, use_cuda)
        
        load_kwargs = get_load_kwargs(use_cuda, use_mps, quantization_config)
        
        # FlashAttention 2 support
        if config.get("use_flash_attention_2", False):
            load_kwargs["attn_implementation"] = "flash_attention_2"
            log.info("FlashAttention 2 enabled")
        
        log.debug("Loading model: device_map=%s, dtype=%s, quantization=%s",
                  load_kwargs.get("device_map"), load_kwargs.get("torch_dtype"),
                  "enabled" if quantization_config else "disabled")
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        
        # Move to device and check MPS size limits (skip if quantized)
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

        # Move to device (skip if using device_map="auto" as model is already placed)
        if load_kwargs.get("device_map") == "auto":
            # Model is already distributed across devices by accelerate
            self.device = next(self.model.parameters()).device
        else:
            self.device = move_to_device(self.model, use_mps, None)
        log.info("Loaded VLM model: %s on %s", model_id, device_name)

    def predict(self, prompt, image=None, time_series_data=None):
        if image is None:
            raise ValueError("VLM model requires an image input")

        # Process image
        image = self._process_image(image)
        
        # Try different VLM input formats
        inputs = self._get_model_inputs(prompt, image)
        
        # Generate response
        streamer = TTFTStreamer()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.get("max_tokens", 64),
            streamer=streamer
        )
        
        # Decode response
        result = self._decode_response(output_ids, inputs)
        return {"output": result, "ttft": streamer.ttft}

    def _process_image(self, image):
        """Process and validate image input"""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image)

        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image object or file path")

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to expected size for models that need specific dimensions
        # LLaVA typically expects 336x336 or uses the processor's image_size
        target_size = None
        if hasattr(self.processor, 'image_processor'):
            ip = self.processor.image_processor
            if hasattr(ip, 'size'):
                size_dict = ip.size
                if isinstance(size_dict, dict):
                    # Handle {"height": X, "width": Y} or {"shortest_edge": X}
                    if 'height' in size_dict and 'width' in size_dict:
                        target_size = (size_dict['width'], size_dict['height'])
                    elif 'shortest_edge' in size_dict:
                        target_size = (size_dict['shortest_edge'], size_dict['shortest_edge'])
                elif isinstance(size_dict, int):
                    target_size = (size_dict, size_dict)
            elif hasattr(ip, 'crop_size'):
                cs = ip.crop_size
                if isinstance(cs, dict):
                    target_size = (cs.get('width', 336), cs.get('height', 336))
                elif isinstance(cs, int):
                    target_size = (cs, cs)
        
        if target_size and image.size != target_size:
            log.debug("Resizing image from %s to %s", image.size, target_size)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
        return image

    def _get_model_inputs(self, prompt, image):
        """Try different VLM input formats and return processed inputs on correct device."""
        device = self.device or next(self.model.parameters()).device
        
        # Check if prompt already contains image placeholder
        has_image_placeholder = "<image>" in prompt.lower()
        
        # Extract text without image placeholder for conversation format
        text_only = prompt
        for marker in ["<image>", "<image>\n", "User: <image>\n", "USER: <image>\n"]:
            text_only = text_only.replace(marker, "").replace(marker.lower(), "")
        text_only = text_only.replace("Assistant:", "").replace("ASSISTANT:", "").strip()
        
        # Build method list - try multiple formats
        if has_image_placeholder:
            # Prompt has image placeholder - try direct first, then conversation format as fallback
            methods = [
                lambda: self.processor(text=prompt, images=image, return_tensors="pt"),
                lambda: self._try_conversation_format(text_only, image),  # Fallback for SmolVLM
                lambda: self._try_llava_format(text_only, image),  # Re-add LLaVA format
            ]
        else:
            # No placeholder - try wrapper formats that add one
            methods = [
                lambda: self._try_conversation_format(prompt, image),
                lambda: self._try_llava_format(prompt, image),
                lambda: self.processor(text=prompt, images=image, return_tensors="pt"),
                lambda: self.processor(text=f"<image>\n{prompt}", images=image, return_tensors="pt"),
            ]
        
        for i, method in enumerate(methods, 1):
            try:
                inputs = method()
                # Validate that we have at least some image tokens
                if "input_ids" in inputs and hasattr(self.processor, "image_token_id"):
                    num_image_tokens = (inputs["input_ids"] == self.processor.image_token_id).sum().item()
                    if num_image_tokens == 0:
                        log.debug("Input method %d produced 0 image tokens, skipping", i)
                        continue
                    log.debug("Input method %d produced %d image tokens", i, num_image_tokens)
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in inputs.items()}
            except Exception as e:
                log.debug("Input method %d failed: %s", i, e)
        
        raise RuntimeError("Could not process VLM inputs with any supported format")
    
    def _try_llava_format(self, prompt, image):
        """Try LLaVA-specific format with USER/ASSISTANT template."""
        llava_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        return self.processor(text=llava_prompt, images=image, return_tensors="pt")
    
    def _try_conversation_format(self, prompt, image):
        """Try conversation format for chat models."""
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=text, images=image, return_tensors="pt")

    def _decode_response(self, output_ids, inputs):
        """Decode response based on input format"""
        try:
            # inputs is a dict, access with bracket notation
            input_ids = inputs.get('input_ids') if isinstance(inputs, dict) else getattr(inputs, 'input_ids', None)
            if input_ids is not None and input_ids.shape[1] > 0:
                input_len = input_ids.shape[1]
                response = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
                
                # For LLaVA-style models, also try to extract answer after ASSISTANT:
                if "ASSISTANT:" in response:
                    response = response.split("ASSISTANT:")[-1].strip()
                return response
        except Exception as e:
            log.debug("Error decoding new tokens only: %s", e)
            
        # Fallback: decode entire output and try to extract answer
        full_text = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Try to extract answer after common markers
        for marker in ["ASSISTANT:", "Answer:", "answer:"]:
            if marker in full_text:
                return full_text.split(marker)[-1].strip()
        
        return full_text

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.processor = None
        self.device = None
        clear_device_cache()
        gc.collect()
        log.debug("Model unloaded")
