from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseModelLoader
from .device_utils import (
    get_device_config, get_load_kwargs, move_to_device, clear_device_cache
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
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Device configuration
        use_cuda, use_mps, device_name = get_device_config(config)
        log.debug("Device: %s", device_name)
        
        load_kwargs = get_load_kwargs(use_cuda, use_mps, None)
        log.debug("Loading model: device_map=%s, dtype=%s",
                  load_kwargs.get("device_map"), load_kwargs.get("torch_dtype"))
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        
        # Move to device
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
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.get("max_tokens", 64)
        )
        
        # Decode response
        return self._decode_response(output_ids, inputs)

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
            
        return image

    def _get_model_inputs(self, prompt, image):
        """Try different VLM input formats and return processed inputs on correct device."""
        device = self.device if self.device is not None else next(self.model.parameters()).device
        
        methods = [
            lambda: self.processor(text=prompt, images=image, return_tensors="pt"),
            lambda: self.processor(text=f"<image>\n{prompt}", images=image, return_tensors="pt"),
            lambda: self._try_conversation_format(prompt, image),
            lambda: self.processor(images=image, return_tensors="pt"),
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                inputs = method()
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in inputs.items()}
            except Exception as e:
                log.debug("Input method %d failed: %s", i, e)
        
        raise RuntimeError("Could not process VLM inputs with any supported format")
    
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
            # Try to decode only new tokens (for conversation models)
            if hasattr(inputs, 'input_ids') and inputs.input_ids.shape[1] > 0:
                return self.processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except:
            pass
            
        # Fallback: decode entire output
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    def unload_model(self):
        log.debug("Unloading model")
        self.model = None
        self.processor = None
        self.device = None
        clear_device_cache()
        gc.collect()
        log.debug("Model unloaded")
