# components/models/huggingface_vlm.py
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseModelLoader
from PIL import Image
import os
import torch
import gc

class HuggingFaceVLMLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.processor = None
        self.config = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id)
        print(f"Loaded VLM model: {model_id}")

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
        """Try different VLM input formats"""
        # Method 1: Direct text and image (most common)
        try:
            return self.processor(text=prompt, images=image, return_tensors="pt")
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
        # Method 2: With image placeholder
        try:
            return self.processor(text=f"<image>\n{prompt}", images=image, return_tensors="pt")
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
        # Method 3: Conversation format (for chat models)
        try:
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            return self.processor(text=text, images=image, return_tensors="pt")
        except Exception as e3:
            print(f"Method 3 failed: {e3}")
            
        # Method 4: Just image (for captioning models)
        try:
            return self.processor(images=image, return_tensors="pt")
        except Exception as e4:
            print(f"Method 4 failed: {e4}")
            
        # If all methods fail, raise error
        raise RuntimeError("Could not process VLM inputs with any supported format")

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
        model_id = self.config.get("model_id", "Unknown model")

        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()

        print(f"Unloaded model: {model_id}")