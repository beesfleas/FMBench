import logging
import sys
import os
from omegaconf import OmegaConf
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Add current directory to path so we can import components
sys.path.append(os.getcwd())

from components.models.huggingface_vlm import HuggingFaceVLMLoader

def test_vlm_loader():
    log.info("Starting VLM Loader Test")

    # Create dummy config
    config = OmegaConf.create({
        "model_id": "HuggingfaceTB/SmolVLM-256M-Instruct",
        "model_category": "VLM",
        "max_tokens": 64,
        "device": {
            "type": "auto"
        }
    })

    # Create dummy image
    log.info("Creating dummy image...")
    image = Image.new('RGB', (224, 224), color='red')
    
    try:
        # Initialize loader
        log.info("Initializing HuggingFaceVLMLoader...")
        loader = HuggingFaceVLMLoader()
        
        # Load model
        log.info(f"Loading model: {config.model_id}")
        loader.load_model(config)
        
        # Run prediction
        prompt = "Describe this image."
        log.info(f"Running prediction with prompt: '{prompt}'")
        
        result = loader.predict(prompt, image=image)
        
        log.info("Prediction successful!")
        log.info(f"Output: {result.get('output')}")
        log.info(f"TTFT: {result.get('ttft')}")
        
    except Exception as e:
        log.error(f"Test failed with error: {e}", exc_info=True)
        raise
    finally:
        if 'loader' in locals():
            log.info("Unloading model...")
            loader.unload_model()

if __name__ == "__main__":
    test_vlm_loader()
