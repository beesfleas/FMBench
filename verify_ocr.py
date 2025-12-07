from omegaconf import OmegaConf
from core.runner import run_benchmark
import logging
import sys
import os

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Define configuration
cfg = OmegaConf.create({
    "model": {
        "model_id": "HuggingfaceTB/SmolVLM-256M-Instruct",
        # "model_id": "HuggingFaceM4/idefics2-8b", # Alternative if smolvlm fails
        "model_category": "VLM",
        "max_tokens": 512,
        "device_preference": "auto",
        "allow_mps_fallback": True
    },
    "scenario": {
        "_target_": "components.scenarios.common_vlm_scenarios.OCRScenario",
        "dataset_name": "lmms-lab/DocVQA",
        "dataset_split": "validation",
        "image_key": "image",
        "question_key": "question",
        "target_key": "answers",
        "prompt_template": "Question: {question}\nAnswer:",
        "num_samples": 5  # Limit samples for quick testing
    },
    "device": {
        "type": "auto"
    },
    "sampling_interval": 1,
    "log_level": "INFO"
})

print("Running OCR Scenario Verification with SmolVLM...")
run_benchmark(cfg)
