
import logging
import os
import sys
import yaml
import warnings
from typing import Dict, Any, List

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("verify_vlm_scenarios")

from components.models.huggingface_vlm import HuggingFaceVLMLoader
# Import scenario classes dynamically or statically
from components.scenarios.common_vlm_scenarios import (
    HaGRIDScenario, 
    GTSRBScenario, 
    OCRScenario, 
    CountBenchQAScenario, 
    VQAScenario
)
from transformers import AutoConfig

MODELS_TO_TEST = ["llava"]
SCENARIOS_TO_TEST = ["docvqa", "countbenchqa", "vqa"]

# Map scenario names to their config files and classes if needed, 
# but usually we can load from config.
SCENARIO_CONFIG_MAP = {
    "hagrid": "hagrid.yaml",
    "gtsrb": "gtsrb.yaml",
    "docvqa": "docvqa.yaml",
    "countbenchqa": "countbenchqa.yaml",
    "vqa": "vqa.yaml"
}

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_verification():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    results = {}

    for model_name in MODELS_TO_TEST:
        model_config_path = os.path.join(base_dir, "conf", "model", f"{model_name}.yaml")
        if not os.path.exists(model_config_path):
            logger.error(f"Model config not found: {model_config_path}")
            continue
            
        logger.info(f"=== Testing Model: {model_name} ===")
        model_config = load_config(model_config_path)
        
        # Load Model
        try:
            loader = HuggingFaceVLMLoader()
            # For verification, we might want to force CPU or check for GPU availability
            # But the loader handles it.
            loader.load_model(model_config)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            results[model_name] = "LOAD_FAILED"
            continue

        for scenario_name in SCENARIOS_TO_TEST:
            scenario_key = f"{model_name}_{scenario_name}"
            logger.info(f"--- Scenario: {scenario_name} ---")
            
            scenario_config_path = os.path.join(base_dir, "conf", "scenario", SCENARIO_CONFIG_MAP[scenario_name])
            if not os.path.exists(scenario_config_path):
                logger.error(f"Scenario config not found: {scenario_config_path}")
                results[scenario_key] = "CONFIG_MISSING"
                continue

            try:
                scenario_config = load_config(scenario_config_path)
                # Lower num_samples for quick verification
                scenario_config['num_samples'] = 2 
                scenario_config['split'] = 'test' if 'test' in scenario_config.get('split', '') else scenario_config.get('split', 'train') # Try to use test split if possible or default
                 # Fallback for some datasets that might mainly have train
                
                # Instantiate Scenario
                # We need to dynamically instantiate the class based on _target_ or manual mapping
                target_class = scenario_config.get("_target_")
                if "HaGRIDScenario" in target_class:
                    scenario = HaGRIDScenario(scenario_config)
                elif "GTSRBScenario" in target_class:
                    scenario = GTSRBScenario(scenario_config)
                elif "OCRScenario" in target_class:
                    scenario = OCRScenario(scenario_config)
                elif "CountBenchQAScenario" in target_class:
                    scenario = CountBenchQAScenario(scenario_config)
                elif "VQAScenario" in target_class:
                    scenario = VQAScenario(scenario_config)
                else:
                    logger.warning(f"Unknown scenario target: {target_class}")
                    continue

                # Load Dataset
                logger.info(f"Loading dataset for {scenario_name}...")
                data = scenario.get_tasks()
                logger.info(f"Loaded {len(data)} samples.")

                # Run Inference on a few samples
                success_count = 0
                for i, task in enumerate(data[:2]): # Run just 2 samples
                    try:
                        logger.info(f"Running sample {i+1}...")
                        prompt = task['prompt']
                        image = task['image']
                        
                        output = loader.predict(prompt, image=image)
                        prediction = output['output']
                        logger.info(f"Prediction: {prediction}")
                        
                        # Compute Metrics (Basic check)
                        metrics = scenario.compute_metrics(prediction, task['target'], task)
                        logger.info(f"Metrics: {metrics}")
                        
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing sample {i+1}: {e}")
                
                if success_count > 0:
                    results[scenario_key] = "PASSED"
                else:
                    results[scenario_key] = "FAILED"

            except Exception as e:
                logger.error(f"Failed scenario {scenario_name} with {model_name}: {e}")
                results[scenario_key] = f"ERROR: {str(e)}"
        
        # Unload model to free memory
        try:
            loader.unload_model()
        except Exception as e:
            logger.error(f"Error unloading model: {e}")

    logger.info("=== Verification Results ===")
    for key, status in results.items():
        logger.info(f"{key}: {status}")

if __name__ == "__main__":
    run_verification()
