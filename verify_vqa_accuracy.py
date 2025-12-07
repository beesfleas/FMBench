import logging
import torch
import warnings
from components.scenarios.common_vlm_scenarios import VQAScenario
from components.models.huggingface_vlm import HuggingFaceVLMLoader

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_accuracy():
    logger.info("Initializing VQA Verification...")
    
    # 1. Load Scenario
    scenario_config = {
        "dataset_name": "HuggingFaceM4/the_cauldron",
        "dataset_config": "vqav2",
        "split": "train",
        "num_samples": 5,
        "image_key": "images",
        "question_key": "texts", 
        "target_key": "texts",
        "prompt_template": "Question: {question} Answer:",
        "trust_remote_code": True,
        "streaming": True 
    }
    
    logger.info(f"Loading Scenario: {scenario_config['dataset_name']}")
    scenario = VQAScenario(scenario_config)
    tasks = scenario.get_tasks()
    logger.info(f"Loaded {len(tasks)} tasks")

    # 2. Load Model
    model_config = {
        "model_id": "HuggingfaceTB/SmolVLM-256M-Instruct",
        "device": {"type": "cuda" if torch.cuda.is_available() else "cpu"},
        "max_tokens": 20
    }
    
    logger.info(f"Loading Model: {model_config['model_id']}")
    loader = HuggingFaceVLMLoader()
    loader.load_model(model_config)
    
    # 3. Predict and Compare
    logger.info("\n--- Starting Verification Loop ---")
    correct = 0
    for i, task in enumerate(tasks):
        logger.info(f"\nTask {i+1}:")
        
        # Get inputs
        prompt = task.get("prompt") or task.get("input")
        image = task.get("image")
        target = task.get("target")

        # Run Prediction
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Target: {target}")
        
        try:
            result = loader.predict(prompt, image=image)
            output = result["output"] if isinstance(result, dict) else result
            
            logger.info(f"Model Output: {output}")
            
            # Evaluate using Scenario Logic
            metrics = scenario.compute_metrics(output, target, task)
            accuracy = metrics.get("accuracy", 0.0)
            
            if accuracy > 0:
                logger.info("Result: ✅ CORRECT")
                correct += 1
            else:
                logger.info("Result: ❌ INCORRECT")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
    # 4. Final Report
    logger.info("\n--- Final Results ---")
    logger.info(f"Total Correct: {correct}/{len(tasks)}")
    logger.info(f"Accuracy: {correct/len(tasks):.2f}")

    loader.unload_model()

if __name__ == "__main__":
    verify_accuracy()
