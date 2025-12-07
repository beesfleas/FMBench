import logging
from components.scenarios.common_vlm_scenarios import VQAScenario

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_vqa():
    logger.info("Verifying VQAScenario...")
    config = {
        "dataset_name": "HuggingFaceM4/the_cauldron",
        "dataset_config": "vqav2",
        "split": "train",
        "num_samples": 5,
        "image_key": "images",            # Expected key for cauldron
        "question_key": "texts",          # Expected key for cauldron (conversion needed)
        "target_key": "texts",
        "prompt_template": "Question: {question} Answer:",
        "trust_remote_code": True,
        "streaming": True 
    }
    try:
        scenario = VQAScenario(config)
        tasks = scenario.get_tasks()
        if tasks:
            logger.info(f"Loaded {len(tasks)} tasks.")
            first_task = tasks[0]
            logger.info(f"Sample task keys: {first_task.keys()}")
            logger.info(f"Sample prompt: {first_task.get('prompt')}")
            logger.info(f"Sample target: {first_task.get('target')}")
            
            # Verify image presence (but don't print binary)
            if "image" in first_task:
                 logger.info(f"Image present: {type(first_task['image'])}")
            else:
                 logger.error("Image NOT present in task!")
        else:
            logger.error("No tasks loaded for VQAScenario")
    except Exception as e:
        logger.error(f"Failed to verify VQA: {e}")

if __name__ == "__main__":
    verify_vqa()
