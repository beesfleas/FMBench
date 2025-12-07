"""
Verification script for CountBenchQA scenario.
Usage: python verify_countbenchqa.py [--with-model]
"""
import logging
import warnings
import argparse
from components.scenarios.common_vlm_scenarios import CountBenchQAScenario

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_scenario_loading():
    """Verify that the scenario can load tasks from the dataset."""
    logger.info("=" * 50)
    logger.info("CountBenchQA Scenario Verification")
    logger.info("=" * 50)
    
    config = {
        "dataset_name": "vikhyatk/CountBenchQA",
        "split": "test",
        "num_samples": 5,
        "image_key": "image",
        "question_key": "question",
        "target_key": "number",
        "prompt_template": "Question: {question}\nAnswer with just the number:",
        "trust_remote_code": True
    }
    
    logger.info(f"\nLoading scenario with config:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    scenario = CountBenchQAScenario(config)
    tasks = scenario.get_tasks()
    
    logger.info(f"\n✅ Loaded {len(tasks)} tasks successfully")
    
    # Show sample tasks
    logger.info("\n--- Sample Tasks ---")
    for i, task in enumerate(tasks[:3]):
        logger.info(f"\nTask {i+1}:")
        logger.info(f"  Question: {task.get('input', 'N/A')}")
        logger.info(f"  Target: {task.get('target', 'N/A')}")
        logger.info(f"  Has Image: {task.get('image') is not None}")
    
    return scenario, tasks


def verify_metrics():
    """Verify that the metrics computation works correctly."""
    logger.info("\n" + "=" * 50)
    logger.info("Metrics Verification")
    logger.info("=" * 50)
    
    config = {"dataset_name": "vikhyatk/CountBenchQA"}
    scenario = CountBenchQAScenario(config)
    
    test_cases = [
        # (output, target, expected_accuracy)
        ("5", "5", 1.0),
        ("There are 5 cats in the image.", "5", 1.0),
        ("I count five objects.", "5", 1.0),
        ("The answer is 3.", "5", 0.0),
        ("two", "2", 1.0),
        ("10", "10", 1.0),
    ]
    
    logger.info("\nTesting compute_metrics:")
    all_passed = True
    for output, target, expected in test_cases:
        metrics = scenario.compute_metrics(output, target, {})
        actual = metrics.get("accuracy", 0.0)
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_passed = False
        logger.info(f"{status} Output: '{output}' | Target: '{target}' | Accuracy: {actual} (expected: {expected})")
    
    return all_passed


def verify_with_model():
    """End-to-end verification with a real VLM model."""
    import torch
    from components.models.huggingface_vlm import HuggingFaceVLMLoader
    
    logger.info("\n" + "=" * 50)
    logger.info("End-to-End Verification with Model")
    logger.info("=" * 50)
    
    # Load scenario
    config = {
        "dataset_name": "vikhyatk/CountBenchQA",
        "split": "test",
        "num_samples": 3,
        "image_key": "image",
        "question_key": "question",
        "target_key": "number",
        "prompt_template": "Question: {question}\nAnswer with just the number:",
        "trust_remote_code": True
    }
    
    scenario = CountBenchQAScenario(config)
    tasks = scenario.get_tasks()
    
    # Load model
    model_config = {
        "model_id": "HuggingfaceTB/SmolVLM-256M-Instruct",
        "device": {"type": "cuda" if torch.cuda.is_available() else "cpu"},
        "max_tokens": 20
    }
    
    logger.info(f"\nLoading Model: {model_config['model_id']}")
    loader = HuggingFaceVLMLoader()
    loader.load_model(model_config)
    
    # Run predictions
    correct = 0
    for i, task in enumerate(tasks):
        logger.info(f"\nTask {i+1}:")
        
        prompt = task.get("prompt") or task.get("input")
        image = task.get("image")
        target = task.get("target")
        
        logger.info(f"Question: {prompt}")
        logger.info(f"Target: {target}")
        
        try:
            result = loader.predict(prompt, image=image)
            output = result["output"] if isinstance(result, dict) else result
            logger.info(f"Model Output: {output}")
            
            metrics = scenario.compute_metrics(output, target, task)
            accuracy = metrics.get("accuracy", 0.0)
            
            if accuracy > 0:
                logger.info("Result: ✅ CORRECT")
                correct += 1
            else:
                logger.info("Result: ❌ INCORRECT")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
    
    logger.info("\n--- Final Results ---")
    logger.info(f"Total Correct: {correct}/{len(tasks)}")
    logger.info(f"Accuracy: {correct/len(tasks):.2%}")
    
    loader.unload_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify CountBenchQA scenario")
    parser.add_argument("--with-model", action="store_true", help="Run end-to-end verification with VLM model")
    args = parser.parse_args()
    
    # Always run these
    verify_scenario_loading()
    verify_metrics()
    
    # Optionally run with model
    if args.with_model:
        verify_with_model()
    else:
        logger.info("\n💡 Run with --with-model flag to test with a real VLM model")
    
    logger.info("\n✅ Verification complete!")
