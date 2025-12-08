"""
Verification script for vLLM loader.
Tests vLLM against multiple NLP scenarios: arc_easy, classification, ner,
sentiment, summarization, and translation.

Usage:
    conda activate testingReq
    python verify_vllm.py
    python verify_vllm.py --scenario sentiment
    python verify_vllm.py --compare  # Compare vLLM vs HuggingFace
"""
import sys
import os
import time
import logging
import argparse

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from components.models.model_factory import get_model_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Scenarios to test
SCENARIOS = ["arc_easy", "classification", "ner", "sentiment", "summarization", "translation"]

# Sample prompts for each scenario type
SCENARIO_PROMPTS = {
    "arc_easy": "Question: What is the main source of energy for the Earth?\nA) The Moon\nB) The Sun\nC) Stars\nD) Planets\n\nAnswer:",
    "classification": "Classify the following text into one of these categories: World, Sports, Business, Technology.\n\nText: The company announced record profits in the quarterly earnings report.\n\nCategory:",
    "ner": "Extract named entities from the following text:\n\nText: Apple CEO Tim Cook announced a new product launch in San Francisco.\n\nEntities:",
    "sentiment": "Analyze the sentiment of the following text. Answer 'Positive' or 'Negative'.\n\nText: This movie was absolutely fantastic! I loved every minute of it.\n\nSentiment:",
    "summarization": "Summarize the following text in one sentence:\n\nText: Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The fish lives at depths of over 8,000 meters and has unique adaptations for surviving in extreme pressure. Researchers believe this discovery could help us understand how life adapts to harsh environments.\n\nSummary:",
    "translation": "Translate the following text from German to English:\n\nGerman: Guten Morgen, wie geht es Ihnen heute?\n\nEnglish:",
}


def test_vllm_loader():
    """Test that vLLM loader can be instantiated."""
    logger.info("Testing vLLM loader instantiation...")
    
    config = {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_category": "LLM",
        "loader_type": "vllm",
        "max_tokens": 64,
        "gpu_memory_utilization": 0.5,  # Use less memory for testing
    }
    
    try:
        loader = get_model_loader(config)
        logger.info("PASS: vLLM loader instantiated: %s", loader.__class__.__name__)
        return loader, config
    except ImportError as e:
        logger.error("SKIP: vLLM not installed - %s", e)
        return None, config
    except Exception as e:
        logger.error("FAIL: Could not create vLLM loader - %s", e)
        return None, config


def test_model_loading(loader, config):
    """Test model loading with vLLM."""
    logger.info("Testing model loading...")
    
    try:
        start = time.time()
        loader.load_model(config)
        elapsed = time.time() - start
        logger.info("PASS: Model loaded in %.2fs", elapsed)
        return True
    except Exception as e:
        logger.error("FAIL: Model loading failed - %s", e)
        return False


def test_inference(loader, scenario_name):
    """Test inference with a sample prompt."""
    prompt = SCENARIO_PROMPTS.get(scenario_name, SCENARIO_PROMPTS["sentiment"])
    logger.info("Testing inference for scenario: %s", scenario_name)
    
    try:
        start = time.time()
        result = loader.predict(prompt)
        elapsed = time.time() - start
        
        logger.info("PASS: Inference completed in %.3fs", elapsed)
        logger.info("  Output: %s", result.get("output", "")[:100])
        logger.info("  TTFT: %s", result.get("ttft"))
        logger.info("  Latency: %.3fs", result.get("latency", 0))
        logger.info("  Tokens: %d", result.get("num_tokens", 0))
        return result
    except Exception as e:
        logger.error("FAIL: Inference failed - %s", e)
        return None


def test_all_scenarios(loader):
    """Test inference across all requested scenarios."""
    logger.info("Testing all scenarios...")
    
    results = {}
    for scenario in SCENARIOS:
        logger.info("\n--- Testing %s ---", scenario)
        result = test_inference(loader, scenario)
        results[scenario] = result
    
    # Summary
    passed = sum(1 for r in results.values() if r is not None)
    logger.info("\n=== Summary ===")
    logger.info("Passed: %d/%d scenarios", passed, len(SCENARIOS))
    
    return results


def test_unload(loader):
    """Test model unloading."""
    logger.info("Testing model unload...")
    
    try:
        loader.unload_model()
        logger.info("PASS: Model unloaded")
        return True
    except Exception as e:
        logger.error("FAIL: Unload failed - %s", e)
        return False


def compare_loaders():
    """Compare vLLM vs HuggingFace performance."""
    logger.info("Comparing vLLM vs HuggingFace loaders...")
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = SCENARIO_PROMPTS["sentiment"]
    num_runs = 3
    
    results = {"vllm": [], "huggingface": []}
    
    for loader_type in ["huggingface", "vllm"]:
        logger.info("\n--- Testing %s ---", loader_type)
        
        config = {
            "model_id": model_id,
            "model_category": "LLM",
            "loader_type": loader_type,
            "max_tokens": 64,
        }
        
        if loader_type == "vllm":
            config["gpu_memory_utilization"] = 0.5
        
        try:
            loader = get_model_loader(config)
            loader.load_model(config)
            
            # Warm-up
            loader.predict(prompt)
            
            # Benchmark runs
            for i in range(num_runs):
                result = loader.predict(prompt)
                latency = result.get("latency", 0)
                results[loader_type].append(latency)
                logger.info("  Run %d: %.3fs", i + 1, latency)
            
            loader.unload_model()
            
        except ImportError as e:
            logger.warning("SKIP: %s - %s", loader_type, e)
        except Exception as e:
            logger.error("FAIL: %s - %s", loader_type, e)
    
    # Print comparison
    logger.info("\n=== Comparison ===")
    for lt, times in results.items():
        if times:
            avg = sum(times) / len(times)
            logger.info("%s: avg=%.3fs", lt, avg)


def main():
    parser = argparse.ArgumentParser(description="Verify vLLM loader")
    parser.add_argument("--scenario", type=str, choices=SCENARIOS,
                        help="Test a specific scenario")
    parser.add_argument("--compare", action="store_true",
                        help="Compare vLLM vs HuggingFace performance")
    parser.add_argument("--all", action="store_true",
                        help="Test all scenarios")
    args = parser.parse_args()
    
    if args.compare:
        compare_loaders()
        return
    
    # Test vLLM loader
    loader, config = test_vllm_loader()
    if loader is None:
        logger.error("Cannot proceed without vLLM loader")
        sys.exit(1)
    
    # Load model
    if not test_model_loading(loader, config):
        sys.exit(1)
    
    try:
        if args.all:
            test_all_scenarios(loader)
        elif args.scenario:
            test_inference(loader, args.scenario)
        else:
            # Default: test one scenario
            test_inference(loader, "sentiment")
    finally:
        test_unload(loader)
    
    logger.info("\n=== Verification Complete ===")


if __name__ == "__main__":
    main()
