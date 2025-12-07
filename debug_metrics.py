import logging
import sys
from components.scenarios.common_nlp_scenarios import SummarizationScenario

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_metrics():
    print("Initializing SummarizationScenario...")
    # Minimal config
    config = {
        "dataset_name": "cnn_dailymail",
        "dataset_config": "3.0.0",
        "split": "test",
        "num_samples": 1
    }
    
    # We don't need real data loading for testing compute_metrics
    scenario = SummarizationScenario(config)
    
    # Dummy data
    output = "The cat sat on the mat."
    target = "The cat was sitting on the mat."
    task = {"input": "Original text..."}
    
    print("\n--- Testing compute_metrics ---")
    metrics = scenario.compute_metrics(output, target, task)
    print("Metrics result:", metrics)
    
    if "meteor" not in metrics:
        print("METEOR missing!")
    if "bert_score" not in metrics:
        print("BERT Score missing!")
    if "bleu" not in metrics:
        print("BLEU missing!")

if __name__ == "__main__":
    test_metrics()
