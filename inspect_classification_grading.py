import logging
import sys
import os

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)

try:
    from components.scenarios.common_nlp_scenarios import TextClassificationScenario
except ImportError as e:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from components.scenarios.common_nlp_scenarios import TextClassificationScenario

def inspect_grading():
    dataset_config = {
        "name": "classification",
        "scenario_class": "components.scenarios.common_nlp_scenarios.TextClassificationScenario",
        "dataset_name": "ag_news",
        "dataset_config": None,
        "split": "test",
        "input_key": "text",
        "target_key": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "num_samples": 5,
        "prompt_template": "Classify: {input}"
    }

    print("Initializing TextClassificationScenario...")
    scenario = TextClassificationScenario(config=dataset_config)
    
    # We don't necessarily need real data to test the METRIC logic, but let's grab one task structure
    # to be safe about method signature.
    task_mock = {"input": "foo", "target": "World", "type": "classification"}
    
    print("\n--- Testing Grading Logic ---")
    
    cases = [
        ("World", "World", 1.0, "Exact Match"),
        ("World", "world", 1.0, "Case Insensitive"),
        ("World", "The category is World.", 1.0, "Containment (Verbose)"),
        ("World", "Sports", 0.0, "Incorrect Label"),
        ("World", "The category is unrelated.", 0.0, "No Match"),
        ("World", "World Sports", 1.0, "Ambiguous Containment (False Positive Risk?)") 
    ]
    
    for target, output, expected_score, desc in cases:
        metrics = scenario.compute_metrics(output, target, task_mock)
        score = metrics.get("accuracy", -1)
        status = "PASS" if score == expected_score else "FAIL"
        print(f"Case: {desc}")
        print(f"  Target: '{target}' | Output: '{output}'")
        print(f"  Score: {score} (Expected: {expected_score}) -> {status}")
        
    print("\nNote: Default logic is 'target in output'. This allows dense reasoning traces to pass if they confirm the label.")

if __name__ == "__main__":
    inspect_grading()
