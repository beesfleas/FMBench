import logging
import sys
import os

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_classification")

try:
    from components.scenarios.common_nlp_scenarios import TextClassificationScenario
except ImportError as e:
    logger.error(f"ImportError: {e}")
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from components.scenarios.common_nlp_scenarios import TextClassificationScenario

def run_verification():
    dataset_config = {
        "name": "classification",
        "scenario_class": "components.scenarios.common_nlp_scenarios.TextClassificationScenario",
        "dataset_name": "ag_news", # Using ag_news defaults
        "dataset_config": None,
        "split": "test",
        "input_key": "text",
        "target_key": "label",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "num_samples": 5,
        "prompt_template": "Classify this news: {input}" # Custom template to test override
    }

    logger.info("Initializing TextClassificationScenario with config...")
    scenario = TextClassificationScenario(config=dataset_config)
    
    # Mock data
    mock_data = [
        {"text": "Refinery explosion raises oil prices.", "label": 2}, # Business
        {"text": "Olympics opening ceremony dates announced.", "label": 1} # Sports
    ]
    
    logger.info("Processing mock dataset...")
    tasks = scenario.process_dataset(mock_data)
    
    if not tasks:
        logger.error("No tasks processed!")
        return

    print("\n--- Checking Prompt Format ---")
    expected_start = "Classify this news: Refinery"
    input_text = tasks[0]['input']
    
    if input_text.startswith("Classify this news:"):
        print("SUCCESS: Input matches custom `prompt_template`.")
    elif "Classify the following news article" in input_text:
         print("WARNING: Input uses hardcoded format, ignoring `prompt_template`.")  
    else:
        print(f"WARNING: Input format is unknown: {input_text}")
        
    print("\n--- Checking Label Mapping ---")
    expected_label = "Business"
    actual_label = tasks[0]['target']
    
    if actual_label == expected_label:
        print(f"SUCCESS: Label mapped correctly ({actual_label}).")
    else:
        print(f"FAILURE: Label mapping incorrect. Expected {expected_label}, got {actual_label}")

if __name__ == "__main__":
    run_verification()
