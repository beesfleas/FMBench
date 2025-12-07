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

def verify_leakage():
    dataset_config = {
        "name": "classification",
        "scenario_class": "components.scenarios.common_nlp_scenarios.TextClassificationScenario",
        "dataset_name": "ag_news",
        "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        # A prompt that lists the options
        "prompt_template": "Options: World, Sports, Business, Sci/Tech. Classify: {input}"
    }

    scenario = TextClassificationScenario(config=dataset_config)
    
    # Simulating a scenario where model repeats the prompt input
    # Target is "Sports"
    target = "Sports" 
    
    # Model Output contains the prompt (which has "Sports") BUT the actual answer is "World" (incorrect)
    output_hallucinated = "Options: World, Sports, Business, Sci/Tech. Classify: Something. Answer: World"
    
    metrics = scenario.compute_metrics(output_hallucinated, target, {})
    print(f"Target: {target}")
    print(f"Output: {output_hallucinated}")
    print(f"Score: {metrics['accuracy']}")
    
    if metrics['accuracy'] == 1.0:
        print("CRITICAL FAIL: Grading gave 1.0 because 'Sports' was found in the prompt portion of the output, even though the answer was 'World'.")
    else:
        print("PASS: Grading correctly identified the mismatch.")

if __name__ == "__main__":
    verify_leakage()
