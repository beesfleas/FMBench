
import logging
import re
from components.scenarios.common_nlp_scenarios import TextClassificationScenario

logging.basicConfig(level=logging.INFO)

# Mock config
config = {
    "label_map": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    "input_key": "text",
    "target_key": "label"
}

# Instantiate scenario (hacky init just for compute_metrics)
scenario = TextClassificationScenario(config)

def test_grading(output, target, description):
    # Create a dummy task usually just needs input key if used, but compute_metrics mostly uses output/target
    task_info = {"input": "Classify..."} 
    result = scenario.compute_metrics(output, target, task_info)
    print(f"[{description}] Target: '{target}' | Output: '{output[:30]}...' -> Accuracy: {result['accuracy']}")
    return result['accuracy']

# Test cases from user logs
# 1. Formatting with Markdown and extra text
output1 = "**Business**\n\nThe article discusses union negotiations..."
target1 = "Business"

# 2. Correct but verbose
output2 = "Sci/Tech\n\nThis news article is about a private tea..."
target2 = "Sci/Tech"

# 3. Another verbose one
output3 = "Science/Technology\n\nThe article discusses a resear..."
target3 = "Sci/Tech" 

# 4. Prompt repetition (simulated)
output4 = "Classify the following... Category: World"
target4 = "World"

# 5. New aliases
output5 = "Sci\nThe article talks about..."
target5 = "Sci/Tech"

output6 = "Tech\nThis is about gadgets..."
target6 = "Sci/Tech"

print("--- Reproduction Tests ---")
failures = 0
if test_grading(output1, target1, "Markdown Bold + Context") != 1.0: failures += 1
if test_grading(output2, target2, "Label at start + Context") != 1.0: failures += 1
if test_grading(output3, target3, "Variant Label (Science/Technology)") != 1.0: failures += 1
if test_grading(output5, target5, "Alias 'Sci'") != 1.0: failures += 1
if test_grading(output6, target6, "Alias 'Tech'") != 1.0: failures += 1

print(f"\nTotal Failures: {failures}")
