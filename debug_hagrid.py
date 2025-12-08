"""Debug script to see actual SmolVLM outputs for HaGRID."""
from datasets import load_dataset
from components.models.model_factory import get_model_loader
from components.scenarios.common_vlm_scenarios import HaGRIDScenario

# Load model
model_config = {
    "model_id": "HuggingfaceTB/SmolVLM-256M-Instruct",
    "model_category": "VLM"
}
loader = get_model_loader(model_config)
loader.load_model(model_config)

# Load dataset
ds = load_dataset('cj-mills/hagrid-sample-30k-384p', split='train')

# Create scenario
scenario = HaGRIDScenario(config={
    'dataset_name': 'cj-mills/hagrid-sample-30k-384p',
    'split': 'train',
    'num_samples': 3,
    'prompt_template': "What hand gesture is shown? Answer with one word."
})

tasks = scenario.process_dataset(ds.select(range(3)))

print("Checking actual model outputs:")
for i, task in enumerate(tasks):
    output = loader.predict(task['prompt'], image=task['image'])
    if isinstance(output, dict):
        output = output.get('output', str(output))
    
    metrics = scenario.compute_metrics(output, task['target'], task)
    print(f"\nTask {i}:")
    print(f"  Target: {task['target']}")
    print(f"  Output: {output[:200]}")
    print(f"  Accuracy: {metrics['accuracy']}")
