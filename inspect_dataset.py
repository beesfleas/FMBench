from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

try:
    print("Loading lmms-lab/DocVQA validation split with config 'DocVQA'...")
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    print("Dataset loaded via streaming.")
    
    first_item = next(iter(dataset))
    print("First item keys:", first_item.keys())
    print("First item sample:", {k: str(v)[:50] for k, v in first_item.items()})

except Exception as e:
    print(f"Error: {e}")
