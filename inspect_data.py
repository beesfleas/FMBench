from datasets import load_dataset
import logging

# Disable heavy logging
logging.getLogger("datasets").setLevel(logging.ERROR)

def inspect(name, config):
    print(f"--- Inspecting {name} ({config}) ---")
    try:
        ds = load_dataset(name, config, split="train", streaming=True)
        item = next(iter(ds))
        print(f"KEYS: {list(item.keys())}")
    except Exception as e:
        print(f"Error: {e}")

inspect("Salesforce/GiftEval", "default")
inspect("autogluon/fev_datasets", "ETT_1H")
