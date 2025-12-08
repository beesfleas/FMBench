from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def debug_gift():
    print("Loading Salesforce/GiftEval...")
    try:
        ds = load_dataset("Salesforce/GiftEval", "default", split="train", trust_remote_code=True, streaming=True)
        count = 0
        for entry in ds:
            print(f"Entry {count} keys: {entry.keys()}")
            target = entry.get('target')
            print(f"Target type: {type(target)}")
            if isinstance(target, list):
                print(f"Target length: {len(target)}")
            else:
                print(f"Target: {target}")
            
            count += 1
            if count >= 3:
                break
    except Exception as e:
        print(f"Error loading GIFT: {e}")

if __name__ == "__main__":
    debug_gift()
