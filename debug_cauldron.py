from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug():
    try:
        logger.info("Loading HuggingFaceM4/the_cauldron subset vqav2...")
        ds = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train", streaming=True, trust_remote_code=True)
        item = next(iter(ds))
        logger.info(f"Keys: {item.keys()}")
        logger.info(f"Images type: {type(item.get('images'))}")
        logger.info(f"Texts type: {type(item.get('texts'))}")
        logger.info(f"Texts content: {item.get('texts')}")
        if item.get('images') and isinstance(item.get('images'), list):
             logger.info(f"First image: {item.get('images')[0]}")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    debug()
