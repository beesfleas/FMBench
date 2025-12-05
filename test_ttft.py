from components.models.huggingface_llm import HuggingFaceLLMLoader
import logging

logging.basicConfig(level=logging.DEBUG)

config = {
    "model_id": "distilgpt2",
    "device": {"type": "cpu"},
    "max_tokens": 10
}

loader = HuggingFaceLLMLoader()
loader.load_model(config)
result = loader.predict("Hello")
print("RESULT:", result)
