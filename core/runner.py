# core/runner.py
from components.models.huggingface import HuggingFaceLoader

def run_benchmark(cfg):
    model_config = cfg.model # cfg.model has everything defined in the chosen YAML model group.
    loader = HuggingFaceLoader()
    print("Loading model:", model_config.model_id)
    loader.load_model(dict(model_id=model_config.model_id))
    prompt = "Once upon a time,"
    result = loader.predict(prompt)
    print("Result:", result)
