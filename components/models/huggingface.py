from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModelLoader

class HuggingFaceLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

    def predict(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=100)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
