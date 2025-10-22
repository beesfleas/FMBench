# components/models/huggingface.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModelLoader

class HuggingFaceLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, config):
        model_id = config.get("model_id")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}") from e

    def predict(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
