from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseModelLoader

class HuggingFaceLLMLoader(BaseModelLoader):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model(self, config):
        model_id = config.get("model_id")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.tokenizer.pad_token = self.tokenizer.unk_token   # idk which to use
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        print(f"Loaded LLM model: {model_id}")

    def predict(self, prompt, image=None, time_series_data=None):
        if image is not None:
            print("Warning: Image input provided to text-only model, ignoring image")

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        output_ids = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=self.config.get("max_tokens", 64),
            pad_token_id=self.tokenizer.pad_token_id     # Explicitly set pad_token_id (idk why this is needed but it works)
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
