from components.models.huggingface import HuggingFaceLoader

def main():
    config = {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    }
    loader = HuggingFaceLoader()
    print("Loading model...")
    loader.load_model(config)
    print("Model loaded.")

    prompt = "Once upon a time,"
    result = loader.predict(prompt)
    print("Result:", result)

if __name__ == "__main__":
    main()
