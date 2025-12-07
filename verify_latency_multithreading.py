import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from components.models.huggingface_llm import HuggingFaceLLMLoader

def verify_latency_multithreading():
    print("Starting verification of latency multithreading...")
    
    # Mock configuration
    config = {
        "model_id": "distilgpt2",
        "device": {"type": "cpu"}, # Force CPU for simplicity in test
        "max_tokens": 10
    }
    
    # Initialize loader
    loader = HuggingFaceLLMLoader()
    
    # We want to use a real model to ensure the code path is actually traversable
    # calling load_model might be slow, so we can mock the internal parts of load_model
    # IF we just want to test the threading logic in predict.
    # However, predict needs self.tokenizer and self.model.
    
    import torch

    print("Mocking model and tokenizer loading...")
    
    # Create mocks
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    # Mock tokenizer call returns a dict with real tensors
    real_input_ids = torch.ones((1, 5), dtype=torch.long)
    real_attention_mask = torch.ones((1, 5), dtype=torch.long)
    
    # We mock the return value of the tokenizer call
    mock_tokenizer.return_value = {
        "input_ids": real_input_ids,
        "attention_mask": real_attention_mask
    }
    mock_tokenizer.decode.return_value = "Mocked output"
    mock_tokenizer.pad_token_id = 50256
    
    mock_model = MagicMock()
    # Mock parameters().device
    mock_param = MagicMock()
    mock_param.device = "cpu"
    mock_model.parameters.return_value = iter([mock_param])
    
    # Mock generate to take a little bit of time so we can measure latency > 0
    def mock_generate(*args, **kwargs):
        time.sleep(0.1) 
        # Return real tensor of shape [1, 15] (5 input + 10 new)
        # inputs + new tokens
        return torch.ones((1, 15), dtype=torch.long)

    mock_model.generate.side_effect = mock_generate
    
    # Inject mocks into loader
    loader.tokenizer = mock_tokenizer
    loader.model = mock_model
    loader.config = config
    
    # Track thread creation
    original_thread_init = threading.Thread.__init__
    created_threads = []

    def side_effect_init(self, *args, **kwargs):
        created_threads.append(self)
        original_thread_init(self, *args, **kwargs)

    print("Patching threading.Thread to verify standard usage...")
    with patch('threading.Thread', side_effect=side_effect_init, autospec=True) as MockThread:
        # We need to forward the call to the real threading.Thread so it actually runs
        # But we also want to catch that it was called.
        # simpler: patch where it is used or just wrap the class.
        pass

    # Actually, let's use a simpler approach: 
    # Just verify that loader.predict runs and returns a latency that essentially matches our sleep.
    
    print("Running loader.predict...")
    start_time_glob = time.time()
    
    # We also need to mock TTFTStreamer because it's imported in the module
    # but we can probably let it run if we don't mock it, 
    # acts on mocked model.generate which might not call streamer hooks.
    # The code: streamer = TTFTStreamer() -> pass to generate -> generate calls streamer.put/end
    # Our mock_generate doesn't call streamer, so ttft will be None.
    # That's fine, we are testing latency threading.
    
    with patch('threading.Thread', wraps=threading.Thread) as mock_thread_cls:
        result = loader.predict("Test prompt")
        
        # Verify a thread was created
        if mock_thread_cls.call_count > 0:
            print("PASS: A new thread was created.")
        else:
            print("FAIL: No new thread was created.")
            sys.exit(1)
            
        print(f"Result: {result}")
        
        # Check latency
        latency = result.get('latency')
        if latency is not None and latency >= 0.1:
            print(f"PASS: Latency captured correctly ({latency:.4f}s >= 0.1s)")
        else:
            print(f"FAIL: Latency incorrect or missing (got {latency})")
            sys.exit(1)

        # Check token count
        num_tokens = result.get('num_tokens')
        # We mocked generate to return [1, 15] and mocked input was [1, 5]
        # So new tokens should be 10.
        if num_tokens == 10:
            print(f"PASS: Token count captured correctly ({num_tokens})")
        else:
            print(f"FAIL: Token count incorrect (got {num_tokens}, expected 10)")
            sys.exit(1)

        # Confirm the thread joined (we can't easily check join on the mock unless we return a mock thread, 
        # but wraps returns a real thread). 
        # If we got the result, join must have happened or we'd be racing/erroring on accessing thread_results.
        
    print("Verification PASSED")

if __name__ == "__main__":
    verify_latency_multithreading()
