
import unittest
from unittest.mock import MagicMock, patch
import torch
from omegaconf import OmegaConf

# Mocking modules to avoid loading heavy dependencies during quick verification if needed
# But ideally we want to test the actual logic.
# Let's try to import the actual classes.

try:
    from components.models.huggingface_llm import HuggingFaceLLMLoader
    from components.scenarios.perplexity_scenario import PerplexityScenario
    from core.runner import run_scenario
except ImportError as e:
    print(f"Import failed: {e}")
    # This might happen if run from wrong dir
    import sys
    sys.path.append('.')
    from components.models.huggingface_llm import HuggingFaceLLMLoader
    from components.scenarios.perplexity_scenario import PerplexityScenario
    from core.runner import run_scenario

class TestPerplexity(unittest.TestCase):

    def setUp(self):
        # Setup specific config
        self.config = OmegaConf.create({
            "model_id": "gpt2",
            "device": "cpu",
            "max_length": 128,
            "stride": 64
        })

    @patch('components.models.huggingface_llm.AutoModelForCausalLM')
    @patch('components.models.huggingface_llm.AutoTokenizer')
    def test_compute_perplexity_logic(self, mock_tokenizer_cls, mock_model_cls):
        # Setup Mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Mock encoding
        mock_input_ids = torch.randint(0, 1000, (1, 200)) # 200 tokens
        mock_tokenizer.return_value = MagicMock(input_ids=mock_input_ids)
        mock_tokenizer.pad_token = None
        
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Mock model output
        # Return a loss
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.5) # Some loss
        mock_model.return_value = mock_output
        mock_model.config.max_position_embeddings = 1024
        
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])

        loader = HuggingFaceLLMLoader()
        loader.load_model(self.config)
        
        # Run compute_perplexity
        ppl = loader.compute_perplexity("Some text content " * 10)
        
        print(f"Computed Perplexity: {ppl}")
        self.assertIsInstance(ppl, float)
        self.assertGreater(ppl, 0.0)
        
    def test_perplexity_scenario_loading(self):
        # Test that tasks are created
        config = {
            "name": "test_perp",
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "split": "test",
            "text_column": "text",
            "num_samples": 5
        }
        
        # Mock datasets.load_dataset to avoid downloading
        with patch('datasets.load_dataset') as mock_load:
            mock_data = [
                {"text": "Line 1"},
                {"text": "Line 2"},
                {"text": ""} # Empty execution
            ]
            mock_load.return_value = mock_data
            
            scenario = PerplexityScenario(config)
            scenario.load_tasks()
            
            self.assertEqual(len(scenario.tasks), 2) # Should filter empty
            self.assertEqual(scenario.tasks[0]['input'], "Line 1")

if __name__ == '__main__':
    unittest.main()
