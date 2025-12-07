from typing import List, Dict, Any, Optional
import logging

from .scenarios import Scenario

log = logging.getLogger(__name__)

try:
    import datasets
except ImportError:
    datasets = None

class DatasetScenario(Scenario):
    """
    Base class for scenarios that load data from Hugging Face datasets.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model: Any = None, **kwargs):
        # Handle both Hydra instantiation (kwargs) and manual instantiation (config dict)
        if config is None:
            config = kwargs
        else:
            config.update(kwargs)
            
        super().__init__(config, model)
        self.dataset_name = config.get("dataset_name")
        self.dataset_config = config.get("dataset_config", None)
        self.split = config.get("split", "validation")
        self.num_samples = config.get("num_samples", None)
        self.tasks = []

    def get_tasks(self) -> List[Dict[str, Any]]:
        if not self.tasks:
            self.load_tasks()
        return self.tasks

    def load_tasks(self):
        """Load tasks from the dataset"""
        if datasets is None:
            log.error("datasets library not found. Please install it with `pip install datasets`")
            self.tasks = []
            return

        log.info("Loading dataset: %s (config: %s, split: %s)", 
                 self.dataset_name, self.dataset_config, self.split)
        try:
            streaming = self.config.get("streaming", False)
            self.dataset = datasets.load_dataset(
                self.dataset_name, 
                self.dataset_config, 
                split=self.split,
                trust_remote_code=self.config.get("trust_remote_code", True),
                streaming=streaming
            )
            
            if self.num_samples:
                if streaming:
                    self.dataset = self.dataset.take(self.num_samples)
                else:
                    self.dataset = self.dataset.select(range(min(len(self.dataset), self.num_samples)))
            
            self.tasks = self.process_dataset(self.dataset)
            log.info("Loaded %d tasks from %s", len(self.tasks), self.dataset_name)
        except FileNotFoundError as e:
            log.error("Dataset '%s' not found: %s", self.dataset_name, e)
            self.tasks = []
        except ValueError as e:
            log.error("Invalid dataset configuration for '%s': %s", self.dataset_name, e)
            self.tasks = []
        except ConnectionError as e:
            log.error("Network error loading dataset '%s': %s", self.dataset_name, e)
            self.tasks = []
        except Exception as e:
            log.error("Unexpected error loading dataset '%s': %s", self.dataset_name, e)
            self.tasks = []

    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        """
        Process the HF dataset into a list of task dictionaries.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        """Evaluate model output for a given task."""
        expected_output = task.get("target")
        return self.compute_metrics(model_output, expected_output, task)
