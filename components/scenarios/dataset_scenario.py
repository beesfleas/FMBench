from typing import List, Dict, Any, Optional
from .scenarios import Scenario
import logging

logger = logging.getLogger(__name__)

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
            logger.error("datasets library not found. Please install it with `pip install datasets`")
            self.tasks = []
            return

        logger.info(f"Loading dataset: {self.dataset_name} (config: {self.dataset_config}, split: {self.split})")
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
            logger.info(f"Loaded {len(self.tasks)} tasks from {self.dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            self.tasks = []

    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        """
        Process the HF dataset into a list of task dictionaries.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        """
        Evaluate model output for a given task.
        Compatible with runner.py loop.
        """
        input_text = task.get("input")
        expected_output = task.get("target")
        
        # Calculate metrics using the base class logic or override
        metrics = self.compute_metrics(model_output, expected_output, task)
        return metrics
