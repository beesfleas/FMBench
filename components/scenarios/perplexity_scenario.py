from typing import List, Dict, Any, Optional
import logging

from .dataset_scenario import DatasetScenario

log = logging.getLogger(__name__)

class PerplexityScenario(DatasetScenario):
    """
    Scenario for evaluating perplexity on a dataset.
    This scenario treats the entire dataset (or a subset) as a sequence of texts
    and computes the perplexity for each text.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model: Any = None, **kwargs):
        super().__init__(config, model, **kwargs)
        self.text_column = self.config.get("text_column", "text")

    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        """
        Process the dataset into a list of tasks.
        For perplexity, each "task" is just a text sample to evaluate.
        """
        tasks = []
        log.info("Processing dataset for perplexity. Text column: %s", self.text_column)
        for i, item in enumerate(dataset):
            text = item.get(self.text_column)
            if text:
                tasks.append({
                    "input": text,
                    "target": None  # No explicit target for perplexity, the text itself is the target
                })
            else:
                log.warning("Item %d in dataset is missing text column '%s'", i, self.text_column)
        
        return tasks

    def evaluate(self, task: Dict[str, Any], model_output: Any = None) -> Dict[str, Any]:
        """
        Evaluate perplexity for a given task.
        Since perplexity is computed by the loader directly (usually), 
        'model_output' might be the perplexity value itself if the runner handles it,
        OR we might need to compute it here if the runner passes the model.
        
        However, standard runner.py flow gets 'output' from loader.predict().
        We need to coordinate with runner.py to call compute_perplexity instead of predict.
        
        If model_output is already the perplexity float (handled in runner modification), just return it.
        """
        metrics = {}
        
        # If runner passed the float result directly:
        if isinstance(model_output, (float, int)):
            metrics["perplexity"] = float(model_output)
        else:
            # Fallback or error
             metrics["perplexity"] = float("nan")
             
        return metrics
