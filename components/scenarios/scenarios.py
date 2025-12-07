import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

log = logging.getLogger(__name__)

class FMBenchLLM(LLM):
    """LangChain LLM wrapper for FMBench models."""
    
    model: Any = None
    model_name: str = "custom_model"
    
    def __init__(self, model: Any, model_name: str = "custom_model", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model_name = model_name

    @property
    def _llm_type(self) -> str:
        return "fmbench_custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the model on the given prompt."""
        if hasattr(self.model, "generate"):
            response = self.model.generate(prompt, **kwargs)
        elif callable(self.model):
            response = self.model(prompt, **kwargs)
        else:
            raise ValueError(f"Model {self.model_name} is not callable and has no generate method.")

        if not isinstance(response, str):
            response = str(response)

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        return response

class Scenario(ABC):
    """
    Base class for all evaluation scenarios.
    """
    
    def __init__(self, config: Dict[str, Any], model: Any = None):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.model = model
        self.llm = FMBenchLLM(model=model, model_name=self.name) if model else None
        self.prompt_template = config.get('prompt_template', "{input}")

    @abstractmethod
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Returns list of task dicts with 'input' and 'target' keys."""
        pass

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for a task. Override in subclasses for custom metrics."""
        metrics = {}
        if isinstance(target, str):
            metrics["accuracy"] = 1.0 if target.strip().lower() in output.strip().lower() else 0.0
        return metrics
