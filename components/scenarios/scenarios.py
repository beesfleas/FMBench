import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
import re

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class FMBenchLLM(LLM):
    """
    Custom LangChain LLM wrapper for FMBench models.
    This allows us to use LangChain's chains and prompts with our models.
    """
    
    # We need to type hint the model appropriately, but since it can be various things
    # (HuggingFace pipeline, generic callable, etc.), we keep it Any for now.
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
        """
        Execute the model on the given prompt.
        
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating.
            run_manager: Callback manager.
            **kwargs: Arbitrary additional keyword arguments.
            
        Returns:
            The model output as a string.
        """
        # This assumes the model object has a 'generate' or similar method, 
        # or is a callable that takes a prompt. 
        # We might need to adapt this based on the actual model objects used in FMBench.
        # For now, assuming a simple callable or a generate method.
        
        if hasattr(self.model, "generate"):
             # Adjust based on actual signature of FMBench models
            response = self.model.generate(prompt, **kwargs)
        elif callable(self.model):
            response = self.model(prompt, **kwargs)
        else:
            raise ValueError(f"Model {self.model_name} is not callable and has no generate method.")

        # If response is not a string, try to extract text. 
        # This is highly dependent on the model's return type.
        if not isinstance(response, str):
            # specific handling for common return types could go here
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
        """
        Returns a list of tasks (inputs and expected outputs) for this scenario.
        Each task should be a dictionary.
        """
        pass

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs the evaluation for this scenario.
        """
        tasks = self.get_tasks()
        results = []
        
        total_metrics = {
            "accuracy": 0.0,
            "latency_sum": 0.0,
            "count": 0
        }

        logger.info(f"Starting evaluation for scenario: {self.name} with {len(tasks)} tasks.")

        for task in tasks:
            input_text = task.get("input")
            expected_output = task.get("target")
            
            # Prepare prompt
            prompt = PromptTemplate.from_template(self.prompt_template)
            formatted_prompt = prompt.format(input=input_text)
            
            # Run inference
            start_time = time.time()
            try:
                output = self.llm.invoke(formatted_prompt)
            except Exception as e:
                logger.error(f"Error running model on task: {e}")
                output = ""
            end_time = time.time()
            latency = end_time - start_time
            
            # Calculate metrics
            task_metrics = self.compute_metrics(output, expected_output, task)
            task_metrics["latency"] = latency
            
            # Aggregate
            total_metrics["accuracy"] += task_metrics.get("accuracy", 0.0)
            total_metrics["latency_sum"] += latency
            total_metrics["count"] += 1
            
            results.append({
                "input": input_text,
                "target": expected_output,
                "output": output,
                "metrics": task_metrics
            })

        # Final aggregation
        if total_metrics["count"] > 0:
            final_metrics = {
                "average_accuracy": total_metrics["accuracy"] / total_metrics["count"],
                "average_latency": total_metrics["latency_sum"] / total_metrics["count"],
                "total_samples": total_metrics["count"]
            }
        else:
            final_metrics = {"error": "No tasks processed"}

        return {
            "scenario_name": self.name,
            "metrics": final_metrics,
            "details": results
        }

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes metrics for a single task.
        Can be overridden by subclasses for specific metrics.
        """
        metrics = {}
        
        # Basic Exact Match / Contains accuracy
        if isinstance(target, str):
            metrics["accuracy"] = 1.0 if target.strip().lower() in output.strip().lower() else 0.0
        
        # Perplexity (Placeholder - requires logits usually)
        # metrics["perplexity"] = ... 
        
        # Fairness (Simple keyword check example)
        # Check for biased terms (very basic list for demonstration)
        # biased_terms = ["bias", "stereotype"] # This should be a comprehensive list or config
        # metrics["fairness_flag"] = 1.0 if any(term in output.lower() for term in biased_terms) else 0.0
        
        return metrics
