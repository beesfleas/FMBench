from typing import List, Dict, Any
from .scenarios import Scenario

class ReasoningScenario(Scenario):
    """
    Scenario for reasoning tasks (e.g., Chain of Thought, Math, Logic).
    """
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        # In a real implementation, this would load from a dataset file.
        # For now, we return a few sample tasks.
        return [
            {
                "input": "If I have 3 apples and I eat one, how many do I have?",
                "target": "2",
                "type": "arithmetic"
            },
            {
                "input": "All men are mortal. Socrates is a man. Therefore...",
                "target": "Socrates is mortal",
                "type": "logic"
            }
        ]

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        metrics = super().compute_metrics(output, target, task)
        
        # Add reasoning-specific metrics
        # e.g., check for "Therefore" or "step-by-step" if CoT is expected
        if "step-by-step" in self.config.get("prompt_template", ""):
            metrics["has_reasoning_steps"] = 1.0 if "\n" in output.strip() else 0.0
            
        return metrics
