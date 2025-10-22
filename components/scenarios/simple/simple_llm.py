# components/scenarios/simple/simple_llm.py
from ..base import BaseScenario
from typing import List, Dict, Any

class SimpleLLMScenario(BaseScenario):
    """Ultra-simple LLM test - just verify model runs"""
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "prompt": "Hello",
                "expected_output": "any response",
                "task_type": "basic_test"
            }
        ]
    
    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        return {
            "task_type": "basic_test",
            "model_output": model_output,
            "has_response": len(model_output.strip()) > 0,
            "success": len(model_output.strip()) > 0
        }