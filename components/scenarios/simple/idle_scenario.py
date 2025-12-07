# components/scenarios/simple/idle_scenario.py
from ..base import BaseScenario
from typing import List, Dict, Any

class IdleScenario(BaseScenario):
    """Baseline measurement - keeps model loaded but doesn't run inference"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idle_duration = kwargs.get('idle_duration', 10)
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        return []
    
    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        return {}
