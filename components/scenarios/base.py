# components/scenarios/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseScenario(ABC):
    """Base class for benchmark scenarios"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
    
    @abstractmethod
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get list of tasks for this scenario"""
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        """Evaluate model output for a given task"""
        raise NotImplementedError