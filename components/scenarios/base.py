# components/scenarios/base.py
# components/scenarios/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseScenario(ABC):
    """Base class for benchmark scenarios"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.name = kwargs.get('name', self.__class__.__name__)
        self.tasks = []
    
    def load_tasks(self):
        """Load tasks for the scenario"""
        self.tasks = self.get_tasks()

    @abstractmethod
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get list of tasks for this scenario"""
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, task: Dict[str, Any], model_output: str) -> Dict[str, Any]:
        """Evaluate model output for a given task"""
        raise NotImplementedError