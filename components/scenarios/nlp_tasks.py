from typing import List, Dict, Any
from .scenarios import Scenario

class NLPScenario(Scenario):
    """
    Scenario for standard NLP tasks (Summarization, Sentiment, etc.)
    """
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        return [
            {
                "input": "Summarize: The quick brown fox jumps over the lazy dog.",
                "target": "A fox jumps over a dog.",
                "type": "summarization"
            },
            {
                "input": "Sentiment: I love this movie!",
                "target": "Positive",
                "type": "sentiment"
            }
        ]
