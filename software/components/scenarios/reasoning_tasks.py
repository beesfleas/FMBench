from typing import List, Dict, Any
from .dataset_scenario import DatasetScenario
from .scenarios import Scenario

class ReasoningScenario(Scenario):
    """
    Simple reasoning scenario for basic tests.
    """
    def get_tasks(self) -> List[Dict[str, Any]]:
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

class ARCScenario(DatasetScenario):
    """
    Scenario for AI2 ARC (Abstraction and Reasoning Corpus) tasks.
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        for item in dataset:
            # ARC has question, choices (text/label), answerKey
            question = item['question']
            choices = item['choices']
            answer_key = item['answerKey']
            
            # Format choices
            formatted_choices = []
            target = ""
            
            for label, text in zip(choices['label'], choices['text']):
                formatted_choices.append(f"{label}. {text}")
                if label == answer_key:
                    target = text # Or keep label if we want to evaluate label match
            
            choices_str = "\n".join(formatted_choices)
            
            tasks.append({
                "input": f"Question: {question}\nChoices:\n{choices_str}\nAnswer:",
                "target": answer_key, # Evaluating against the label (A, B, C, D) is usually standard for ARC
                "type": "reasoning",
                "choices": choices,
                "answer_key": answer_key
            })
        return tasks
