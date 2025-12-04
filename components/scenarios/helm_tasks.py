from typing import List, Dict, Any
from .dataset_scenario import DatasetScenario

class HELMScenario(DatasetScenario):
    """
    Scenario for HELM style tasks, using MMLU (cais/mmlu) as a proxy.
    """
    
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        for item in dataset:
            # MMLU has question, choices, answer (index)
            choices = item['choices']
            answer_idx = item['answer']
            target = choices[answer_idx]
            
            formatted_choices = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)])
            
            tasks.append({
                "input": f"{item['question']}\nChoices:\n{formatted_choices}\nAnswer:",
                "target": target,
                "type": "knowledge",
                "choices": choices,
                "answer_idx": answer_idx
            })
        return tasks
