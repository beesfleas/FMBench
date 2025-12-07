from typing import List, Dict, Any
from .dataset_scenario import DatasetScenario
import logging

logger = logging.getLogger(__name__)

class VQAScenario(DatasetScenario):
    """
    Scenario for Visual Question Answering (e.g., VQAv2).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        image_key = self.config.get("image_key", "image")
        question_key = self.config.get("question_key", "question")
        target_key = self.config.get("target_key", "multiple_choice_answer")
        
        # Get prompt template
        prompt_template = self.config.get("prompt_template", "Question: {question} Answer:")

        for item in dataset:
            image = item.get(image_key)
            question_obj = item.get(question_key)
            answer_obj = item.get(target_key)
            
            # Handle Image List (e.g., The Cauldron)
            if isinstance(image, list) and len(image) > 0:
                image = image[0]
                
            question = None
            answer = None

            # Handle The Cauldron style (list of dicts in question_key/target_key)
            # logic: check if question_key points to a list of dicts
            if isinstance(question_obj, list) and len(question_obj) > 0 and isinstance(question_obj[0], dict):
                first_turn = question_obj[0]
                question = first_turn.get("user", first_turn.get("question")) # Try 'user' or 'question'
                # If target_key is same as question_key, extract answer from here too
                if target_key == question_key:
                    answer = first_turn.get("assistant", first_turn.get("answer"))
            
            # Fallback/Standard String Handling
            if question is None and isinstance(question_obj, str):
                question = question_obj
            
            if answer is None:
                if isinstance(answer_obj, str):
                    answer = answer_obj
                elif isinstance(answer_obj, list) and len(answer_obj) > 0 and isinstance(answer_obj[0], str):
                    # List of strings (common in VQA for multiple reference answers)
                    answer = answer_obj[0] 

            if image is None or question is None:
                continue

            # Format prompt
            prompt_text = prompt_template.replace("{question}", question)
            
            tasks.append({
                "image": image,
                "input": prompt_text,
                "prompt": prompt_text,
                "target": answer,
                "type": "vqa"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute metrics for VQA.
        Usually exact match or loose containment.
        """
        metrics = {"accuracy": 0.0}
        if not isinstance(target, str):
            # Sometimes target is a list of answers
            if isinstance(target, list):
                # Check if output matches any of the valid answers
                # But for VQAv2 standard "multiple_choice_answer" is a single string usually in HF dataset
                pass
            return metrics
            
        # Normalize
        output_norm = output.lower().strip()
        target_norm = target.lower().strip()
        
        # Basic containment or exact match
        # VQA evaluation can be complex (VQA score), but we'll start with simple exact match
        if target_norm == output_norm:
            metrics["accuracy"] = 1.0
        elif target_norm in output_norm:
             # Loose match
            metrics["accuracy"] = 1.0
            
        return metrics
