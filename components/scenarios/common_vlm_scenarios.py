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



def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Levenshtein distance between two strings using DP.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

class OCRScenario(DatasetScenario):
    """
    Scenario for OCR/DocVQA tasks.
    Evaluates using ANLS (Average Normalized Levenshtein Similarity).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        image_key = self.config.get("image_key", "image")
        question_key = self.config.get("question_key", "question")
        target_key = self.config.get("target_key", "answers")
        
        # Get prompt template
        prompt_template = self.config.get("prompt_template", "Question: {question}\nAnswer:")

        for item in dataset:
            image = item.get(image_key)
            question = item.get(question_key)
            answers = item.get(target_key)
            
            # Handle possible complex nesting if needed (similar to VQA)
            # Most DocVQA datasets on HF have 'answers' as a list of strings
            
            if image is None or question is None:
                continue
                
            # If answers is a list of strings, that's what we want.
            # If it's a string, wrap it.
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                # Try to extract if it's some other object
                continue

            # Format prompt
            prompt_text = prompt_template.replace("{question}", question)
            
            tasks.append({
                "image": image,
                "input": prompt_text,
                "prompt": prompt_text,
                "target": answers, # List of valid answers
                "type": "docvqa"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute ANLS score.
        ANLS = 1 - NL if NL < 0.5 else 0
        NL = Levenshtein(pred, gt) / max(len(pred), len(gt))
        Score is max over all ground truths.
        
        For verbose outputs, also checks for containment.
        """
        metrics = {"anls": 0.0, "accuracy": 0.0}
        
        output_norm = output.lower().strip()
        
        valid_answers = target
        if isinstance(target, str):
            valid_answers = [target]
        
        max_score = 0.0
        
        for answer in valid_answers:
            if not isinstance(answer, str):
                continue
                
            ans_norm = answer.lower().strip()
            
            # Exact match check for accuracy metric (auxiliary)
            if output_norm == ans_norm:
                metrics["accuracy"] = 1.0
                max_score = 1.0
                continue
            
            # Containment check - if answer is in output, give full score
            if ans_norm in output_norm:
                metrics["accuracy"] = 1.0
                score = 1.0  # Full ANLS for containment
            else:
                # ANLS Calculation on full strings
                if not output_norm and not ans_norm:
                    score = 1.0
                elif not output_norm or not ans_norm:
                    score = 0.0
                else:
                    dist = levenshtein_distance(output_norm, ans_norm)
                    max_len = max(len(output_norm), len(ans_norm))
                    nl = dist / max_len
                    score = 1.0 - nl if nl < 0.5 else 0.0
            
            if score > max_score:
                max_score = score
        
        metrics["anls"] = max_score
        
        return metrics


class CountBenchQAScenario(DatasetScenario):
    """
    Scenario for CountBenchQA - Visual counting QA benchmark.
    Evaluates VLMs on counting objects in images.
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        import re
        tasks = []
        image_key = self.config.get("image_key", "image")
        question_key = self.config.get("question_key", "question")
        target_key = self.config.get("target_key", "answer")
        
        # Get prompt template
        prompt_template = self.config.get("prompt_template", "Question: {question}\nAnswer:")

        for item in dataset:
            image = item.get(image_key)
            question = item.get(question_key)
            answer = item.get(target_key)
            
            if image is None or question is None:
                continue
            
            # Normalize answer to string (it may be int or string)
            if answer is not None:
                answer = str(answer).strip()
            
            # Format prompt
            prompt_text = prompt_template.replace("{question}", question)
            
            tasks.append({
                "image": image,
                "input": prompt_text,
                "prompt": prompt_text,
                "target": answer,
                "type": "countbenchqa"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute metrics for CountBenchQA.
        Extracts numbers from model output and compares to target count.
        """
        import re
        metrics = {"accuracy": 0.0}
        
        if target is None:
            return metrics
        
        # Normalize target to string
        target_str = str(target).strip()
        
        # Try to extract target number
        try:
            target_num = int(target_str)
        except ValueError:
            # Target is not a valid integer
            return metrics
        
        # Normalize output
        output_norm = output.lower().strip()
        
        # Strategy 1: Check if output contains the exact number
        # Look for the number as a standalone word/value
        number_pattern = r'\b' + str(target_num) + r'\b'
        if re.search(number_pattern, output_norm):
            metrics["accuracy"] = 1.0
            return metrics
        
        # Strategy 2: Extract all numbers from output and check if target is among them
        # This handles cases like "There are 5 cats in the image."
        found_numbers = re.findall(r'\b(\d+)\b', output_norm)
        if found_numbers:
            for num_str in found_numbers:
                try:
                    if int(num_str) == target_num:
                        metrics["accuracy"] = 1.0
                        return metrics
                except ValueError:
                    continue
        
        # Strategy 3: Handle word numbers (one, two, three, etc.)
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10
        }
        for word, num in word_to_num.items():
            if word in output_norm and num == target_num:
                metrics["accuracy"] = 1.0
                return metrics
        
        return metrics
