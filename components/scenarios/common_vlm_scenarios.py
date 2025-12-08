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
        
        If slo_threshold is set in config, also computes slo_violation:
        1.0 if |predicted - target| > slo_threshold, else 0.0.
        """
        import re
        metrics = {"accuracy": 0.0}
        
        # Get SLO threshold from config (optional)
        slo_threshold = self.config.get("slo_threshold", None)
        
        if target is None:
            if slo_threshold is not None:
                metrics["slo_violation"] = 1.0  # Can't evaluate, treat as violation
            return metrics
        
        # Normalize target to string
        target_str = str(target).strip()
        
        # Try to extract target number
        try:
            target_num = int(target_str)
        except ValueError:
            # Target is not a valid integer
            if slo_threshold is not None:
                metrics["slo_violation"] = 1.0  # Can't evaluate, treat as violation
            return metrics
        
        # Normalize output
        output_norm = output.lower().strip()
        
        # Extract predicted number from output
        predicted_num = None
        
        # Strategy 1: Check if output contains the exact target number
        # Look for the number as a standalone word/value
        number_pattern = r'\b' + str(target_num) + r'\b'
        if re.search(number_pattern, output_norm):
            metrics["accuracy"] = 1.0
            predicted_num = target_num
        
        # Strategy 2: Extract all numbers from output
        # This handles cases like "There are 5 cats in the image."
        if predicted_num is None:
            found_numbers = re.findall(r'\b(\d+)\b', output_norm)
            if found_numbers:
                for num_str in found_numbers:
                    try:
                        num_val = int(num_str)
                        if num_val == target_num:
                            metrics["accuracy"] = 1.0
                            predicted_num = num_val
                            break
                        # Store first found number as fallback for SLO check
                        if predicted_num is None:
                            predicted_num = num_val
                    except ValueError:
                        continue
        
        # Strategy 3: Handle word numbers (one, two, three, etc.)
        if metrics["accuracy"] == 0.0:
            word_to_num = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10
            }
            for word, num in word_to_num.items():
                if word in output_norm:
                    if num == target_num:
                        metrics["accuracy"] = 1.0
                    # Store word number as predicted if not yet found
                    if predicted_num is None:
                        predicted_num = num
                    break
        
        # Compute SLO violation if threshold is set
        if slo_threshold is not None:
            if predicted_num is None:
                # Could not extract any number from output
                metrics["slo_violation"] = 1.0
            else:
                diff = abs(predicted_num - target_num)
                metrics["slo_violation"] = 1.0 if diff > slo_threshold else 0.0
        
        return metrics


class GTSRBScenario(DatasetScenario):
    """
    Scenario for GTSRB (German Traffic Sign Recognition Benchmark).
    Evaluates VLMs on classifying traffic signs into 43 categories.
    """
    # Default traffic sign class names (GTSRB has 43 classes)
    DEFAULT_LABEL_MAP = {
        0: "Speed limit 20",
        1: "Speed limit 30",
        2: "Speed limit 50",
        3: "Speed limit 60",
        4: "Speed limit 70",
        5: "Speed limit 80",
        6: "End of speed limit 80",
        7: "Speed limit 100",
        8: "Speed limit 120",
        9: "No passing",
        10: "No passing for vehicles over 3.5t",
        11: "Right-of-way at next intersection",
        12: "Priority road",
        13: "Yield",
        14: "Stop",
        15: "No vehicles",
        16: "Vehicles over 3.5t prohibited",
        17: "No entry",
        18: "General caution",
        19: "Dangerous curve left",
        20: "Dangerous curve right",
        21: "Double curve",
        22: "Bumpy road",
        23: "Slippery road",
        24: "Road narrows on right",
        25: "Road work",
        26: "Traffic signals",
        27: "Pedestrians",
        28: "Children crossing",
        29: "Bicycles crossing",
        30: "Beware of ice/snow",
        31: "Wild animals crossing",
        32: "End of all speed and passing limits",
        33: "Turn right ahead",
        34: "Turn left ahead",
        35: "Ahead only",
        36: "Go straight or right",
        37: "Go straight or left",
        38: "Keep right",
        39: "Keep left",
        40: "Roundabout mandatory",
        41: "End of no passing",
        42: "End of no passing for vehicles over 3.5t",
    }

    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        image_key = self.config.get("image_key", "image")
        target_key = self.config.get("target_key", "label")
        
        # Get label map from config or use default
        label_map = self.config.get("label_map", self.DEFAULT_LABEL_MAP)
        # Convert string keys to int if necessary
        if label_map and isinstance(next(iter(label_map.keys())), str):
            label_map = {int(k): v for k, v in label_map.items()}
        
        # Get prompt template
        prompt_template = self.config.get(
            "prompt_template", 
            "What type of traffic sign is shown in this image? Answer with only the sign name."
        )
        
        # Build list of class names for the prompt if needed
        class_names = list(label_map.values()) if label_map else []

        for item in dataset:
            image = item.get(image_key)
            label = item.get(target_key)
            
            if image is None or label is None:
                continue
            
            # Convert label to class name
            if isinstance(label, int) and label_map:
                target_name = label_map.get(label, f"Class {label}")
            else:
                target_name = str(label)
            
            tasks.append({
                "image": image,
                "input": prompt_template,
                "prompt": prompt_template,
                "target": target_name,
                "target_label": label,  # Keep original label for reference
                "type": "gtsrb",
                "class_names": class_names,  # Available classes
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute accuracy for GTSRB classification.
        Checks if the target class name is present in the model output.
        """
        metrics = {"accuracy": 0.0}
        
        if target is None:
            return metrics
        
        # Normalize strings for comparison
        output_norm = output.lower().strip()
        target_norm = str(target).lower().strip()
        
        # Exact match
        if target_norm == output_norm:
            metrics["accuracy"] = 1.0
            return metrics
        
        # Containment match - target in output
        if target_norm in output_norm:
            metrics["accuracy"] = 1.0
            return metrics
        
        # Check for key terms match (e.g., "speed limit 30" contains "30")
        # Extract numbers from both
        import re
        target_numbers = re.findall(r'\d+', target_norm)
        output_numbers = re.findall(r'\d+', output_norm)
        
        # Also check for key action words
        target_words = set(target_norm.split())
        output_words = set(output_norm.split())
        
        # Check if critical words overlap (e.g., "stop", "yield", "speed", "limit")
        critical_words = {"stop", "yield", "speed", "limit", "no", "passing", "entry", 
                         "right", "left", "ahead", "roundabout", "pedestrians", "children",
                         "work", "caution", "priority", "road"}
        
        target_critical = target_words & critical_words
        output_critical = output_words & critical_words
        
        # If target has numbers, check if they appear in output
        if target_numbers:
            if set(target_numbers) <= set(output_numbers) and target_critical <= output_critical:
                metrics["accuracy"] = 1.0
                return metrics
        else:
            # No numbers, check if critical words match
            if target_critical and target_critical <= output_critical:
                metrics["accuracy"] = 1.0
        
        return metrics
