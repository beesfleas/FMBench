from typing import List, Dict, Any
import logging
import re

import torch
from .dataset_scenario import DatasetScenario

# Optional dependencies for metrics computation
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    sacrebleu = None
    SACREBLEU_AVAILABLE = False

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    meteor_score = None
    NLTK_AVAILABLE = False

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    bert_score = None
    BERT_SCORE_AVAILABLE = False

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    download_model = None
    load_from_checkpoint = None
    COMET_AVAILABLE = False

log = logging.getLogger(__name__)

class SentimentScenario(DatasetScenario):
    """
    Scenario for Sentiment Analysis (e.g., IMDB, SST2).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        # Label mapping can be customized via config, defaulting to simple binary for IMDB/SST2 if not provided
        label_map = self.config.get("label_map", {0: "Negative", 1: "Positive"})
        
        input_key = self.config.get("input_key", "text")
        target_key = self.config.get("target_key", "label")
        
        # Get prompt template, default to simple format if not provided
        prompt_template = self.config.get("prompt_template", "Sentiment: {input}")

        for item in dataset:
            text = item.get(input_key)
            label_idx = item.get(target_key)
            
            # Skip if keys are missing
            if text is None or label_idx is None:
                continue

            target_label = label_map.get(label_idx, str(label_idx))
            
            # Use template to format input
            input_text = prompt_template.replace("{input}", text)
            
            tasks.append({
                "input": input_text,
                "target": target_label,
                "type": "sentiment"
            })
        return tasks

class SummarizationScenario(DatasetScenario):
    """
    Scenario for Summarization (e.g., CNN/DailyMail, XSum).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        input_key = self.config.get("input_key", "article")
        target_key = self.config.get("target_key", "highlights")
        
        # Get prompt template
        prompt_template = self.config.get("prompt_template", "Summarize: {input}")
        
        for item in dataset:
            text = item.get(input_key)
            summary = item.get(target_key)
            
            if text is None or summary is None:
                continue
            
            # Use template
            input_text = prompt_template.replace("{input}", text)

            tasks.append({
                "input": input_text,
                "target": summary,
                "type": "summarization"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """Compute BLEU, METEOR, and BERTScore for summarization."""
        metrics = {"accuracy": 0.0}
        skipped_metrics = []
        
        if not isinstance(target, str):
            return metrics

        # 1. BLEU Score
        if SACREBLEU_AVAILABLE:
            try:
                bleu = sacrebleu.sentence_bleu(output, [target])
                metrics["bleu"] = bleu.score
            except Exception as e:
                log.error("Error calculating BLEU: %s", e)
        else:
            skipped_metrics.append("bleu")

        # 2. METEOR Score
        if NLTK_AVAILABLE:
            try:
                # Ensure wordnet is downloaded
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    log.debug("Downloading nltk wordnet...")
                    nltk.download('wordnet', quiet=True)
                
                # METEOR expects tokenized list
                reference = target.split()
                hypothesis = output.split()
                metrics["meteor"] = meteor_score([reference], hypothesis)
            except Exception as e:
                log.error("Error calculating METEOR: %s", e)
        else:
            skipped_metrics.append("meteor")

        # 3. BERT Score
        if self.config.get("use_expensive_metrics", True):
            if BERT_SCORE_AVAILABLE:
                try:
                    # BERTScore calculation
                    P, R, F1 = bert_score.score([output], [target], lang="en", verbose=False)
                    metrics["bert_score"] = F1.mean().item()
                except Exception as e:
                    log.error("Error calculating BERT Score: %s", e)
            else:
                skipped_metrics.append("bert_score")
        
        if skipped_metrics:
            metrics["_skipped_metrics"] = skipped_metrics
            
        return metrics

class NERScenario(DatasetScenario):
    """
    Scenario for Named Entity Recognition (e.g., CoNLL2003).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        input_key = self.config.get("input_key", "tokens")
        target_key = self.config.get("target_key", "ner_tags")
        
        # CoNLL2003 tag mapping (override via config "tag_map")
        default_idx_to_tag = {
            0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 
            5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
        }
        idx_to_tag = self.config.get("tag_map", default_idx_to_tag)

        for item in dataset:
            tokens = item.get(input_key)
            tags = item.get(target_key)
            
            if tokens is None or tags is None:
                continue

            sentence = " ".join(tokens)
            
            # Extract entities in "Name (Type)" format
            entities = []
            current_entity = []
            current_type = None
            
            for token, tag_idx in zip(tokens, tags):
                tag = idx_to_tag.get(tag_idx, "O")
                if tag == "O":
                    if current_entity:
                        entities.append(f"{' '.join(current_entity)} ({current_type})")
                        current_entity = []
                        current_type = None
                elif tag.startswith("B-"):
                    if current_entity:
                        entities.append(f"{' '.join(current_entity)} ({current_type})")
                    current_entity = [token]
                    current_type = tag[2:]
                elif tag.startswith("I-"):
                    if current_type == tag[2:]:
                        current_entity.append(token)
                    else:
                        # Mismatch or broken sequence
                        if current_entity:
                           entities.append(f"{' '.join(current_entity)} ({current_type})")
                        current_entity = [token]
                        current_type = tag[2:]
            
            if current_entity:
               entities.append(f"{' '.join(current_entity)} ({current_type})")
            
            target_str = ", ".join(entities) if entities else "None"

            # Enable prompt template
            prompt_template = self.config.get("prompt_template", "Extract named entities from the following text: {input}")
            input_text = prompt_template.replace("{input}", sentence)

            tasks.append({
                "input": input_text,
                "target": target_str,
                "type": "ner"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """Compute recall-based accuracy for NER."""
        if not isinstance(target, str):
            log.debug("Target is not string: %s", target)
            return {"accuracy": 0.0}

        expected_entities = [e.strip() for e in target.split(",")]
        
        if target == "None":
            return {"accuracy": 1.0 if "none" in output.lower() or not output.strip() else 0.0}

        # Check recall: fraction of expected entities found in output
        found_count = 0
        output_lower = output.lower()
        
        for entity_str in expected_entities:
            # Parse entity name from "Name (Type)" format
            if "(" in entity_str and entity_str.endswith(")"):
                entity_name = entity_str.rsplit("(", 1)[0].strip()
            else:
                entity_name = entity_str
                
            if entity_name.lower() in output_lower:
                found_count += 1
        
        accuracy = found_count / len(expected_entities) if expected_entities else 1.0
        return {"accuracy": accuracy}
    
class TextClassificationScenario(DatasetScenario):
    """
    Scenario for Text Classification (e.g., AG News).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        input_key = self.config.get("input_key", "text")
        target_key = self.config.get("target_key", "label")
        
        # AG News labels: 0: World, 1: Sports, 2: Business, 3: Sci/Tech
        default_label_map = {
            0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"
        }
        label_map = self.config.get("label_map", default_label_map)

        for item in dataset:
            text = item.get(input_key)
            label_idx = item.get(target_key)
            
            if text is None or label_idx is None:
                continue

            target_label = label_map.get(label_idx, str(label_idx))
            
            # Use template to format input
            prompt_template = self.config.get("prompt_template", "Classify the following news article into a category: {input}")
            input_text = prompt_template.replace("{input}", text)
            
            tasks.append({
                "input": input_text,
                "target": target_label,
                "type": "classification"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """Compute classification accuracy with flexible label matching."""
        if not isinstance(target, str):
            return {"accuracy": 0.0}
            
        # 1. Normalize strings
        output_lower = output.lower()
        target_lower = target.lower().strip()
        
        # Label aliases for flexible matching
        aliases = {
            "science/technology": "sci/tech",
            "science & technology": "sci/tech",
            "technology": "sci/tech",
            "science": "sci/tech",
            "sci": "sci/tech",
            "tech": "sci/tech",
            "sports": "sports",
            "business": "business",
            "world": "world"
        }
        
        # Resolve target to an alias if possible, or keep as is
        target_norm = aliases.get(target_lower, target_lower)
        
        # Remove markdown formatting
        cleaned_output = re.sub(r"[\*\_]", " ", output_lower)

        log.debug("Classification Grading - Target: '%s' (Norm: '%s') | Output: '%s...'",
                  target, target_norm, cleaned_output[:50])

        # Match strategies: start of output, "Category:" prefix, or exact line match
        start_clean = re.sub(r"^[^a-z0-9]+", "", cleaned_output).strip()
        
        if start_clean.startswith(target_norm):
            log.debug("Match found at start")
            return {"accuracy": 1.0}
            
        if f"category: {target_norm}" in cleaned_output:
            log.debug("Match found via 'Category:' prefix")
            return {"accuracy": 1.0}
            
        for alias, mapped in aliases.items():
            if mapped == target_norm and start_clean.startswith(alias):
                log.debug("Match found via alias '%s' at start", alias)
                return {"accuracy": 1.0}

        # Check first 3 lines for exact match
        lines = [l.strip() for l in cleaned_output.split('\n') if l.strip()]
        for line in lines[:3]:
            line_clean = re.sub(r"[^a-z0-9\s]", "", line).strip()
            if line_clean == target_norm:
                return {"accuracy": 1.0}
            for alias, mapped in aliases.items():
                if mapped == target_norm and line_clean == alias:
                    log.debug("Match found via alias on line")
                    return {"accuracy": 1.0}

        log.debug("No match found")
        return {"accuracy": 0.0}


class TranslationScenario(DatasetScenario):
    """
    Scenario for Translation (e.g., WMT).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        # Config for nested keys: "translation.de"
        source_config = self.config.get("source_language", "de")
        target_config = self.config.get("target_language", "en")
        item_key = self.config.get("input_key", "translation")

        # Get prompt template
        prompt_template = self.config.get("prompt_template", "Translate to English: {input}")
        
        for item in dataset:
            translation_obj = item.get(item_key)
            if not translation_obj or not isinstance(translation_obj, dict):
                continue
            
            source_text = translation_obj.get(source_config)
            target_text = translation_obj.get(target_config)
            
            if not source_text or not target_text:
                continue

            # Use template to format input
            input_text = prompt_template.replace("{input}", source_text)

            tasks.append({
                "input": input_text,
                "source": source_text, # Added for COMET
                "target": target_text,
                "type": "translation"
            })
        return tasks

    def compute_metrics(self, output: str, target: Any, task: Dict[str, Any]) -> Dict[str, float]:
        """Compute BLEU, COMET, and containment accuracy for translation."""
        metrics = {"accuracy": 0.0}
        skipped_metrics = []
        
        if not isinstance(target, str):
            return metrics
            
        # Normalize
        output_norm = output.lower().strip()
        target_norm = target.lower().strip()
        
        # 1. Basic Containment Accuracy
        if target_norm in output_norm:
            metrics["accuracy"] = 1.0
            
        # BLEU Score
        if SACREBLEU_AVAILABLE:
            try:
                bleu = sacrebleu.sentence_bleu(output, [target])
                metrics["bleu"] = bleu.score
            except Exception as e:
                log.error("Error calculating BLEU: %s", e)
        else:
            skipped_metrics.append("bleu")

        # COMET Score (requires source sentence)
        source = task.get("source")
        if source and self.config.get("use_expensive_metrics", True):
            if COMET_AVAILABLE:
                try:
                    # Lazy model loading
                    if not hasattr(self, "_comet_model"):
                        model_path = download_model("Unbabel/wmt22-comet-da")
                        self._comet_model = load_from_checkpoint(model_path)
                        
                        if torch.cuda.is_available():
                            self._comet_model = self._comet_model.cuda()
                            
                    data = [{"src": source, "mt": output, "ref": target}]
                    model_output = self._comet_model.predict(data, batch_size=1, gpus=1 if torch.cuda.is_available() else 0)
                    metrics["comet"] = model_output.scores[0]
                    
                except Exception as e:
                    log.error("Error calculating COMET: %s", e)
            else:
                skipped_metrics.append("comet")
        
        if skipped_metrics:
            metrics["_skipped_metrics"] = skipped_metrics
                
        return metrics
