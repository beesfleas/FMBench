from typing import List, Dict, Any
from .dataset_scenario import DatasetScenario
import logging
import re
import torch

logger = logging.getLogger(__name__)

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
        """
        Compute metrics for Summarization.
        Calculates:
        - BLEU (sacrebleu)
        - METEOR (nltk)
        - BERT Score (bert_score)
        """
        metrics = {"accuracy": 0.0}
        if not isinstance(target, str):
            return metrics

        # 1. BLEU Score
        try:
            import sacrebleu
            bleu = sacrebleu.sentence_bleu(output, [target])
            metrics["bleu"] = bleu.score
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")

        # 2. METEOR Score
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            # Ensure wordnet is downloaded
            try:
                 nltk.data.find('corpora/wordnet')
            except LookupError:
                 logger.info("Downloading nltk wordnet...")
                 nltk.download('wordnet', quiet=True)
            
            # METEOR expects tokenized list
            reference = target.split()
            hypothesis = output.split()
            metrics["meteor"] = meteor_score([reference], hypothesis)
        except ImportError as e:
            logger.warning(f"NLTK not installed or import failed: {e}")
        except Exception as e:
            logger.error(f"Error calculating METEOR: {e}")

        # 3. BERT Score
        if self.config.get("use_expensive_metrics", True):
            try:
                import bert_score
                # BERTScore calculation
                P, R, F1 = bert_score.score([output], [target], lang="en", verbose=False)
                metrics["bert_score"] = F1.mean().item()
            except ImportError as e:
                logger.warning(f"bert_score not installed or import failed: {e}")
            except Exception as e:
                logger.error(f"Error calculating BERT Score: {e}")
            
        return metrics

class NERScenario(DatasetScenario):
    """
    Scenario for Named Entity Recognition (e.g., CoNLL2003).
    """
    def process_dataset(self, dataset) -> List[Dict[str, Any]]:
        tasks = []
        input_key = self.config.get("input_key", "tokens")
        target_key = self.config.get("target_key", "ner_tags")
        
        # Tags for CoNLL2003 (Example map, can be config driven)
        # 0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC, 7: B-MISC, 8: I-MISC
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

            # Join tokens for input
            sentence = " ".join(tokens)
            
            # Format expected output as list of (Entity, Type)
            # This is a simplified representation for generation measurement
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
        """
        Compute metrics for NER.
        Extracts entities from target and checks if they are in the output.
        """
        if not isinstance(target, str):
            logger.warning(f"Target is not string: {target}")
            return {"accuracy": 0.0}

        # Target is "Entity (Type), Entity (Type)"
        # We split by comma to get individual entities
        expected_entities = [e.strip() for e in target.split(",")]
        
        # If no entities expected, and output is "None" or similar
        if target == "None":
            return {"accuracy": 1.0 if "none" in output.lower() or not output.strip() else 0.0}

        # Check recall-like accuracy: fraction of expected entities found in output
        found_count = 0
        output_lower = output.lower()
        
        for entity_str in expected_entities:
            # entity_str is "India (LOC)" -> we check if "India" is in output 
            # (or strict check "India (LOC)" depending on requirement)
            # Relaxing to just Entity name might be safer if model doesn't output type perfectly
            
            # Let's try to parse the entity name out of "Name (Type)"
            # Assumes format "Name (Type)"
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
        """
        Compute metrics for Text Classification.
        Robust to prompt repetition, verbose output, and markdown formatting.
        """
        if not isinstance(target, str):
            return {"accuracy": 0.0}
            
        # 1. Normalize strings
        output_lower = output.lower()
        target_lower = target.lower().strip()
        
        # 2. Define aliases for common labels (AG News mostly)
        # We can map standard long forms to the dataset labels
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
        
        # 3. Clean output of Markdown (bold/italic) roughly
        # Remove chars like * and _
        cleaned_output = re.sub(r"[\*\_]", " ", output_lower)

        logger.info(f"Classification Grading - Target: '{target}' (Norm: '{target_norm}') | Output Preview: '{cleaned_output[:50]}...'")

        
        # 4. Check for target existence
        # We prefer the label to be at the START of the response, or closely following "Category:"
        # But sometimes models assume they are continuing text.
        
        # Heuristic A: Check if strict target is present as a standalone word/phrase
        # escaping regex specials in target
        target_regex = re.escape(target_norm)
        
        # Look for target as a whole word
        # We search in the first 200 chars (generous header) OR the last 100 chars (summary) to avoid finding it in a long prompt repetition?
        # Actually user logs show prompt is at the END sometimes? No, user logs show:
        # "Output: Business\n\nThe article..." -> Start
        # "Output: ... Category: World" -> End
        
        # Let's search entire string but check bounds? No, simple containment for classification is usually fine 
        # UNLESS the prompt contains the label.
        # But here valid labels are single words like "Business". "Business" is a common word.
        # So we MUST be careful.
        
        # Strategy:
        # - If output starts with target (ignoring non-alnum), it's a match.
        # - If output contains "Category: <target>", match.
        # - If output is short (< 50 chars), and contains target, match.
        
        # Strip leading non-alphanum
        start_clean = re.sub(r"^[^a-z0-9]+", "", cleaned_output).strip()
        
        if start_clean.startswith(target_norm):
            logger.info("Match found at start.")
            return {"accuracy": 1.0}
            
        if f"category: {target_norm}" in cleaned_output:
            logger.info("Match found via 'Category:' prefix.")
            return {"accuracy": 1.0}
            
        # Check aliases at start
        for alias, mapped in aliases.items():
            if mapped == target_norm and start_clean.startswith(alias):
                logger.info(f"Match found via alias '{alias}' at start.")
                return {"accuracy": 1.0}

        # Last resort: If the output contains the label and is not super long (likely just the answer)
        # OR if we can find the label on a line by itself
        lines = [l.strip() for l in cleaned_output.split('\n') if l.strip()]
        for line in lines[:3]: # Check first 3 lines
            # If line is exactly the target (plus maybe punctuation)
            line_clean = re.sub(r"[^a-z0-9\s]", "", line).strip()
            if line_clean == target_norm:
                return {"accuracy": 1.0}
            # Check aliases
            for alias, mapped in aliases.items():
                if mapped == target_norm and line_clean == alias:
                    logger.info("Match found via alias on line.")
                    return {"accuracy": 1.0}

        logger.info("No match found.")
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
        """
        Compute metrics for Translation.
        Calculates:
        - BLEU (sacrebleu)
        - COMET (unbabel-comet) - lazy loaded
        - Accuracy (basic containment)
        """
        metrics = {"accuracy": 0.0}
        if not isinstance(target, str):
            return metrics
            
        # Normalize
        output_norm = output.lower().strip()
        target_norm = target.lower().strip()
        
        # 1. Basic Containment Accuracy
        if target_norm in output_norm:
            metrics["accuracy"] = 1.0
            
        # 2. BLEU Score
        try:
            import sacrebleu
            # sacrebleu expects list of references for each hypothesis
            # output is single hypothesis, target is single reference
            # For robustness, we treat output as a single sentence for now.
            bleu = sacrebleu.sentence_bleu(output, [target])
            metrics["bleu"] = bleu.score
        except ImportError:
            logger.warning("sacrebleu not installed, skipping BLEU")
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")

        # 3. COMET Score
        # COMET requires the source sentence as well
        source = task.get("source")
        if source and self.config.get("use_expensive_metrics", True):
            try:
                # Lazy import and model loading
                if not hasattr(self, "_comet_model"):
                    from comet import download_model, load_from_checkpoint
                    
                    # Use a lightweight or standard COMET model
                    model_path = download_model("Unbabel/wmt22-comet-da")
                    self._comet_model = load_from_checkpoint(model_path)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        self._comet_model = self._comet_model.cuda()
                        
                # Prepare data for COMET
                data = [{"src": source, "mt": output, "ref": target}]
                # Predict
                model_output = self._comet_model.predict(data, batch_size=1, gpus=1 if torch.cuda.is_available() else 0)
                metrics["comet"] = model_output.scores[0]
                
            except ImportError:
                logger.warning("unbabel-comet not installed or torch missing, skipping COMET")
            except Exception as e:
                logger.error(f"Error calculating COMET: {e}")
                
        return metrics
