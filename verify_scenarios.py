import sys
import os
import logging
from components.scenarios.common_nlp_scenarios import (
    SentimentScenario,
    SummarizationScenario,
    NERScenario,
    TextClassificationScenario,
    TranslationScenario
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_sentiment():
    logger.info("Verifying SentimentScenario...")
    config = {
        "dataset_name": "imdb",
        "split": "test",
        "num_samples": 5,
        "input_key": "text",
        "target_key": "label",
        "label_map": {0: "Negative", 1: "Positive"}
    }
    scenario = SentimentScenario(config)
    tasks = scenario.get_tasks()
    if tasks:
        logger.info(f"Loaded {len(tasks)} tasks.")
        logger.info(f"Sample task: {tasks[0]}")
    else:
        logger.error("No tasks loaded for SentimentScenario")

def verify_summarization():
    logger.info("Verifying SummarizationScenario...")
    config = {
        "dataset_name": "cnn_dailymail",
        "dataset_config": "3.0.0",
        "split": "test",
        "num_samples": 5,
        "input_key": "article",
        "target_key": "highlights"
    }
    scenario = SummarizationScenario(config)
    tasks = scenario.get_tasks()
    if tasks:
        logger.info(f"Loaded {len(tasks)} tasks.")
        logger.info(f"Sample task: {tasks[0]}")
    else:
        logger.error("No tasks loaded for SummarizationScenario")

def verify_ner():
    logger.info("Verifying NERScenario...")
    config = {
        "dataset_name": "conll2003",
        "split": "test",
        "num_samples": 5,
        "input_key": "tokens",
        "target_key": "ner_tags"
    }
    scenario = NERScenario(config)
    tasks = scenario.get_tasks()
    if tasks:
        logger.info(f"Loaded {len(tasks)} tasks.")
        logger.info(f"Sample task: {tasks[0]}")
    else:
        logger.error("No tasks loaded for NERScenario")

def verify_classification():
    logger.info("Verifying TextClassificationScenario...")
    config = {
        "dataset_name": "ag_news",
        "split": "test",
        "num_samples": 5,
        "input_key": "text",
        "target_key": "label"
    }
    scenario = TextClassificationScenario(config)
    tasks = scenario.get_tasks()
    if tasks:
        logger.info(f"Loaded {len(tasks)} tasks.")
        logger.info(f"Sample task: {tasks[0]}")
    else:
        logger.error("No tasks loaded for TextClassificationScenario")

def verify_translation():
    logger.info("Verifying TranslationScenario...")
    config = {
        "dataset_name": "wmt16",
        "dataset_config": "de-en",
        "split": "test",
        "num_samples": 5,
        "input_key": "translation",
        "source_language": "de",
        "target_language": "en"
    }
    scenario = TranslationScenario(config)
    tasks = scenario.get_tasks()
    if tasks:
        logger.info(f"Loaded {len(tasks)} tasks.")
        logger.info(f"Sample task: {tasks[0]}")
    else:
        logger.error("No tasks loaded for TranslationScenario")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, help="Scenario to verify (sentiment, summarization, ner, classification, translation)")
    parser.add_argument("--all", action="store_true", help="Verify all scenarios")
    args = parser.parse_args()

    scenarios = {
        "sentiment": verify_sentiment,
        "summarization": verify_summarization,
        "ner": verify_ner,
        "classification": verify_classification,
        "translation": verify_translation
    }

    if args.all:
        for name, func in scenarios.items():
            print(f"\n--- {name} ---")
            try:
                func()
            except Exception as e:
                logger.error(f"Error validating {name}: {e}")
    elif args.scenario in scenarios:
        scenarios[args.scenario]()
    else:
        print("Please specify --scenario <name> or --all")
        print("Available scenarios:", list(scenarios.keys()))
