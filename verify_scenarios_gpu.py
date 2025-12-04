import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from components.scenarios import ReasoningScenario, HELMScenario, NLPScenario

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting verification...")
    
    # Load a small model
    model_name = "gpt2"
    logger.info(f"Loading model: {model_name}")
    
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using device: {device} (0=GPU, -1=CPU)")
    
    try:
        # Create a pipeline which is a callable
        generator = pipeline('text-generation', model=model_name, device=device, max_new_tokens=20)
        
        # Test ReasoningScenario
        logger.info("\nTesting ReasoningScenario...")
        reasoning_config = {"name": "ReasoningTest", "prompt_template": "Question: {input}\nAnswer:"}
        reasoning_scenario = ReasoningScenario(reasoning_config, model=generator)
        results = reasoning_scenario.evaluate()
        logger.info(f"Reasoning Metrics: {results['metrics']}")
        for item in results['details']:
            logger.info(f"Input: {item['input']}")
            logger.info(f"Target: {item['target']}")
            logger.info(f"Output: {item['output']}")
            logger.info("-" * 20)
        
        # Test HELMScenario
        logger.info("\nTesting HELMScenario...")
        helm_config = {"name": "HELMTest", "prompt_template": "{input}"}
        helm_scenario = HELMScenario(helm_config, model=generator)
        results = helm_scenario.evaluate()
        logger.info(f"HELM Metrics: {results['metrics']}")
        for item in results['details']:
            logger.info(f"Input: {item['input']}")
            logger.info(f"Target: {item['target']}")
            logger.info(f"Output: {item['output']}")
            logger.info("-" * 20)
        
        # Test NLPScenario
        logger.info("\nTesting NLPScenario...")
        nlp_config = {"name": "NLPTest", "prompt_template": "{input}"}
        nlp_scenario = NLPScenario(nlp_config, model=generator)
        results = nlp_scenario.evaluate()
        logger.info(f"NLP Metrics: {results['metrics']}")
        for item in results['details']:
            logger.info(f"Input: {item['input']}")
            logger.info(f"Target: {item['target']}")
            logger.info(f"Output: {item['output']}")
            logger.info("-" * 20)
        
        logger.info("\nVerification completed successfully!")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
