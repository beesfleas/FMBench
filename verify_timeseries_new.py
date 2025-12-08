import logging
import sys
import yaml
from omegaconf import OmegaConf
from components.scenarios.common_timeseries_scenarios import GiftEvalScenario, FevBenchScenario
from components.models.huggingface_timeseries import HuggingFaceTimeSeriesLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def verify_scenario(scenario_cls, config_path):
    print(f"\n--- Verifying {config_path} ---", flush=True)
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = OmegaConf.create(config_dict)
        
        # Initialize scenario
        scenario = scenario_cls(**config)
        
        # Get tasks
        tasks = scenario.get_tasks()
        print(f"Generated {len(tasks)} tasks.", flush=True)
        
        if not tasks:
            print("No tasks generated!", flush=True)
            return None
            
        # Inspect first task
        first_task = tasks[0]
        print(f"First task keys: {first_task.keys()}", flush=True)
        print(f"Time series data type: {type(first_task['time_series_data'])}", flush=True)
        print(f"Context length: {len(first_task['time_series_data'])}", flush=True)
        print(f"Ground truth length: {len(first_task['ground_truth'])}", flush=True)
        
        return tasks
    except Exception as e:
        print(f"FAILED to verify {config_path}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def test_model_inference(tasks):
    log.info("--- Testing Model Inference with simple-timeseries (or chronos if avail) ---")
    
    # Using chronos-t5-small from existing config structure might be complex without full runner.
    # Instantiate HuggingFaceTimeseriesModel directly if possible.
    
    # Need model config
    model_config_path = "conf/model/chronos-t5-small.yaml"
    with open(model_config_path, 'r') as f:
         model_cfg_dict = yaml.safe_load(f)
    model_cfg = OmegaConf.create(model_cfg_dict)
    
    try:
        log.info(f"Initializing model: {model_cfg.model_id}")
        # Initialize loader
        loader = HuggingFaceTimeSeriesLoader()
        loader.load_model(model_cfg)
        model = loader # The loader exposes predict method
        
        # Taking a few tasks
        test_tasks = tasks[:3]
        
        # Prepare prompts (model usually expects just the context)
        # HuggingFaceTimeseriesModel.predict takes a list of contexts usually?
        # Let's check the signature or usage in existing code if needed.
        # But for now, let's try calling predict on the prompt.
        
        prompts = [t["time_series_data"] for t in test_tasks]
        
        log.info("Running inference...")
        responses = model.predict(prompts)
        
        log.info(f"Got {len(responses)} responses.")
        if responses:
            log.info(f"First response: {responses[0]}")
            
    except Exception as e:
        log.error(f"Inference failed: {e}")

if __name__ == "__main__":
    # 1. Verify GIFT-EVAL
    gift_tasks = verify_scenario(GiftEvalScenario, "conf/scenario/gift_eval.yaml")
    
    # 2. Verify FEV-Bench
    fev_tasks = verify_scenario(FevBenchScenario, "conf/scenario/fev_bench.yaml")
    
    # 3. Quick Inference Test (using GIFT-EVAL tasks as they are standard)
    if gift_tasks:
        test_model_inference(gift_tasks)
