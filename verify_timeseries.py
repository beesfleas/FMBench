
import sys
import os
import logging
from omegaconf import OmegaConf
import torch
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from components.models.huggingface_timeseries import HuggingFaceTimeSeriesLoader
from components.scenarios.common_timeseries_scenarios import M3Scenario, FevBenchScenario

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def verify():
    # 1. Configs
    models_to_test = [
        {"model_id": "ibm-granite/granite-timeseries-patchtst", "name": "PatchTST"}
    ]
    
    scenario_config = OmegaConf.create({
        "dataset_name": "autogluon/fev_datasets",
        "subset": "ETT_1H",
        "prediction_length": 24, # Standard for ETT
        "limit": 2
    })

    # 2. Load Scenario
    log.info("Loading Scenario...")
    scenario = FevBenchScenario(**scenario_config) # Unpack for kwargs
    tasks = scenario.get_tasks()
    log.info(f"Loaded {len(tasks)} tasks.")
    if not tasks:
        log.error("No tasks loaded.")
        return

    # 3. Test each model
    for model_info in models_to_test:
        log.info(f"\n{'='*20}\nTesting {model_info['name']}\n{'='*20}")
        model_config = OmegaConf.create({
            "model_id": model_info["model_id"],
            "prediction_length": 8,
            "num_samples": 5,
            "allow_mps_fallback": True
        })

        loader = HuggingFaceTimeSeriesLoader()
        try:
            loader.load_model(model_config)
        except Exception as e:
            log.error(f"Failed to load {model_info['name']}: {e}")
            continue

        # Run tasks
        for i, task in enumerate(tasks[:1]): # Run just 1 for speed
            log.info(f"Task {i}: {task['task_id']}")
            
            # Prepare input
            context = task['time_series_data']
            
            try:
                output = loader.predict(time_series_data=context)
                log.info(f"Prediction type: {type(output)}")
                if isinstance(output, dict) and "forecast" in output:
                    log.info(f"Forecast sample: {output['forecast'][:3]}...")
                
                # Evaluate
                metrics = scenario.evaluate(task, output)
                log.info(f"Metrics: {metrics}")
            except Exception as e:
                log.error(f"Inference failed for {model_info['name']}: {e}")
        
        # Cleanup
        if hasattr(loader, 'unload_model'):
            loader.unload_model()

    log.info("Verification complete.")

if __name__ == "__main__":
    verify()
