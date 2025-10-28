from components.models.model_factory import get_model_loader
from omegaconf import DictConfig
import hydra
import time

def get_profiler(device_config: DictConfig):
    """
    Factory function to instantiate the correct device profiler
    based on the configuration.
    """
    profiler_class_path = device_config.profiler_class
    print(f"Instantiating profiler from: {profiler_class_path}")
    
    # Use hydra's utility to instantiate the class from its path
    # We pass the device config to the profiler's __init__
    profiler = hydra.utils.instantiate(
        {"_target_": profiler_class_path},
        config=device_config
    )
    return profiler

def run_benchmark(cfg: DictConfig):
    model_config = cfg.model
    device_config = cfg.device
    
    # Load model (Your existing logic)
    loader = get_model_loader(dict(model_config))
    print(f"Loading {model_config.model_category} model: {model_config.model_id}")
    loader.load_model(dict(model_config))
    
    # 2. Instantiate the Device Profiler using the factory
    profiler = get_profiler(device_config)

    # 3. Run the test (scenario or basic) inside the profiler's context
    print("\n--- Starting Benchmark Run ---")
    start_time = time.perf_counter()
    
    device_metrics = {}
    
    try:
        # The 'with' block automatically calls
        # profiler.start_monitoring() and profiler.stop_monitoring()
        with profiler:
            if hasattr(cfg, 'scenario') and cfg.scenario:
                run_scenario_test(loader, cfg)
            else:
                run_basic_test(loader, model_config)
        
        # Get metrics *after* the 'with' block has finished
        device_metrics = profiler.get_metrics()
        
    except Exception as e:
        print(f"!!! Benchmark run failed: {e} !!!")
        # Ensure monitoring stops even if the task fails
        if profiler._is_monitoring:
            profiler.stop_monitoring()
    
    end_time = time.perf_counter()
    total_time_s = end_time - start_time
    print(f"--- Benchmark Run Finished ({total_time_s:.2f}s) ---")

    # 4. Report results (simple print for now)
    print("\n--- Collected Metrics ---")
    print("Device Metrics:")
    print(device_metrics)
    print(f"Total Wall Time: {total_time_s:.2f}s")
    print("-------------------------\n")


def run_scenario_test(loader, cfg):
    """Run ultra-simple scenario test"""
    # Import scenarios inside the function to avoid circular dependencies
    from components.scenarios.simple.simple_llm import SimpleLLMScenario
    from components.scenarios.simple.simple_vlm import SimpleVLMScenario
    from components.scenarios.simple.simple_timeseries import SimpleTimeSeriesScenario
    
    model_category = cfg.model.model_category
    
    # Get scenario
    if model_category == "VLM":
        scenario = SimpleVLMScenario(dict(cfg.scenario))
    elif model_category == "TIME_SERIES":
        scenario = SimpleTimeSeriesScenario(dict(cfg.scenario))
    else:
        scenario = SimpleLLMScenario(dict(cfg.scenario))
    
    tasks = scenario.get_tasks()
    print(f"Running {scenario.name} with {len(tasks)} task(s)...")
    
    # Run single task
    task = tasks[0]
    print(f"Prompt: {task['prompt']}")
    
    try:
        if model_category == "VLM":
            # VLM uses prompt (text) and image
            result = loader.predict(task['prompt'], task.get('image'))
        elif model_category == "TIME_SERIES":
            # Time Series model expects the data (tensor) to be passed as the 'prompt' argument.
            result = loader.predict(task.get('time_series_data'))
        else:
            # LLM uses only prompt (text)
            result = loader.predict(task['prompt'])
        
        print(f"Result: {result}")
        
        evaluation = scenario.evaluate(task, result)
        print(f"Success: {evaluation['success']}")
        
    except Exception as e:
        print(f"Error: {e}")

def run_basic_test(loader, model_config):
    """Run basic test without scenario"""
    print("Running basic test...")
    
    if model_config.model_category == "VLM":
        print("VLM model loaded. Use +scenario=simple_vlm to test.")
    elif model_config.model_category == "TIME_SERIES":
        print("Time Series model loaded. Use +scenario=simple_timeseries to test.")
    else:
        # Basic LLM test
        prompt = "Hello, what is your name?"
        print(f"Test Prompt: {prompt}")