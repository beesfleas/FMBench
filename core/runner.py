# core/runner.py
from components.models.huggingface_llm import HuggingFaceLLMLoader
from components.models.huggingface_vlm import HuggingFaceVLMLoader
from components.models.huggingface_timeseries import HuggingFaceTimeSeriesLoader

def get_model_loader(model_config):
    model_category = model_config.get("model_category", "LLM")
    
    if model_category == "VLM":
        return HuggingFaceVLMLoader()
    elif model_category == "TIME_SERIES":
        return HuggingFaceTimeSeriesLoader()
    else:
        return HuggingFaceLLMLoader()

def run_benchmark(cfg):
    model_config = cfg.model
    
    # Load model
    loader = get_model_loader(dict(model_config))
    print(f"Loading {model_config.model_category} model: {model_config.model_id}")
    loader.load_model(dict(model_config))
    
    # Check if scenario is provided
    if hasattr(cfg, 'scenario') and cfg.scenario:
        run_scenario_test(loader, cfg)
    else:
        run_basic_test(loader, model_config)

def run_scenario_test(loader, cfg):
    """Run ultra-simple scenario test"""
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
            result = loader.predict(task['prompt'], task.get('image'))
        elif model_category == "TIME_SERIES":
            result = loader.predict(task['prompt'], time_series_data=task.get('time_series_data'))
        else:
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
        prompt = "Hello"
        result = loader.predict(prompt)
        print(f"Basic test result: {result}")