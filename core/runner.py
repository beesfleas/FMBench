from components.models.model_factory import get_model_loader
from omegaconf import DictConfig, OmegaConf
import hydra
import time
import torch

def resolve_device(device_cfg: DictConfig) -> DictConfig:
    """
    Resolve device based on config and availability.
    - type == 'cpu'  -> force CPU
    - type == 'cuda' -> require GPU (error if not available)
    - type == 'auto' -> use CUDA if available else CPU
    Returns a new DictConfig with concrete type ('cpu' or 'cuda') and profiler_class.
    """
    dev_type = device_cfg.get("type", "auto")
    has_cuda = torch.cuda.is_available()

    if dev_type == "cpu":
        return OmegaConf.create({
            "type": "cpu",
            "name": "cpu_profiler",
            "profiler_class": "components.devices.cpu_profiler.LocalCpuProfiler"
        })
    if dev_type == "cuda":
        if not has_cuda:
            raise RuntimeError("device=cuda requested but no CUDA device is available.")
        return OmegaConf.create({
            "type": "cuda",
            "cuda_device": device_cfg.get("cuda_device", 0),
            "name": "nvidia_gpu_profiler",
            "profiler_class": "components.devices.nvidia_gpu_profiler.NvidiaGpuProfiler"
        })
    # auto
    if has_cuda:
        return OmegaConf.create({
            "type": "cuda",
            "cuda_device": device_cfg.get("cuda_device", 0),
            "name": "nvidia_gpu_profiler",
            "profiler_class": "components.devices.nvidia_gpu_profiler.NvidiaGpuProfiler"
        })
    return OmegaConf.create({
        "type": "cpu",
        "name": "cpu_profiler",
        "profiler_class": "components.devices.cpu_profiler.LocalCpuProfiler"
    })


def get_profiler(device_config: DictConfig):
    devicdevice_confige_type = resolve_device(device_config)
    profiler_class_path = device_config.profiler_class
    print(f"Instantiating profiler from: {profiler_class_path}")
    return hydra.utils.instantiate({"_target_": profiler_class_path, "config": device_config})

def run_benchmark(cfg: DictConfig):
    model_config = cfg.model
    device_config = resolve_device(cfg.device)
    
    # Load model (Your existing logic)
    loader = get_model_loader(dict(model_config))
    print(f"Loading {model_config.model_category} model: {model_config.model_id}")
    model_cfg_dict = dict(model_config)
    model_cfg_dict["device_preference"] = device_config.type  # 'cpu' or 'cuda'
    loader.load_model(model_cfg_dict)
    
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