from components.models.model_factory import get_model_loader
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import json
from components.devices.profiler_manager import ProfilerManager
from typing import Optional, Tuple

def resolve_device_intent(device_cfg: DictConfig) -> str:
    """
    Resolves the user's *intent* for device type.
    - type == 'cpu'  -> force CPU
    - type == 'cuda' -> *request* CUDA (if available)
    - type == 'auto' -> use CUDA if available, else CPU
    """
    dev_type = device_cfg.get("type", "auto").lower()
    has_cuda = torch.cuda.is_available()

    if dev_type == "cpu":
        return "cpu"
    if dev_type == "cuda":
        if not has_cuda:
            raise RuntimeError("device=cuda requested but no CUDA device is available.")
        return "cuda"
    return "cuda" if has_cuda else "cpu"

def _setup_environment(cfg: DictConfig) -> ProfilerManager:
    """
    Resolves device intent (from config).
    Initializes the ProfilerManager based on the platform.
    """
    # Resolve Device Intent and update config
    device_type = resolve_device_intent(cfg.get("device", OmegaConf.create({"type": "auto"})))
    cfg.device.type = device_type
    print(f"Resolved device intent: {device_type}")
    
    # Setup Device Profilers
    profiler_manager = ProfilerManager(cfg)
    return profiler_manager

def _setup_benchmark(cfg: DictConfig) -> Tuple[object, Optional[object]]:
    """
    Loads the model and scenario (if one is provided).
    """
    # Load Model
    print("Loading model...")
    loader = get_model_loader(cfg.model)
    loader.load_model(cfg.model)
    print(f"Loaded {cfg.model.model_category} model: {cfg.model.model_id}")

    # Load Scenario
    scenario = None
    if "scenario" in cfg and cfg.scenario:
        print(f"Loading scenario: {cfg.scenario._target_}")
        scenario = hydra.utils.instantiate(cfg.scenario)
        scenario.load_tasks()
        print(f"Scenario '{scenario.name}' loaded with {len(scenario.tasks)} tasks.")
    else:
        print("No scenario specified, running basic sanity test.")
        
    return loader, scenario

def _run_execution(loader: object, scenario: Optional[object], model_config: DictConfig, profiler_manager: ProfilerManager):
    """
    Run inference with profiling.
    """
    print("Starting benchmark execution...")
    all_metrics = {}

    with profiler_manager:
        if scenario:
            run_scenario(loader, scenario, model_config.model_category)
        else:
            run_basic_test(loader, model_config)
    
    print("Profiling complete.")
    all_metrics["device_metrics"] = profiler_manager.get_all_metrics()
    print("Collected device metrics.")
    
    # TODO: Add scenario metrics collection here
    # all_metrics["scenario_metrics"] = scenario.get_results()
    
    return all_metrics

def _teardown_and_aggregate(loader: Optional[object], all_metrics: dict):
    """
    Unload model and aggregate metrics.
    """
    # Unload Model
    if loader:
        print("Unloading model...")
        if hasattr(loader, 'unload_model') and callable(loader.unload_model):
            loader.unload_model()
        else:
            print("(Loader has no unload_model method, skipping.)")
            
    # Run Metric Aggregator
    _aggregate_metrics(all_metrics)

def _aggregate_metrics(all_metrics: dict):
    """
    Placeholder for the metric aggregator.
    """
    print("\n--- Benchmark Complete: Final Metrics ---")
    if all_metrics:
        try:
            # Use json for a clean print of the collected metrics
            print(json.dumps(all_metrics, indent=2, default=str))
        except Exception:
            # Fallback
            print(all_metrics)
    else:
        print("No metrics collected.")
    print("-----------------------------------------")

def run_benchmark(cfg: DictConfig):
    """
    Main benchmark orchestration function, now following the ideal flow.
    """
    print("Starting benchmark run...")
    print("--- Config Info ---\n", OmegaConf.to_yaml(cfg))

    loader, profiler_manager = None, None
    all_metrics = {}

    try:
        # Setup device profilers
        profiler_manager = _setup_environment(cfg)
        # Print hardware info
        profiler_manager.print_hardware_info()
        # Load model
        loader, scenario = _setup_benchmark(cfg)
        # Inference
        all_metrics = _run_execution(loader, scenario, cfg.model, profiler_manager)

    except Exception as e:
        print(f"\n--- FATAL BENCHMARK ERROR ---")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("-------------------------------")
    
    finally:
        # Unload model and Aggregate results
        _teardown_and_aggregate(loader, all_metrics)

# --- These helper functions are unchanged ---

def run_scenario(loader, scenario, model_category):
    """
    Placeholder for scenario-based benchmarking.
    """
    print(f"Running scenario (placeholder): {scenario.name}")
    # Future implementation will iterate through scenario.tasks,
    # call loader.predict(), and run scenario.evaluate().
    pass


def run_basic_test(loader, model_config):
    """Run basic sanity test without scenario"""
    print("Running basic test...")
    
    if model_config.model_category == "VLM":
        print("VLM model loaded. Use +scenario=simple_vlm to test.")
    elif model_config.model_category == "TIME_SERIES":
        print("Time Series model loaded. Use +scenario=simple_timeseries to test.")
    else:
        # Basic LLM test
        prompt = "Tell me a joke."
        print(f"Test Prompt:\n{prompt}")
        try:
            result = loader.predict(prompt)
            print(f"Test Result:\n{result}")
        except Exception as e:
            print(f"Test prediction failed: {e}")