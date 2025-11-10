from components.models.model_factory import get_model_loader
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import json
from components.devices.profiler_manager import ProfilerManager

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

    # Auto-detection
    if has_cuda:
        print("Auto-detected CUDA, using GPU.")
        return OmegaConf.create({
            "type": "cuda",
            "cuda_device": device_cfg.get("cuda_device", 0),
            "name": "nvidia_gpu_profiler",
            "profiler_class": "components.devices.nvidia_gpu_profiler.NvidiaGpuProfiler"
        })
    else:
        print("Auto-detected no CUDA, using CPU.")
        return OmegaConf.create({
            "type": "cpu",
            "name": "cpu_profiler",
            "profiler_class": "components.devices.cpu_profiler.LocalCpuProfiler"
        })

def run_benchmark(cfg: DictConfig):
    """
    Main benchmark orchestration function.
    """
    print("Starting benchmark run...")
    print("--- Provided Configs ---\n", OmegaConf.to_yaml(cfg)) # Can be noisy

    # 1. Resolve device
    device_cfg = resolve_device(cfg.get("device", OmegaConf.create({"type": "auto"})))
    cfg.device = device_cfg # Update config with resolved device
    print(f"Resolved device: {device_cfg.type}")

    # 2. Initialize ProfilerManager
    profiler_manager = None
    try:
        profiler_manager = ProfilerManager(cfg) 
        print("ProfilerManager initialized.")
    except Exception as e:
        print(f"Error initializing ProfilerManager: {e}. Continuing without profiling.")

    # 3. Load Model
    print("Loading model...")
    loader = get_model_loader(cfg.model)
    loader.load_model(cfg.model)
    print("Model loaded.")

    # 4. Load Scenario
    scenario = None
    if "scenario" in cfg and cfg.scenario:
        print(f"Loading scenario: {cfg.scenario._target_}")
        try:
            # Instantiate scenario using Hydra
            scenario = hydra.utils.instantiate(cfg.scenario)
            scenario.load_tasks()
            print(f"Scenario '{scenario.name}' loaded with {len(scenario.tasks)} tasks.")
        except Exception as e:
            print(f"Error loading scenario: {e}")
            raise
    else:
        print("No scenario specified, running basic test.")

    # 5. Run Benchmark and Profile
    print("Starting benchmark execution...")
    all_metrics = {} # This will store device metrics

    try:
        if profiler_manager:
            # Run WITH profiling
            print("Starting run with profiling...")
            with profiler_manager:
                if scenario:
                    run_scenario(loader, scenario, cfg.model.model_category)
                else:
                    run_basic_test(loader, cfg.model)
            
            print("Profiling complete.")
            device_metrics = profiler_manager.get_all_metrics()
            all_metrics["device_metrics"] = device_metrics
            print("Collected device metrics.")

        else:
            # Run WITHOUT profiling
            print("Starting run without profiling...")
            if scenario:
                run_scenario(loader, scenario, cfg.model.model_category)
            else:
                run_basic_test(loader, cfg.model)
            print("Run complete (no profiling).")

    except Exception as e:
        print(f"Benchmark run failed: {e}")

    # 7. Print Final Metrics
    print("\n--- Benchmark Complete ---")
    if all_metrics: # Only print if we collected device metrics
        try:
            # Use json for a clean print of the collected metrics
            print(json.dumps(all_metrics, indent=2, default=str))
        except Exception:
            # Fallback
            print(all_metrics)
    else:
        print("No metrics collected.")
    print("--------------------------")


def run_scenario(loader, scenario, model_category):
    """
    Placeholder for scenario-based benchmarking.
    This function will be implemented to run detailed tasks
    and measure application-level metrics.
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
        prompt = "Hello, what is your name?"
        print(f"Test Prompt: {prompt}")
        #changes i think ti was missing these
        try:
            result = loader.predict(prompt)
            print(f"Test Result: {result}")
        except Exception as e:
            print(f"Test prediction failed: {e}")