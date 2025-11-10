from components.models.model_factory import get_model_loader
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import json
from components.devices.profiler_manager import ProfilerManager

# --- Helper Functions for Orchestration ---

def _setup_components(cfg: DictConfig):
    """
    Initialize all necessary components.
    Returns: (loader, profiler_manager, scenario)
    """
    # 1. Initialize ProfilerManager
    try:
        profiler_manager = ProfilerManager(cfg) 
        print("ProfilerManager initialized.")
    except Exception as e:
        print(f"Error initializing ProfilerManager: {e}. Profiling will be disabled.")
        profiler_manager = None

    # 2. Load Model
    try:
        print("Loading model...")
        loader = get_model_loader(cfg.model)
        loader.load_model(cfg.model)
        print("Model loaded.")
    except Exception as e:
        print(f"Fatal Error: Could not load model: {e}")
        raise 

    # 3. Load Scenario
    scenario = None
    if "scenario" in cfg and cfg.scenario:
        try:
            print(f"Loading scenario: {cfg.scenario._target_}")
            scenario = hydra.utils.instantiate(cfg.scenario)
            scenario.load_tasks()
            print(f"Scenario '{scenario.name}' loaded with {len(scenario.tasks)} tasks.")
        except Exception as e:
            print(f"Fatal Error: Could not load scenario: {e}")
            raise # Re-raise
    else:
        print("No scenario specified, running basic sanity test.")

    return loader, profiler_manager, scenario

def _run_execution(cfg: DictConfig, loader, profiler_manager, scenario):
    """
    Run the core benchmark logic, with or without profiling.
    Returns: (all_metrics)
    """
    print("Starting benchmark execution...")
    all_metrics = {}

    if profiler_manager:
        # Run WITH profiling
        print("Starting run with profiling...")
        with profiler_manager:
            if scenario:
                run_scenario(loader, scenario, cfg.model.model_category)
            else:
                run_basic_test(loader, cfg.model)
        
        # Metrics are collected *after* the 'with' block exits
        all_metrics["device_metrics"] = profiler_manager.get_all_metrics()
        print("Collected device metrics.")

    else:
        # Run WITHOUT profiling
        print("Starting run without profiling...")
        if scenario:
            run_scenario(loader, scenario, cfg.model.model_category)
        else:
            run_basic_test(loader, cfg.model)
        print("Run complete (no profiling).")

    return all_metrics

def _run_metric_aggregator(all_metrics: dict):
    """
    (Placeholder) Future component for analyzing and saving results.
    For now, it just prints the final metrics.
    """
    print("\n--- Benchmark Complete: Final Metrics ---")
    
    if all_metrics:
        try:
            # Use json for a clean print of the collected metrics
            print(json.dumps(all_metrics, indent=2, default=str))
        except Exception as e:
            print(f"Error printing JSON, falling back to raw print: {e}")
            print(all_metrics)
    else:
        print("No metrics were collected.")
        
    print("-----------------------------------------")

def _teardown(loader, all_metrics):
    """
    Clean up resources and pass metrics to the aggregator.
    """
    # 1. Unload Model
    loader.unload_model()
            
    # 2. Run Metric Aggregator (Placeholder)
    _run_metric_aggregator(all_metrics)


# --- Main Orchestration Function ---

def run_benchmark(cfg: DictConfig):
    """
    Main benchmark orchestration function.
    Orchestrates Setup, Execution, and Teardown.
    """
    print("Starting benchmark run...")
    print("--- Provided Configs ---\n", OmegaConf.to_yaml(cfg))

    loader = None
    all_metrics = {}
    
    try:
        # 1. SETUP
        loader, profiler_manager, scenario = _setup_components(cfg)

        # 2. EXECUTION
        all_metrics = _run_execution(cfg, loader, profiler_manager, scenario)

    except Exception as e:
        print(f"\n--- Benchmark RUN FAILED ---")
        print(f"An error occurred during Setup or Execution: {e}")
    
    finally:
        # 3. TEARDOWN
        _teardown(loader, all_metrics)


# --- Scenario & Test Functions ---

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
        prompt = "Tell me a joke."
        print(f"Test Prompt:\n{prompt}")
        try:
            result = loader.predict(prompt)
            print(f"Test Result:\n{result}")
        except Exception as e:
            print(f"Test prediction failed: {e}")