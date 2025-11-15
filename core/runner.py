import logging
from components.models.model_factory import get_model_loader
from omegaconf import DictConfig, OmegaConf
import hydra
import json
from components.devices.profiler_manager import ProfilerManager
from typing import Optional, Tuple

log = logging.getLogger(__name__)

def _setup_profilers(cfg: DictConfig) -> ProfilerManager:
    """
    Initialize the ProfilerManager based on platform and config.
    """
    log.debug(f"Initializing profilers with device config: {cfg.get('device', {})}")
    profiler_manager = ProfilerManager(cfg)
    return profiler_manager

def _setup_benchmark(cfg: DictConfig) -> Tuple[object, Optional[object]]:
    """
    Loads the model and scenario.
    """
    # Load Model
    log.debug("Loading model...")
    loader = get_model_loader(cfg.model)
    loader.load_model(cfg.model)

    # Load Scenario
    scenario = None
    if "scenario" in cfg and cfg.scenario:
        log.info(f"Loading scenario: {cfg.scenario._target_}")
        scenario = hydra.utils.instantiate(cfg.scenario)
        scenario.load_tasks()
        log.info(f"Scenario '{scenario.name}' loaded with {len(scenario.tasks)} tasks.")
    else:
        log.warning("No scenario specified, running basic sanity test.")
        
    return loader, scenario

def _run_execution(loader: object, scenario: Optional[object], model_config: DictConfig, profiler_manager: ProfilerManager):
    """
    Run inference with profiling.
    """
    log.info("Starting benchmark execution...")
    all_metrics = {}

    with profiler_manager:
        if scenario:
            run_scenario(loader, scenario, model_config.model_category)
        else:
            run_basic_test(loader, model_config)
    
    log.info("Benchmark completed.")
    all_metrics["device_metrics"] = profiler_manager.get_all_metrics()
    
    # TODO: Add scenario metrics collection here
    # all_metrics["scenario_metrics"] = scenario.get_results()
    
    return all_metrics

def _teardown_and_aggregate(loader: Optional[object], all_metrics: dict):
    """
    Unload model and aggregate metrics.
    """
    # Unload Model
    if loader:
        log.debug("Unloading model...")
        if hasattr(loader, 'unload_model') and callable(loader.unload_model):
            loader.unload_model()
        else:
            log.warning("(Loader has no unload_model method, skipping.)")
            
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
    Main benchmark orchestration function.
    """
    log.info("Starting FMBench...")
    log.info("--- Received Configs ---\n" + OmegaConf.to_yaml(cfg) + "------------------------")

    loader, profiler_manager = None, None
    all_metrics = {}

    try:
        # Setup device profilers (handles device detection and profiler selection)
        profiler_manager = _setup_profilers(cfg)
        
        # Load model and scenario
        loader, scenario = _setup_benchmark(cfg)
        
        # Run inference with profiling
        all_metrics = _run_execution(loader, scenario, cfg.model, profiler_manager)

    except Exception as e:
        log.critical("\n--- FATAL BENCHMARK ERROR ---", exc_info=True)
        log.critical(f"{type(e).__name__}: {e}")
        log.critical("-------------------------------")
    
    finally:
        # Unload model and aggregate results
        _teardown_and_aggregate(loader, all_metrics)

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
    log.info("Running basic test...")
    
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
            print(f"Model Response:\n{result}")
        except Exception as e:
            log.error(f"Test prediction failed: {e}", exc_info=True)