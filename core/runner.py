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
    log.debug("Initializing profilers with device config: %s", cfg.get('device', {}))
    profiler_manager = ProfilerManager(cfg)
    return profiler_manager

def _setup_benchmark(cfg: DictConfig) -> Tuple[object, Optional[object]]:
    """
    Loads the model and scenario.
    """
    # Load Model
    model_id = cfg.model.get("model_id", "unknown")
    log.info("Loading model: %s", model_id)
    loader = get_model_loader(cfg.model)
    log.debug("Model loader: %s", loader.__class__.__name__)
    loader.load_model(cfg.model)

    # Load Scenario
    scenario = None
    if "scenario" in cfg and cfg.scenario:
        log.info("Loading scenario: %s", cfg.scenario._target_)
        scenario = hydra.utils.instantiate(cfg.scenario)
        scenario.load_tasks()
        log.info("Scenario loaded: %s (%d tasks)", scenario.name, len(scenario.tasks))
    else:
        log.warning("No scenario specified, running basic sanity test")
        
    return loader, scenario

def _run_execution(loader: object, scenario: Optional[object], model_config: DictConfig, profiler_manager: ProfilerManager):
    """
    Run inference with profiling.
    """
    log.info("Starting benchmark execution")
    all_metrics = {}

    with profiler_manager:
        if scenario:
            run_scenario(loader, scenario, model_config.model_category)
        else:
            run_basic_test(loader, model_config)
    
    all_metrics["device_metrics"] = profiler_manager.get_all_metrics()
    log.debug("Collected metrics from %d profiler(s)", len(all_metrics["device_metrics"]))
    
    # TODO: Add scenario metrics collection here
    # all_metrics["scenario_metrics"] = scenario.get_results()
    
    return all_metrics

def _teardown_and_aggregate(loader: Optional[object], all_metrics: dict):
    """
    Unload model and aggregate metrics.
    """
    # Unload Model
    if loader:
        if hasattr(loader, 'unload_model') and callable(loader.unload_model):
            log.debug("Unloading model")
            loader.unload_model()
        else:
            log.warning("Model loader does not support unloading")
            
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
    log.info("Starting FMBench")
    log.debug("Configuration:\n%s", OmegaConf.to_yaml(cfg))

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
        log.critical("Fatal benchmark error: %s: %s", type(e).__name__, e, exc_info=True)
    
    finally:
        # Unload model and aggregate results
        _teardown_and_aggregate(loader, all_metrics)
        log.info("Benchmark completed")

def run_scenario(loader, scenario, model_category):
    """
    Placeholder for scenario-based benchmarking.
    """
    log.info("Running scenario: %s", scenario.name)
    # Future implementation will iterate through scenario.tasks,
    # call loader.predict(), and run scenario.evaluate().
    pass


def run_basic_test(loader, model_config):
    """Run basic sanity test without scenario"""
    if model_config.model_category == "VLM":
        log.warning("VLM model requires a scenario. Use +scenario=simple_vlm to test")
    elif model_config.model_category == "TIME_SERIES":
        log.warning("Time Series model requires a scenario. Use +scenario=simple_timeseries to test")
    else:
        # Basic LLM test
        prompt = "Tell me a joke."
        log.debug("Running basic test with prompt: %s", prompt)
        try:
            result = loader.predict(prompt)
            log.info("Test completed successfully (response length: %d)", len(result) if result else 0)
        except Exception as e:
            log.error("Test prediction failed: %s", e, exc_info=True)