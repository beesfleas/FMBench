import logging
from components.models.model_factory import get_model_loader
from omegaconf import DictConfig, OmegaConf
import hydra
import json
from components.devices.profiler_manager import ProfilerManager
from components.scenarios.perplexity_scenario import PerplexityScenario
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
    
    # Sync device preference from global device config if needed
    device_type = cfg.get("device", {}).get("type", "auto")
    if device_type in ["cuda", "cuda-only"]:
        log.info(f"Global device type '{device_type}' implies CUDA preference for model.")
        OmegaConf.set_struct(cfg.model, False) # Allow adding new keys
        cfg.model.device_preference = "cuda"
        OmegaConf.set_struct(cfg.model, True)

    # Inject allow_mps_fallback to model config
    if cfg.get("allow_mps_fallback") is not None:
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.allow_mps_fallback = cfg.allow_mps_fallback
        OmegaConf.set_struct(cfg.model, True)

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
            results = run_scenario(loader, scenario, model_config.model_category)
            # Aggregate scenario metrics
            if results:
                # Calculate average accuracy if available
                accuracies = [r.get("accuracy", 0.0) for r in results if "accuracy" in r]
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    all_metrics["accuracy"] = avg_accuracy
                
                # Report TTFT for the first query only
                if results and results[0].get("ttft") is not None:
                    first_token_ttft = results[0]["ttft"]
                    all_metrics["first_token_ttft"] = first_token_ttft
                    log.info(f"First Token TTFT: {first_token_ttft:.4f}s")

                # Generic metric aggregation
                # Collect all keys from first result that are floats and not in ignore list
                ignore_keys = {"ttft", "latency", "num_tokens", "accuracy", "perplexity", "input", "target", "source", "output"}
                if results:
                    sample = results[0]
                    for key in sample.keys():
                        if key not in ignore_keys and isinstance(sample[key], (int, float)):
                            # Calculate average
                            values = [r.get(key) for r in results if r.get(key) is not None]
                            if values:
                                avg_val = sum(values) / len(values)
                                all_metrics[f"avg_{key}"] = avg_val
                                log.info(f"Average {key}: {avg_val:.4f}")

                # Calculate Average Latency from 5th question onwards (warm-up)
                latencies = [r.get("latency") for r in results[4:] if r.get("latency") is not None]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    all_metrics["avg_latency"] = avg_latency
                    log.info(f"Average Latency (samples {min(5, len(results))}-{len(results)}): {avg_latency:.4f}s")

                # Calculate Average Tokens per Output
                token_counts = [r.get("num_tokens") for r in results if r.get("num_tokens") is not None]
                if token_counts:
                    avg_tokens = sum(token_counts) / len(token_counts)
                    all_metrics["avg_tokens_per_output"] = avg_tokens
                    log.info(f"Average Tokens per Output: {avg_tokens:.2f}")

                # Calculate Average Perplexity
                ppls = [r.get("perplexity") for r in results if r.get("perplexity") is not None]
                if ppls:
                    avg_ppl = sum(ppls) / len(ppls)
                    all_metrics["average_perplexity"] = avg_ppl
                    log.info(f"Average Perplexity: {avg_ppl:.4f}")

                all_metrics["total_samples"] = len(results)
                # Store full results if needed, or just summary
                # all_metrics["scenario_results"] = results  
        else:
            run_basic_test(loader, model_config)
    
    all_metrics["device_metrics"] = profiler_manager.get_all_metrics()
    log.debug("Collected metrics from %d profiler(s)", len(all_metrics["device_metrics"]))
    
    return all_metrics

def _teardown_and_aggregate(loader: Optional[object], all_metrics: dict):
    """
    Unload model and aggregate metrics.
    """
    # Unload Model
    if loader:
        if hasattr(loader, 'unload_model') and callable(loader.unload_model):
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
    Run the scenario tasks and collect metrics.
    """
    log.info("Running scenario: %s", scenario.name)
    
    results = []
    
    is_perplexity = isinstance(scenario, PerplexityScenario)
    
    for i, task in enumerate(scenario.tasks):
        log.debug(f"Processing task {i+1}/{len(scenario.tasks)}")
        
        prompt = task.get("prompt")
        if not prompt and "input" in task:
            # If prompt is not pre-calculated, format it using the scenario's template
            from langchain_core.prompts import PromptTemplate
            tmpl = PromptTemplate.from_template(scenario.prompt_template)
            prompt = tmpl.format(input=task["input"])
        image = task.get("image")
        time_series_data = task.get("time_series_data")
        
        try:
            additional_metrics = {}
            # Dispatch based on model category to avoid passing unsupported arguments
            additional_metrics = {}
            # Dispatch based on model category to avoid passing unsupported arguments
            if is_perplexity:
                # Special handling for perplexity
                # input is the full text
                prompt = task["input"]
                # We expect the loader to have compute_perplexity
                if hasattr(loader, 'compute_perplexity'):
                    ppl = loader.compute_perplexity(prompt)
                    raw_output = ppl # Pass float directly
                    output = ppl
                else:
                    log.error("Model loader %s does not support compute_perplexity", loader.__class__.__name__)
                    raw_output = float("nan")
                    output = raw_output

            elif model_category == "LLM":
                raw_output = loader.predict(prompt)
            elif model_category == "VLM":
                raw_output = loader.predict(prompt, image=image)
            elif model_category == "TIME_SERIES":
                raw_output = loader.predict(prompt=prompt, time_series_data=time_series_data)
            else:
                # Fallback for unknown categories
                raw_output = loader.predict(prompt)
            
            if isinstance(raw_output, dict) and "output" in raw_output:
                output = raw_output["output"]
                additional_metrics = {k: v for k, v in raw_output.items() if k != "output"}
            elif not is_perplexity:
                output = raw_output
                additional_metrics = {}
            else:
                 # Perplexity case, raw_output is float, no additional metrics usually from predict
                 additional_metrics = {}
                
            metrics = scenario.evaluate(task, output)
            metrics.update({k: v for k, v in additional_metrics.items() if v is not None})
            
            # Log progress every 10% or at least every 10 tasks
            if (i + 1) % max(1, len(scenario.tasks) // 10) == 0:
                log.info(f"Processed {i+1}/{len(scenario.tasks)} tasks")
            
            log.debug(f"Task {i+1} Result: {metrics}")
            if "latency" in additional_metrics:
                log.info(f"Task {i+1} Latency: {additional_metrics['latency']:.4f}s, Output: {output[:50] if isinstance(output, str) else output}...")
            
            if is_perplexity:
                 log.info(f"Task {i+1} Perplexity: {metrics.get('perplexity')}")

            results.append(metrics)
            
        except Exception as e:
            log.error(f"Task {i+1} failed: {e}", exc_info=True)
            
    return results

def run_basic_test(loader, model_config):
    """Run basic sanity test without scenario"""
    if model_config.model_category == "VLM":
        log.warning("VLM model requires a scenario. Use +scenario=simple_vlm to test")
    elif model_config.model_category == "TIME_SERIES":
        log.warning("Time Series model requires a scenario. Use +scenario=simple_timeseries to test")
    elif model_config.model_category == "LLM":
        log.warning("LLM model requires a scenario. Use +scenario=simple_llm to test")
    else:
        # Basic LLM test
        prompt = "Tell me a joke."
        log.info("Running basic test with prompt: %s", prompt)
        try:
            result = loader.predict(prompt)
            if isinstance(result, dict):
                log.info("Test completed successfully. Response: %s (TTFT: %s)", 
                         result.get("output"), result.get("ttft"))
            else:
                log.info("Test completed successfully. Response: %s", result)
        except Exception as e:
            log.error("Test prediction failed: %s", e, exc_info=True)