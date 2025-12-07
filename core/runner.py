import logging
import time
from typing import Optional, Tuple

import hydra
import json
from omegaconf import DictConfig, OmegaConf
from langchain_core.prompts import PromptTemplate

from components.models.model_factory import get_model_loader
from components.devices.profiler_manager import ProfilerManager
from components.scenarios.perplexity_scenario import PerplexityScenario

log = logging.getLogger(__name__)

# Number of initial samples to skip for latency calculation (warm-up period)
WARMUP_SAMPLES = 4

def _setup_profilers(cfg: DictConfig) -> ProfilerManager:
    """
    Initialize the ProfilerManager based on platform and config.
    """
    log.debug("Initializing profilers with device config: %s", cfg.get('device', {}))
    
    # Create a meaningful run name based on timestamp and model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_id = cfg.model.get("model_id", "unknown")
    # Use just the model name (last part of path) for the run name
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    run_name = f"{timestamp}_{model_name}"
    
    profiler_manager = ProfilerManager(cfg, run_name=run_name)
    return profiler_manager

def _setup_benchmark(cfg: DictConfig) -> Tuple[object, Optional[object]]:
    """
    Loads the model and scenario.
    """
    # Load Model
    model_id = cfg.model.get("model_id", "unknown")
    log.info("Loading model: %s", model_id)
    
    # Create merged config to avoid mutating Hydra's immutable config
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    
    # Sync device preference from global config
    device_type = cfg.get("device", {}).get("type", "auto")
    if device_type in ["cuda", "cuda-only"]:
        log.debug("Global device type '%s' implies CUDA preference for model.", device_type)
        model_config["device_preference"] = "cuda"

    # Inject allow_mps_fallback from global config if specified
    if cfg.get("allow_mps_fallback") is not None:
        model_config["allow_mps_fallback"] = cfg.allow_mps_fallback

    loader = get_model_loader(model_config)
    log.debug("Model loader: %s", loader.__class__.__name__)
    loader.load_model(model_config)

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
    
    # Initialize metrics with metadata
    all_metrics = {
        "metadata": {
            "model_id": model_config.get("model_id", "unknown"),
            "model_category": model_config.get("model_category", "unknown"),
            "scenario": scenario.name if scenario else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results_dir": str(profiler_manager.results_dir) if profiler_manager.results_dir else None,
        }
    }

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
                    log.info("First Token TTFT: %.4fs", first_token_ttft)

                # Aggregate numeric metrics (excluding known special keys)
                ignore_keys = {"ttft", "latency", "num_tokens", "accuracy", "perplexity", "input", "target", "source", "output"}
                all_keys = set()
                for res in results:
                    for k, v in res.items():
                        if k not in ignore_keys:
                            try:
                                float(v)
                                all_keys.add(k)
                            except (ValueError, TypeError):
                                pass
                
                for key in sorted(all_keys):
                    values = []
                    for r in results:
                        val = r.get(key)
                        if val is not None:
                            try:
                                values.append(float(val))
                            except (ValueError, TypeError):
                                pass
                    if values:
                        avg_val = sum(values) / len(values)
                        all_metrics[f"avg_{key}"] = avg_val
                        log.info("Average %s: %.4f", key, avg_val)

                # Calculate Average Latency after warm-up period
                latencies = [r.get("latency") for r in results[WARMUP_SAMPLES:] if r.get("latency") is not None]
                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    all_metrics["avg_latency"] = avg_latency
                    start_sample = min(WARMUP_SAMPLES + 1, len(results))
                    log.info("Average Latency (samples %d-%d): %.4fs", start_sample, len(results), avg_latency)

                # Calculate Average Tokens per Output
                token_counts = [r.get("num_tokens") for r in results if r.get("num_tokens") is not None]
                if token_counts:
                    avg_tokens = sum(token_counts) / len(token_counts)
                    all_metrics["avg_tokens_per_output"] = avg_tokens
                    log.info("Average Tokens per Output: %.2f", avg_tokens)

                # Calculate Average Perplexity
                ppls = [r.get("perplexity") for r in results if r.get("perplexity") is not None]
                if ppls:
                    avg_ppl = sum(ppls) / len(ppls)
                    all_metrics["average_perplexity"] = avg_ppl
                    log.info("Average Perplexity: %.4f", avg_ppl)

                all_metrics["total_samples"] = len(results)
                # Store full results if needed, or just summary
                # all_metrics["scenario_results"] = results  
    
    all_metrics["device_metrics"] = profiler_manager.get_all_metrics()
    log.debug("Collected metrics from %d profiler(s)", len(all_metrics["device_metrics"]))
    
    return all_metrics

def _teardown_and_aggregate(loader: Optional[object], all_metrics: dict, profiler_manager: Optional[ProfilerManager] = None):
    """
    Unload model, save metrics to JSON, and print summary.
    """
    # Unload Model
    if loader:
        if hasattr(loader, 'unload_model') and callable(loader.unload_model):
            loader.unload_model()
        else:
            log.warning("Model loader does not support unloading")
    
    # Save metrics to JSON file in results directory
    if profiler_manager and profiler_manager.results_dir and all_metrics:
        _save_metrics_json(all_metrics, profiler_manager.results_dir)
    
    # Print summary to console
    _print_metrics_summary(all_metrics)


def _save_metrics_json(all_metrics: dict, results_dir):
    """
    Save metrics to a JSON file in the results directory.
    """
    json_filepath = results_dir / "summary.json"
    try:
        with open(json_filepath, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        log.info("Metrics saved to: %s", json_filepath)
    except Exception as e:
        log.error("Failed to save metrics JSON: %s", e)


def _print_metrics_summary(all_metrics: dict):
    """Print key metrics summary to console."""
    print("\n" + "=" * 50)
    print("  BENCHMARK SUMMARY")
    print("=" * 50)
    
    if not all_metrics:
        print("  No metrics collected.")
        print("=" * 50)
        return
    
    # Model info
    metadata = all_metrics.get("metadata", {})
    if metadata.get("model_id"):
        print(f"  Model:      {metadata['model_id']}")
    if metadata.get("scenario"):
        print(f"  Scenario:   {metadata['scenario']}")
    
    print("-" * 50)
    
    # Key performance metrics
    if "accuracy" in all_metrics:
        print(f"  Accuracy:   {all_metrics['accuracy']:.2%}")
    if "avg_latency" in all_metrics:
        print(f"  Latency:    {all_metrics['avg_latency']:.3f}s (avg)")
    if "average_perplexity" in all_metrics:
        print(f"  Perplexity: {all_metrics['average_perplexity']:.2f}")
    if "total_samples" in all_metrics:
        print(f"  Samples:    {all_metrics['total_samples']}")
    
    # Hardware metrics (one section per device)
    device_metrics = all_metrics.get("device_metrics", {})
    max_name_len = 32
    for profiler_name, metrics in device_metrics.items():
        print("-" * 50)
        # Device name
        name = metrics.get("device_name", profiler_name)
        if len(name) > max_name_len:
            name = name[:max_name_len - 3] + "..."
        print(f"  Device:     {name}")
        # Energy
        if "total_energy_joules" in metrics and metrics["total_energy_joules"] > 0:
            print(f"  Energy:     {metrics['total_energy_joules']:.1f} J")
        # Samples
        if "num_samples" in metrics and metrics["num_samples"] > 0:
            print(f"  HW Samples: {metrics['num_samples']}")
    
    # Results location
    results_dir = metadata.get("results_dir")
    if results_dir:
        print("-" * 50)
        print(f"  Results:    {results_dir}")
    
    print("=" * 50)

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
        # Unload model, save results, and print summary
        _teardown_and_aggregate(loader, all_metrics, profiler_manager)
        log.info("Benchmark completed")

def run_scenario(loader, scenario, model_category):
    """
    Run the scenario tasks and collect metrics.
    """
    log.info("Running scenario: %s", scenario.name)
    
    # Handle idle/baseline scenarios (no tasks, just sleep)
    if not scenario.tasks:
        idle_duration = getattr(scenario, 'idle_duration', 60)
        log.info("Idle scenario: sleeping for %ds (baseline measurement)", idle_duration)
        time.sleep(idle_duration)
        return []
    
    results = []
    
    is_perplexity = isinstance(scenario, PerplexityScenario)
    
    for i, task in enumerate(scenario.tasks):
        log.debug("Processing task %d/%d", i + 1, len(scenario.tasks))
        
        prompt = task.get("prompt")
        if not prompt and "input" in task:
            tmpl = PromptTemplate.from_template(scenario.prompt_template)
            prompt = tmpl.format(input=task["input"])
        image = task.get("image")
        time_series_data = task.get("time_series_data")
        
        try:
            additional_metrics = {}
            # Dispatch based on model category
            if is_perplexity:
                prompt = task["input"]
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
                additional_metrics = {}
                
            metrics = scenario.evaluate(task, output)
            metrics.update({k: v for k, v in additional_metrics.items() if v is not None})
            
            # Log progress every 10% or at least every 10 tasks
            if (i + 1) % max(1, len(scenario.tasks) // 10) == 0:
                log.info("Processed %d/%d tasks", i + 1, len(scenario.tasks))
            
            log.debug("Task %d Result: %s", i + 1, metrics)
            if "latency" in additional_metrics:
                log.debug("Task %d Latency: %.4fs, Output: %s...", i + 1, additional_metrics['latency'], output[:50] if isinstance(output, str) else output)
            
            if is_perplexity:
                log.debug("Task %d Perplexity: %s", i + 1, metrics.get('perplexity'))

            results.append(metrics)
            
        except Exception as e:
            log.error("Task %d failed: %s", i + 1, e, exc_info=True)
            
    return results
