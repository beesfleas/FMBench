#!/usr/bin/env python3
"""Run FMBench with multiple configurations. Logs saved to suite_logs/."""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from generate_graphs import main as generate_graphs

# =============================================================================
# CONFIGURATION - Edit these to customize benchmark runs
# =============================================================================

# Global settings applied to all runs
GLOBAL_SETTINGS = {
    "log_level": "INFO",
}

# Models to benchmark (runs all scenarios for each model)
MODELS = [
    "distilgpt2",
    # "qwen2.5-1.5b",
    # "qwen2.5-1.5b-quantized",
    # "qwen2.5-7b",
    # "qwen2.5-7b-quantized",
    # "qwen3-0.6b",
    # "qwen3-0.6b-quantized",
    # "qwen3-4b",
    # "qwen3-4b-quantized",
    # "qwen3-8b",
    # "qwen3-8b-quantized",

    # "llama2-7b",
    # "llama2-7b-quantized",
    # "llama3.2-1b",
    # "llama3.2-1b-quantized",
    # "llama3.2-3b",
    # "llama3.2-3b-quantized",

    # "falcon-7b",
    # "falcon-7b-quantized",
]

# Default num_samples for scenarios (set to None to disable)
DEFAULT_NUM_SAMPLES = "10"

# Scenarios with optional overrides (name -> extra params dict or None)
# If a scenario has no overrides, use None or {}
SCENARIOS = {
    "idle":                 {"scenario.idle_duration": "10", "_skip_num_samples": True},
    "arc_easy":             {},
    "arc_challenge":        {},
    "classification":       {},
    "ner":                  {},
    "perplexity_c4":        {},
    "perplexity_wikitext2": {},
    # "sentiment":            {},
    # "summarization":        {},
    # "translation":          {},
    # "summarization":        {"scenario.use_expensive_metrics": "True", "scenario.num_samples": "20"},
    # "translation":          {"scenario.use_expensive_metrics": "True", "scenario.num_samples": "20"},
}

# =============================================================================


def build_configs(models=MODELS, scenarios=SCENARIOS, global_settings=GLOBAL_SETTINGS,
                  default_num_samples=DEFAULT_NUM_SAMPLES):
    """Generate config list from models × scenarios."""
    configs = []
    for model in models:
        for scenario_name, scenario_params in scenarios.items():
            params = dict(scenario_params or {})
            skip_num_samples = params.pop("_skip_num_samples", False)
            
            config = {**global_settings, "model": model, "scenario": scenario_name}
            if default_num_samples and not skip_num_samples:
                config["scenario.num_samples"] = default_num_samples
            config.update(params)
            configs.append(config)
    return configs


CONFIGS = build_configs()

LOG_DIR = Path("suite_logs")


def run_benchmark(config, log_file):
    """Run one benchmark, streaming output to console and log file."""
    args = [sys.executable, "run.py"] + [f"{k}={v}" for k, v in config.items()]
    
    start = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
    
    proc.wait()
    return proc.returncode == 0, time.time() - start


def fmt_time(secs):
    """Format seconds as human-readable duration."""
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{int(secs // 60)}m {secs % 60:.0f}s"
    return f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"


def log(msg, file):
    """Print to console and write to log file."""
    print(msg)
    file.write(msg + "\n")
    file.flush()


def main():
    if not CONFIGS:
        sys.exit("No configurations defined. Edit CONFIGS in benchmark_suite.py")
    
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"suite_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    with open(log_path, "w", encoding="utf-8") as f:
        log(f"FMBench Suite - {len(CONFIGS)} configs - Log: {log_path}", f)
        log("=" * 60, f)
        
        results = []
        total_start = time.time()
        
        for i, cfg in enumerate(CONFIGS, 1):
            cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
            log(f"\n[{i}/{len(CONFIGS)}] {cfg_str}", f)
            log("-" * 60, f)
            
            ok, dur = run_benchmark(cfg, f)
            results.append((cfg_str, ok, dur))
            log(f"{'PASSED' if ok else 'FAILED'} in {fmt_time(dur)}\n", f)
        
        # Summary
        log("=" * 60, f)
        log("SUMMARY", f)
        for cfg_str, ok, dur in results:
            log(f"  [{'PASS' if ok else 'FAIL'}] {fmt_time(dur):>10}  {cfg_str}", f)
        
        passed = sum(ok for _, ok, _ in results)
        log(f"\n{passed}/{len(results)} passed in {fmt_time(time.time() - total_start)}", f)
        log(f"Log: {log_path}", f)
    
    # Generate graphs from results
    graph_dir = generate_graphs(log_path)
    print(f"Graphs: {graph_dir}")
    
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()