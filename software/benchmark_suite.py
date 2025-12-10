#!/usr/bin/env python3
"""
Run FMBench with multiple configurations. Logs saved to suite_logs/.

Usage:
    python benchmark_suite.py [--device-level SoC|Mobile|Server] [--summary-only] [-y]

Configuration:
    Edit fmbench_config.py to customize which models and scenarios to run.
"""
import argparse
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from suite_config import (
    DEVICE_LEVEL, GLOBAL_SETTINGS, DEFAULT_NUM_SAMPLES,
    DEVICE_LIMITS, BENCHMARK_CONFIG
)
from suite_utils.suite_utils import (
    get_model_category, get_model_parameter_count,
    is_model_allowed_for_device, format_time
)

# =============================================================================
# CONFIG BUILDING
# =============================================================================

def build_config(model: str, scenario: str, scenario_params: Dict) -> Dict:
    """Build a single benchmark configuration."""
    config = {**GLOBAL_SETTINGS, "model": model, "scenario": scenario}
    
    # Add default num_samples unless skipped
    if not scenario_params.pop("_skip_num_samples", False):
        config["scenario.num_samples"] = DEFAULT_NUM_SAMPLES
    
    # Add scenario-specific params
    config.update(scenario_params)
    return config


def build_configs(device_level: str = DEVICE_LEVEL) -> List[Dict]:
    """Build all benchmark configurations, filtering by device level."""
    configs = []
    warnings = []
    filtered = []
    
    for expected_category, config in BENCHMARK_CONFIG.items():
        models = config.get("models", [])
        scenarios = config.get("scenarios", {})
        
        for model in models:
            # Verify category match
            actual_category = get_model_category(model)
            if actual_category != expected_category:
                warnings.append(
                    f"Warning: Model '{model}' is {actual_category}, "
                    f"but listed under {expected_category}. Skipping."
                )
                continue
            
            # Filter by device capability
            if not is_model_allowed_for_device(model, device_level):
                param_count = get_model_parameter_count(model)
                size_str = f"{param_count}B" if param_count else "unknown size"
                filtered.append(f"{model} ({size_str})")
                continue
            
            # Create configs for all scenarios
            for scenario_name, scenario_params in scenarios.items():
                configs.append(build_config(model, scenario_name, dict(scenario_params)))
    
    # Print warnings
    if warnings:
        print("\n".join(warnings), "\n")
    
    # Print filtered models
    if filtered and device_level != "Server":
        print(f"Device Level: {device_level}")
        print(f"Filtered out {len(filtered)} model(s):")
        for model_info in filtered:
            print(f"  • {model_info}")
        if any("unknown size" in info for info in filtered):
            print("  Note: Unknown models filtered conservatively.")
            print("        Add to KNOWN_MODELS dict in fmbench_config.py if needed.\n")
        else:
            print()
    
    return configs


def print_summary(configs: List[Dict], device_level: str) -> None:
    """Print benchmark summary."""
    if not configs:
        print("No configurations to run.")
        return
    
    # Group by category and model
    by_category = defaultdict(lambda: defaultdict(list))
    for cfg in configs:
        category = get_model_category(cfg["model"])
        by_category[category][cfg["model"]].append(cfg["scenario"])
    
    # Print header
    limit = DEVICE_LIMITS[device_level]
    limit_str = f"<= {limit}B" if limit != float("inf") else "all models"
    
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Device Level: {device_level} ({limit_str})")
    print(f"Total configurations: {len(configs)}\n")
    
    # Print by category
    for category in sorted(by_category.keys()):
        print(f"{category}:")
        for model in sorted(by_category[category].keys()):
            scenarios = sorted(by_category[category][model])
            param_count = get_model_parameter_count(model)
            size_str = f" ({param_count}B)" if param_count else ""
            print(f"  • {model}{size_str}: {len(scenarios)} scenarios")
            print(f"    {', '.join(scenarios)}")
        print()
    
    print("=" * 70)

# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

LOG_DIR = Path("suite_logs")


def run_benchmark(config: Dict, log_file) -> Tuple[bool, float]:
    """Run one benchmark, return (success, duration)."""
    base_dir = Path(__file__).parent
    run_script = base_dir / "run.py"
    args = [sys.executable, str(run_script)] + [f"{k}={v}" for k, v in config.items()]
    
    start = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
        output_lines.append(line)
    
    proc.wait()
    
    # Check for failure indicators in output
    output = "".join(output_lines)
    no_metrics = "No metrics collected" in output
    
    success = proc.returncode == 0 and not no_metrics
    return success, time.time() - start


def log_message(msg: str, log_file):
    """Print and write to log file."""
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def run_benchmarks(configs: List[Dict], device_level: str) -> int:
    """Run all benchmarks and return exit code."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"suite_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    with open(log_path, "w", encoding="utf-8") as f:
        log_message(f"FMBench Suite - {len(configs)} configs - Device: {device_level} - Log: {log_path}", f)
        log_message("=" * 70, f)
        
        results = []
        total_start = time.time()
        
        # Run each benchmark
        for i, cfg in enumerate(configs, 1):
            cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
            log_message(f"\n[{i}/{len(configs)}] {cfg_str}", f)
            log_message("-" * 70, f)
            
            success, duration = run_benchmark(cfg, f)
            results.append((cfg_str, success, duration))
            status = "PASSED" if success else "FAILED"
            log_message(f"{status} in {format_time(duration)}\n", f)
        
        # Final summary
        log_message("=" * 70, f)
        log_message("FINAL SUMMARY", f)
        log_message("=" * 70, f)
        for cfg_str, success, duration in results:
            status = "PASS" if success else "FAIL"
            log_message(f"  [{status}] {format_time(duration):>10}  {cfg_str}", f)
        
        passed = sum(success for _, success, _ in results)
        total_time = time.time() - total_start
        log_message(f"\n{passed}/{len(results)} passed in {format_time(total_time)}", f)
        log_message(f"Log saved to: {log_path}", f)
    
    # Generate graphs
    print("\nGenerating graphs...")
    try:
        # Call generate_graphs.py as a subprocess to handle argument parsing cleanly
        base_dir = Path(__file__).parent
        generate_graphs_script = base_dir / "generate_graphs.py"
        cmd = [sys.executable, str(generate_graphs_script), str(log_path)]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate graphs: {e}")
    except Exception as e:
        print(f"Warning: Error invoking graph generation: {e}")
    
    return 0 if passed == len(results) else 1

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run FMBench benchmark suite with multiple model/scenario combinations"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary and exit"
    )
    parser.add_argument(
        "--device-level",
        choices=["SoC", "Mobile", "Server"],
        default=DEVICE_LEVEL,
        help=f"Device capability level (default: {DEVICE_LEVEL})"
    )
    args = parser.parse_args()
    
    # Build configurations
    configs = build_configs(device_level=args.device_level)
    
    if not configs:
        print("Error: No configurations to run.")
        print(f"Device Level: {args.device_level}")
        print("Edit BENCHMARK_CONFIG in fmbench_config.py to add models and scenarios.")
        sys.exit(1)
    
    # Print summary
    print_summary(configs, args.device_level)
    
    # Exit early if summary-only
    if args.summary_only:
        sys.exit(0)
    
    # Confirm before running
    if not args.yes:
        try:
            response = input("\nProceed with benchmark? [Y/n]: ").strip().lower()
            if response and response != 'y':
                print("Cancelled.")
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)
    
    # Run benchmarks
    sys.exit(run_benchmarks(configs, args.device_level))


if __name__ == "__main__":
    main()
