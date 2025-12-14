#!/usr/bin/env python3
"""Generate graphs from FMBench suite results."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd

from suite_config import PROFILER_PATTERNS, SKIP_ACCURACY_SCENARIOS
from suite_utils.plotting_utils import (
    generate_scenario_plots,
    generate_summary_plots,
    generate_idle_power_table
)
from suite_utils.report_generator import generate_latex_table


def parse_suite_log(log_path: Path) -> List[Tuple[Path, str]]:
    """Parse suite log to extract result directory paths and their configs."""
    results = []
    # Pattern to match config line: [1/5] model=...
    config_pattern = re.compile(r'\[\d+/\d+\] (.+)')
    # Pattern to match result path
    path_pattern = re.compile(r'Results will be saved to: (.+)')
    
    current_config = ""
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for config line
            config_match = config_pattern.search(line)
            if config_match:
                current_config = config_match.group(1).strip()
                continue
                
            # Check for result path
            path_match = path_pattern.search(line)
            if path_match:
                path = Path(path_match.group(1).strip())
                results.append((path, current_config))
                # Reset config to avoid reusing it incorrectly (though usually 1:1)
                current_config = ""
    
    return results


def load_results(result_info: List[Tuple[Path, str]]) -> Tuple[pd.DataFrame, dict]:
    """Load summary.json from each result directory into a DataFrame.
    
    Returns:
        DataFrame with results and dict with device info.
    """
    records = []
    device_info = {}  # Will store device names by type
    
    for result_dir, config_str in result_info:
        summary_path = result_dir / 'summary.json'
        if not summary_path.exists():
            continue
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        
        metadata = data.get('metadata', {})
        model_id = metadata.get('model_id', 'unknown')
        # Use short model name (last part of path)
        model_name = model_id.split('/')[-1]
        scenario = metadata.get('scenario', 'unknown')
        
        record = {
            'model': model_name,
            'scenario': scenario,
            'latency': data.get('avg_latency'),
            'ttft': data.get('first_token_ttft'),
            'accuracy': data.get('accuracy'),
            'perplexity': data.get('average_perplexity'),
            'sMAPE': data.get('avg_sMAPE'),
            'tokens_per_output': data.get('avg_tokens_per_output'),
        }
        
        # Extract energy, power, and device info from profilers
        device_metrics = data.get('device_metrics', {})
        energy = None
        idle_power = None
        
        # Hardware Metrics
        cpu_util = None
        gpu_util = None
        ram_usage = None
        vram_usage = None
        
        for key, metrics in device_metrics.items():
            key_lower = key.lower()
            
            # Detect device type and store device name
            for pattern, label in PROFILER_PATTERNS.items():
                if pattern in key_lower:
                    if label not in device_info:
                        device_info[label] = metrics.get('device_name')
                    break
            
            # Extract CPU Util (from CPU profiler)
            if 'cpuprofiler' in key_lower:
                if metrics.get('average_cpu_utilization_percent') is not None:
                    cpu_util = metrics.get('average_cpu_utilization_percent')
                if metrics.get('average_memory_mb') is not None:
                    ram_usage = metrics.get('average_memory_mb')
            
            # Extract GPU metrics (Energy, Util, VRAM)
            # Prefer discrete GPU or specific accelerator profilers
            if 'cpuprofiler' not in key_lower:
                if metrics.get('total_energy_joules') is not None:
                    # Sum energy if multiple devices? Currently picking first found
                    if energy is None:
                        energy = metrics.get('total_energy_joules')
                    else:
                        energy += metrics.get('total_energy_joules')
                
                # For idle scenario, capture average power
                if scenario == 'Idle Baseline':
                    if metrics.get('average_power_watts') is not None:
                        idle_power = metrics.get('average_power_watts')

                # GPU Utilization
                # Check standardized key or fallback
                g_util = metrics.get('average_gpu_utilization_percent') or metrics.get('average_utilization_percent')
                if g_util is not None:
                    # If multiple GPUs, maybe average them? For now pick first non-None
                    if gpu_util is None:
                        gpu_util = g_util
                
                # VRAM
                if metrics.get('average_memory_mb') is not None:
                    if vram_usage is None:
                        vram_usage = metrics.get('average_memory_mb')
        
        record['energy'] = energy
        record['idle_power_watts'] = idle_power
        record['cpu_util'] = cpu_util
        record['gpu_util'] = gpu_util
        record['ram_usage'] = ram_usage
        record['vram_usage'] = vram_usage
        
        # Track if accuracy is meaningful for this scenario
        record['has_valid_accuracy'] = (
            record['accuracy'] is not None 
            and scenario not in SKIP_ACCURACY_SCENARIOS
            and not scenario.lower().startswith('perplexity')
        )

        # Track if sMAPE is available
        record['has_valid_smape'] = record['sMAPE'] is not None
        
        # Add flags from config string
        # Clean up: remove model=... and scenario=... as they are redundant
        flags = config_str
        if flags:
            parts = flags.split()
            cleaned_parts = [p for p in parts if not p.startswith('model=') and not p.startswith('scenario=')]
            
            formatted_parts = []
            for p in cleaned_parts:
                # Remove common prefixes
                p = p.replace('scenario.', '')
                # Replace assignment with colon
                p = p.replace('=', ': ')
                # Replace underscores with spaces
                p = p.replace('_', ' ')
                formatted_parts.append(p)
                
            record['flags'] = ", ".join(formatted_parts)
        else:
            record['flags'] = 'N/A'
        
        records.append(record)
    
    return pd.DataFrame(records), device_info


def collect_result_dirs(inputs: List[str]) -> List[Tuple[Path, str]]:
    """Collect all result directories from provided inputs (logs or root dirs)."""
    results = []
    
    for inp in inputs:
        path = Path(inp)
        if not path.exists():
            print(f"Warning: Input {path} does not exist.")
            continue
            
        if path.is_file():
            # If it's a log file, parse it
            if path.suffix == '.log' or 'suite_' in path.name:
                print(f"Parsing log file: {path}")
                results.extend(parse_suite_log(path))
            else:
                print(f"Warning: Skipping file {path} (not a recognized log).")
        
        elif path.is_dir():
            # If it's a directory, look for summary.json recursively
            print(f"Scanning directory: {path}")
            found = list(path.rglob('summary.json'))
            print(f"  Found {len(found)} result(s).")
            # For direct directory scan, we don't have the log config string
            results.extend([(f.parent, "") for f in found])
            
    # Remove duplicates (based on path)
    unique_results = {}
    for p, c in results:
        res_p = p.resolve()
        # If we already have this path, prefer the one with config
        if res_p not in unique_results or (not unique_results[res_p] and c):
            unique_results[res_p] = c
            
    return list(unique_results.items())


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate graphs from HoliBench results.")
    parser.add_argument('inputs', nargs='+', help="Path to suite logs (.log) or result directories.")
    parser.add_argument('-o', '--output-dir', help="Directory to save graphs. Default: <first_input>_combined_graphs")
    
    args = parser.parse_args()
    
    # Collect all valid result directories
    # result_info is list of (path, config_str)
    result_info = collect_result_dirs(args.inputs)
    
    if not result_info:
        print("No result directories found from inputs.")
        return
        
    print(f"Total entries to process: {len(result_info)}")
    
    df, device_info = load_results(result_info)
    if df.empty:
        print("No valid results loaded (data is empty).")
        return

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        first_input = Path(args.inputs[0])
        
        # If multiple inputs are provided, create a unique combined folder
        if len(args.inputs) > 1:
            # Create a descriptive name from input stems (limit to first 3 to avoid huge paths)
            stems = [Path(i).stem.replace('suite_', '') for i in args.inputs[:3]]
            base_name = "_".join(stems)
            if len(args.inputs) > 3:
                base_name += f"_and_{len(args.inputs)-3}_more"
            
            # Add timestamp to prevent overwriting
            timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"combined_graphs_{base_name}_{timestamp_suffix}"
            
            output_dir = first_input.parent / folder_name
        else:
            # Default name based on the single input
            safe_stem = first_input.stem
            # If input is a log file, put folder next to it
            output_dir = first_input.parent / f"{safe_stem}_graphs"
            
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate LaTeX tables
    generate_latex_table(df, output_dir, device_info, timestamp)
    
    # Generate Graphs
    generate_scenario_plots(df, output_dir, device_info, timestamp)
    generate_summary_plots(df, output_dir, device_info, timestamp)
    generate_idle_power_table(df, output_dir, device_info, timestamp)
    
    print(f"Graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
