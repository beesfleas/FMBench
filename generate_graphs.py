#!/usr/bin/env python3
"""Generate graphs from FMBench suite results."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Scenarios where accuracy is not meaningful
SKIP_ACCURACY_SCENARIOS = {'Idle Baseline', 'summarization', 'translation'}


def parse_suite_log(log_path: Path) -> list[Path]:
    """Parse suite log to extract result directory paths."""
    result_dirs = []
    pattern = re.compile(r'Results will be saved to: (.+)')
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                result_dirs.append(Path(match.group(1).strip()))
    
    return result_dirs


def load_results(result_dirs: list[Path]) -> pd.DataFrame:
    """Load summary.json from each result directory into a DataFrame."""
    records = []
    
    for result_dir in result_dirs:
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
            'accuracy': data.get('accuracy'),
        }
        
        # Extract energy from GPU profiler (check all GPU profilers)
        device_metrics = data.get('device_metrics', {})
        energy = None
        for key, metrics in device_metrics.items():
            if 'nvidiagpuprofiler' in key.lower():
                energy = metrics.get('total_energy_joules')
                if energy is not None:
                    break
        record['energy'] = energy
        
        # Track if accuracy is meaningful for this scenario
        record['has_valid_accuracy'] = (
            record['accuracy'] is not None 
            and scenario not in SKIP_ACCURACY_SCENARIOS
            and not scenario.lower().startswith('perplexity')
        )
        
        records.append(record)
    
    return pd.DataFrame(records)


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                        title: str, output_path: Path):
    """Create a scatter plot with model labels."""
    # Filter out rows with missing data for the columns we need
    plot_df = df.dropna(subset=[x_col, y_col])
    
    if plot_df.empty:
        return False
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, s=100)
    
    # Add model labels
    for _, row in plot_df.iterrows():
        plt.annotate(
            row['model'], 
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )
    
    plt.title(title)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return True


def generate_scenario_plots(df: pd.DataFrame, output_dir: Path):
    """Generate plots for each scenario."""
    scenarios = df['scenario'].unique()
    
    for scenario in scenarios:
        scenario_df = df[df['scenario'] == scenario]
        safe_name = scenario.lower().replace(' ', '_')
        
        # Skip idle scenario entirely
        if scenario == 'Idle Baseline':
            continue
        
        # 1. Latency vs Accuracy (only if scenario has valid accuracy)
        if scenario_df['has_valid_accuracy'].any():
            create_scatter_plot(
                scenario_df, 'latency', 'accuracy',
                f'{scenario}: Latency vs Accuracy',
                output_dir / f'{safe_name}_latency_vs_accuracy.png'
            )
        
        # Check if energy data is available
        has_energy = scenario_df['energy'].notna().any()
        
        if has_energy:
            # 2. Latency vs Energy
            create_scatter_plot(
                scenario_df, 'latency', 'energy',
                f'{scenario}: Latency vs Energy',
                output_dir / f'{safe_name}_latency_vs_energy.png'
            )
            
            # 3. Accuracy vs Energy (only if scenario has valid accuracy)
            if scenario_df['has_valid_accuracy'].any():
                create_scatter_plot(
                    scenario_df, 'accuracy', 'energy',
                    f'{scenario}: Accuracy vs Energy',
                    output_dir / f'{safe_name}_accuracy_vs_energy.png'
                )


def generate_summary_plots(df: pd.DataFrame, output_dir: Path):
    """Generate summary plots averaging metrics per model across valid scenarios."""
    # Filter to only scenarios with valid accuracy for summary
    valid_df = df[df['has_valid_accuracy']].copy()
    
    if valid_df.empty:
        print("No valid accuracy data for summary plots")
        return
    
    # Average metrics per model
    summary_df = valid_df.groupby('model').agg({
        'latency': 'mean',
        'accuracy': 'mean',
        'energy': 'mean'
    }).reset_index()
    
    # 1. Latency vs Accuracy
    create_scatter_plot(
        summary_df, 'latency', 'accuracy',
        'Summary: Average Latency vs Accuracy',
        output_dir / 'summary_latency_vs_accuracy.png'
    )
    
    # Check if energy data is available
    has_energy = summary_df['energy'].notna().any()
    
    if has_energy:
        # 2. Latency vs Energy
        create_scatter_plot(
            summary_df, 'latency', 'energy',
            'Summary: Average Latency vs Energy',
            output_dir / 'summary_latency_vs_energy.png'
        )
        
        # 3. Accuracy vs Energy
        create_scatter_plot(
            summary_df, 'accuracy', 'energy',
            'Summary: Average Accuracy vs Energy',
            output_dir / 'summary_accuracy_vs_energy.png'
        )


def main(log_path: Path) -> Path:
    """Main entry point - parse log, load results, generate plots."""
    log_path = Path(log_path)
    
    # Create output directory next to the log file
    output_dir = log_path.parent / f"{log_path.stem}_graphs"
    output_dir.mkdir(exist_ok=True)
    
    # Parse log and load results
    result_dirs = parse_suite_log(log_path)
    if not result_dirs:
        print(f"No result directories found in {log_path}")
        return output_dir
    
    df = load_results(result_dirs)
    if df.empty:
        print("No results loaded")
        return output_dir
    
    print(f"Loaded {len(df)} results from {len(result_dirs)} directories")
    
    # Generate plots
    generate_scenario_plots(df, output_dir)
    generate_summary_plots(df, output_dir)
    
    print(f"Graphs saved to: {output_dir}")
    return output_dir


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_graphs.py <suite_log_path>")
        sys.exit(1)
    main(Path(sys.argv[1]))
