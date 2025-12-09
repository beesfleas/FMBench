#!/usr/bin/env python3
"""Generate graphs from FMBench suite results."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Scenarios where accuracy is not meaningful
SKIP_ACCURACY_SCENARIOS = {'Idle Baseline', 'summarization', 'translation'}

# Axis labels with units
AXIS_LABELS = {
    'latency': 'Latency (seconds)',
    'accuracy': 'Accuracy',
    'energy': 'Energy (Joules)',
    'sMAPE': 'sMAPE (%)',
}

# Map profiler key patterns to device type labels
PROFILER_PATTERNS = {
    'nvidiagpuprofiler': 'GPU',
    'macprofiler': 'Mac',
    'jetsonprofiler': 'Jetson',
    'piprofiler': 'Raspberry Pi',
    'cpuprofiler': 'CPU',
}


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


def load_results(result_dirs: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Load summary.json from each result directory into a DataFrame.
    
    Returns:
        DataFrame with results and dict with device info.
    """
    records = []
    device_info = {}  # Will store device names by type
    
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
            'sMAPE': data.get('avg_sMAPE'),
        }
        
        # Extract energy, power, and device info from profilers
        device_metrics = data.get('device_metrics', {})
        energy = None
        idle_power = None
        
        for key, metrics in device_metrics.items():
            key_lower = key.lower()
            
            # Detect device type and store device name
            for pattern, label in PROFILER_PATTERNS.items():
                if pattern in key_lower:
                    if label not in device_info:
                        device_info[label] = metrics.get('device_name')
                    break
            
            # Extract energy (prefer GPU/accelerator over CPU)
            if 'cpuprofiler' not in key_lower:
                if metrics.get('total_energy_joules') is not None:
                    energy = metrics.get('total_energy_joules')
                # For idle scenario, capture average power
                if scenario == 'Idle Baseline':
                    idle_power = metrics.get('average_power_watts')
        
        record['energy'] = energy
        record['idle_power_watts'] = idle_power
        
        # Track if accuracy is meaningful for this scenario
        record['has_valid_accuracy'] = (
            record['accuracy'] is not None 
            and scenario not in SKIP_ACCURACY_SCENARIOS
            and not scenario.lower().startswith('perplexity')
        )

        # Track if sMAPE is available
        record['has_valid_smape'] = record['sMAPE'] is not None
        
        records.append(record)
    
    return pd.DataFrame(records), device_info




def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                        title: str, output_path: Path,
                        device_info: dict, timestamp: str):
    """Create a scatter plot with model labels, device info, and timestamp."""
    # Filter out rows with missing data for the columns we need
    plot_df = df.dropna(subset=[x_col, y_col])
    
    if plot_df.empty:
        return False
    
    plt.figure(figsize=(12, 10))  # Increased size slightly for better visibility
    
    # helper for clean model names (remove path/extensions if present)
    def clean_model_name(name):
        return str(name).split('/')[-1]

    plot_df = plot_df.assign(model_clean=plot_df['model'].apply(clean_model_name))
    
    # Use hue for different models to give them different colors
    # s=150 sets point size
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue='model_clean', s=150, style='model_clean', palette='deep')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(AXIS_LABELS.get(x_col, x_col), fontsize=12)
    plt.ylabel(AXIS_LABELS.get(y_col, y_col), fontsize=12)
    
    # Relocate legend if it's too big or obscuring
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Build footer text with device info and timestamp on one line
    footer_parts = [f"{k}: {v}" for k, v in device_info.items() if v]
    footer_parts.append(f"Generated: {timestamp}")
    footer_text = '  |  '.join(footer_parts)
    
    plt.figtext(0.02, 0.01, footer_text, ha='left', fontsize=8, style='italic', backgroundcolor='white')
    
    plt.tight_layout(rect=[0, 0.03, 0.85, 1])  # Adjust rect to make room for external legend
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def create_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                    title: str, output_path: Path,
                    device_info: dict, timestamp: str):
    """Create a bar plot for a single metric."""
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return False
        
    plt.figure(figsize=(10, 6))
    
    # Clean model names for display
    def clean_model_name(name):
        return str(name).split('/')[-1]
    
    # Use .assign or .loc to avoid copy warning
    plot_df = plot_df.assign(model_clean=plot_df['model'].apply(clean_model_name))
    
    # Create barplot
    # Fix future warning: assign x to hue and set legend=False
    sns.barplot(data=plot_df, x='model_clean', y=y_col, hue='model_clean', palette='viridis', legend=False)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(AXIS_LABELS.get(y_col, y_col), fontsize=12)
    
    # Rotate x labels to avoid overlap
    plt.xticks(rotation=45, ha='right')
    
    # Build footer text
    footer_parts = [f"{k}: {v}" for k, v in device_info.items() if v]
    footer_parts.append(f"Generated: {timestamp}")
    footer_text = '  |  '.join(footer_parts)
    
    plt.figtext(0.02, 0.01, footer_text, ha='left', fontsize=8, style='italic', backgroundcolor='white')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def generate_scenario_plots(df: pd.DataFrame, output_dir: Path,
                            device_info: dict, timestamp: str):
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
                output_dir / f'{safe_name}_latency_vs_accuracy.png',
                device_info, timestamp
            )
        
        # 2. sMAPE Bar Plot (if valid sMAPE)
        if scenario_df['has_valid_smape'].any():
            create_bar_plot(
                scenario_df, 'model', 'sMAPE',
                f'{scenario}: sMAPE by Model',
                output_dir / f'{safe_name}_smape_bar.png',
                device_info, timestamp
            )
            
            # Also Latency vs sMAPE scatter
            create_scatter_plot(
                scenario_df, 'latency', 'sMAPE',
                f'{scenario}: Latency vs sMAPE',
                output_dir / f'{safe_name}_latency_vs_smape.png',
                device_info, timestamp
            )
        
        # Check if energy data is available
        has_energy = scenario_df['energy'].notna().any()
        
        if has_energy:
            # 3. Latency vs Energy
            create_scatter_plot(
                scenario_df, 'latency', 'energy',
                f'{scenario}: Latency vs Energy',
                output_dir / f'{safe_name}_latency_vs_energy.png',
                device_info, timestamp
            )
            
            # 4. Accuracy/sMAPE vs Energy
            if scenario_df['has_valid_accuracy'].any():
                create_scatter_plot(
                    scenario_df, 'accuracy', 'energy',
                    f'{scenario}: Accuracy vs Energy',
                    output_dir / f'{safe_name}_accuracy_vs_energy.png',
                    device_info, timestamp
                )
            
            if scenario_df['has_valid_smape'].any():
                create_scatter_plot(
                    scenario_df, 'sMAPE', 'energy',
                    f'{scenario}: sMAPE vs Energy',
                    output_dir / f'{safe_name}_smape_vs_energy.png',
                    device_info, timestamp
                )


def generate_summary_plots(df: pd.DataFrame, output_dir: Path,
                           device_info: dict, timestamp: str):
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
        'Summary: Average Latency vs Average Accuracy',
        output_dir / 'summary_latency_vs_accuracy.png',
        device_info, timestamp
    )
    
    # Check if energy data is available
    has_energy = summary_df['energy'].notna().any()
    
    if has_energy:
        # 2. Latency vs Energy
        create_scatter_plot(
            summary_df, 'latency', 'energy',
            'Summary: Average Latency vs Average Energy',
            output_dir / 'summary_latency_vs_energy.png',
            device_info, timestamp
        )
        
        # 3. Accuracy vs Energy
        create_scatter_plot(
            summary_df, 'accuracy', 'energy',
            'Summary: Average Accuracy vs AverageEnergy',
            output_dir / 'summary_accuracy_vs_energy.png',
            device_info, timestamp
        )


def generate_idle_power_table(df: pd.DataFrame, output_dir: Path,
                               device_info: dict, timestamp: str):
    """Generate a table showing idle power usage per model."""
    # Filter to idle scenarios only
    idle_df = df[df['scenario'] == 'Idle Baseline'].copy()
    
    if idle_df.empty or idle_df['idle_power_watts'].isna().all():
        print("No idle power data available")
        return False
    
    # Select relevant columns and sort by power
    table_df = idle_df[['model', 'idle_power_watts']].dropna()
    table_df = table_df.sort_values('idle_power_watts')
    table_df.columns = ['Model', 'Idle Power (W)']
    
    if table_df.empty:
        return False
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(8, max(3, len(table_df) * 0.5 + 2)))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=[[row['Model'], f"{row['Idle Power (W)']:.2f}"] 
                  for _, row in table_df.iterrows()],
        colLabels=['Model', 'Idle Power (W)'],
        cellLoc='center',
        loc='center',
        colWidths=[0.6, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_df) + 1):
        color = '#D9E2F3' if i % 2 == 0 else 'white'
        for j in range(2):
            table[(i, j)].set_facecolor(color)
    
    # Title
    plt.title('Idle Power Usage by Model', fontsize=14, fontweight='bold', pad=20)
    
    # Footer with device info and timestamp on one line
    footer_parts = [f"{k}: {v}" for k, v in device_info.items() if v]
    footer_parts.append(f"Generated: {timestamp}")
    footer_text = '  |  '.join(footer_parts)
    plt.figtext(0.02, 0.02, footer_text, ha='left', fontsize=7, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_dir / 'idle_power_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def collect_result_dirs(inputs: list[str]) -> list[Path]:
    """Collect all result directories from provided inputs (logs or root dirs)."""
    result_dirs = []
    
    for inp in inputs:
        path = Path(inp)
        if not path.exists():
            print(f"Warning: Input {path} does not exist.")
            continue
            
        if path.is_file():
            # If it's a log file, parse it
            if path.suffix == '.log' or 'suite_' in path.name:
                print(f"Parsing log file: {path}")
                result_dirs.extend(parse_suite_log(path))
            else:
                print(f"Warning: Skipping file {path} (not a recognized log).")
        
        elif path.is_dir():
            # If it's a directory, look for summary.json recursively
            print(f"Scanning directory: {path}")
            found = list(path.rglob('summary.json'))
            print(f"  Found {len(found)} result(s).")
            # The result dir is the parent of summary.json
            result_dirs.extend([f.parent for f in found])
            
    # Remove duplicates and resolve paths
    unique_dirs = list(set([d.resolve() for d in result_dirs]))
    return unique_dirs


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate graphs from FMBench results.")
    parser.add_argument('inputs', nargs='+', help="Path to suite logs (.log) or result directories.")
    parser.add_argument('-o', '--output-dir', help="Directory to save graphs. Default: <first_input>_combined_graphs")
    
    args = parser.parse_args()
    
    # Collect all valid result directories
    result_dirs = collect_result_dirs(args.inputs)
    
    if not result_dirs:
        print("No result directories found from inputs.")
        return
        
    print(f"Total entries to process: {len(result_dirs)}")
    
    df, device_info = load_results(result_dirs)
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
            output_dir = first_input.parent / f"{safe_stem}_graphs"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    print(f"Devices: {device_info}")
    
    # Generate plots
    generate_scenario_plots(df, output_dir, device_info, timestamp)
    generate_summary_plots(df, output_dir, device_info, timestamp)
    generate_idle_power_table(df, output_dir, device_info, timestamp)
    
    print(f"Graphs saved to: {output_dir}")


if __name__ == '__main__':
    main()
