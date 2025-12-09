#!/usr/bin/env python3
"""Generate graphs from FMBench suite results."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

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

# Scenario Categories for grouping in report
SCENARIO_CATEGORIES = {
    "LLM Scenarios": {
        "ARC Easy", "ARC Challenge", "classification", "ner", 
        "perplexity_c4", "perplexity_wikitext2", "sentiment", 
        "summarization", "translation"
    },
    "VLM Scenarios": {
        "HaGRID", "GTSRB", "CountBenchQA", "docvqa", "VQAv2"
    },
    "Time-Series Scenarios": {
        "FEV-Bench", "GIFT-EVAL", "M3 Monthly Forecasting"
    },
    "Baseline Scenarios": {
        "Idle Baseline"
    }
}


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


def generate_latex_table(df: pd.DataFrame, output_dir: Path, device_info: dict, timestamp: str) -> None:
    """Generate a valid LaTeX file containing tables for all scenarios."""
    latex_path = output_dir / "results.tex"
    
    # Columns map: internal_col -> (Display Name, decimal_places)
    # The order here determines column order in LaTeX
    columns_map = {
        'model': ('Model', None),
        'latency': ('Latency (s)', 2),
        'ttft': ('TTFT (s)', 3),
        'tokens_per_output': ('Tok/Out', 1),
        'accuracy': ('Accuracy', 2),
        'perplexity': ('Perplexity', 2),
        'sMAPE': ('sMAPE (\\%)', 2),
        'energy': ('Energy (J)', 1),
        'idle_power_watts': ('Idle Power (W)', 1),
        'cpu_util': ('CPU Util (\\%)', 1),
        'ram_usage': ('RAM (MB)', 0),
        'gpu_util': ('GPU Util (\\%)', 1),
        'vram_usage': ('VRAM (MB)', 0),
    }

    # Format device info
    device_str = " \\\\ ".join([f"\\textbf{{{k}}}: {v}" for k, v in device_info.items() if v])

    # Header
    content = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage{adjustbox}",  # Added for table scaling
        r"\geometry{margin=1in}",
        r"\begin{document}",
        r"\title{FMBench Results}",
        f"\\date{{{device_str} \\\\[1ex] Generated: {timestamp}}}",
        r"\maketitle",
        r"",
    ]

    # -------------------------------------------------------------------------
    # Benchmark Summary Page
    # -------------------------------------------------------------------------
    content.append(r"\section*{Benchmark Summary}")
    
    # Use minipage to put tables side-by-side
    content.append(r"\noindent")
    content.append(r"\begin{minipage}[t]{0.45\textwidth}")
    
    # 1. Models Table
    content.append(r"\subsection*{Evaluated Models}")
    content.append(r"\begin{table}[H]")
    content.append(r"\centering")
    content.append(r"\begin{tabular}{l}")
    content.append(r"\toprule")
    content.append(r"\textbf{Model Name} \\")
    content.append(r"\midrule")
    
    unique_models = sorted(df['model'].unique())
    for model in unique_models:
        safe_model = str(model).replace("_", "\\_")
        content.append(f"{safe_model} \\\\")
        
    content.append(r"\bottomrule")
    content.append(r"\end{tabular}")
    content.append(r"\end{table}")
    content.append(r"\end{minipage}")
    content.append(r"\hfill")
    content.append(r"\begin{minipage}[t]{0.50\textwidth}")

    # 2. Scenarios Table
    content.append(r"\subsection*{Scenario Configurations}")
    content.append(r"\begin{table}[H]")
    content.append(r"\centering")
    content.append(r"\begin{adjustbox}{max width=\textwidth}")
    # Use p column for flags to allow line breaking
    content.append(r"\begin{tabular}{l p{5cm}}")
    content.append(r"\toprule")
    content.append(r"\textbf{Scenario} & \textbf{Flags} \\")
    content.append(r"\midrule")
    
    # Unique scenarios+flags
    unique_scenarios = df[['scenario', 'flags']].drop_duplicates().sort_values(['scenario'])
    
    for _, row in unique_scenarios.iterrows():
        scenario = str(row['scenario']).replace("_", "\\_").replace("%", "\\%")
        # Replace comma separator with newline for LaTeX p-column
        flags = str(row['flags']).replace("_", "\\_").replace("%", "\\%").replace(", ", r" \newline ")
        content.append(f"{scenario} & {flags} \\\\")
        
    content.append(r"\bottomrule")
    content.append(r"\end{tabular}")
    content.append(r"\end{adjustbox}")
    content.append(r"\end{table}")
    content.append(r"\end{minipage}")
    
    content.append(r"\newpage")
    # -------------------------------------------------------------------------

    # Get all scenarios present in data
    present_scenarios = set(df['scenario'].unique())
    
    # Process categories in specific order
    category_order = ["LLM Scenarios", "VLM Scenarios", "Time-Series Scenarios", "Baseline Scenarios"]
    
    # Track which scenarios have been processed to handle any uncategorized ones
    processed_scenarios = set()

    for category in category_order:
        # Find scenarios in this category that are present in the data
        cat_scenarios = SCENARIO_CATEGORIES.get(category, set())
        scenarios_to_report = sorted(list(cat_scenarios.intersection(present_scenarios)))
        
        if not scenarios_to_report:
            continue
            
        content.append(f"\\section*{{{category}}}")
        
        for scenario in scenarios_to_report:
            processed_scenarios.add(scenario)
            scenario_df = df[df['scenario'] == scenario].copy()
            safe_scenario = scenario.replace("_","\\_").replace("%","\\%")
            
            if scenario_df.empty:
                continue
                
            content.append(f"\\subsection*{{{safe_scenario}}}")
            content.append(r"\begin{table}[H]")
            content.append(r"\centering")
            
            # Determine valid columns (not all None)
            valid_cols = []
            for col in columns_map:
                # Check if this column has ANY non-null data in this scenario
                if col == 'model' or scenario_df[col].notna().any():
                    valid_cols.append(col)
            
            content.append(r"\begin{adjustbox}{max width=\textwidth}")
            content.append(r"\begin{tabular}{" + "l" + "c" * (len(valid_cols)-1) + "}")
            content.append(r"\toprule")
    
            # Build Header Row
            headers = [columns_map[c][0] for c in valid_cols]
            content.append(" & ".join(headers) + r" \\")
            content.append(r"\midrule")
            
            # Build Rows
            for _, row in scenario_df.iterrows():
                row_items = []
                for col in valid_cols:
                    val = row.get(col)
                    if val is None or pd.isna(val):
                        row_items.append("-")
                    elif col == 'model':
                        row_items.append(str(val).replace("_", "\\_"))
                    else:
                        precision = columns_map[col][1]
                        try:
                            fval = float(val)
                            if precision == 0:
                                 row_items.append(f"{int(fval)}")
                            else:
                                 row_items.append(f"{fval:.{precision}f}")
                        except (ValueError, TypeError):
                             row_items.append(str(val))
                
                content.append(" & ".join(row_items) + r" \\")
                
            content.append(r"\bottomrule")
            content.append(r"\end{tabular}")
            content.append(r"\end{adjustbox}")
            content.append(f"\\caption{{Results for {safe_scenario}}}")
            content.append(r"\end{table}")
            content.append(r"")

    # Handle uncategorized scenarios
    uncategorized = sorted(list(present_scenarios - processed_scenarios))
    if uncategorized:
        content.append(r"\section*{Other Scenarios}")
        for scenario in uncategorized:
            scenario_df = df[df['scenario'] == scenario].copy()
            safe_scenario = scenario.replace("_","\\_").replace("%","\\%")
            
            if scenario_df.empty:
                continue
                
            content.append(f"\\subsection*{{{safe_scenario}}}")
            content.append(r"\begin{table}[H]")
            content.append(r"\centering")
            
            # Determine valid columns
            valid_cols = []
            for col in columns_map:
                if col == 'model' or scenario_df[col].notna().any():
                    valid_cols.append(col)
            
            content.append(r"\begin{adjustbox}{max width=\textwidth}")
            content.append(r"\begin{tabular}{" + "l" + "c" * (len(valid_cols)-1) + "}")
            content.append(r"\toprule")
            headers = [columns_map[c][0] for c in valid_cols]
            content.append(" & ".join(headers) + r" \\")
            content.append(r"\midrule")
            
            for _, row in scenario_df.iterrows():
                row_items = []
                for col in valid_cols:
                    val = row.get(col)
                    if val is None or pd.isna(val):
                        row_items.append("-")
                    elif col == 'model':
                        row_items.append(str(val).replace("_", "\\_"))
                    else:
                        precision = columns_map[col][1]
                        try:
                            fval = float(val)
                            if precision == 0:
                                 row_items.append(f"{int(fval)}")
                            else:
                                 row_items.append(f"{fval:.{precision}f}")
                        except (ValueError, TypeError):
                             row_items.append(str(val))
                content.append(" & ".join(row_items) + r" \\")
                
            content.append(r"\bottomrule")
            content.append(r"\end{tabular}")
            content.append(r"\end{adjustbox}")
            content.append(f"\\caption{{Results for {safe_scenario}}}")
            content.append(r"\end{table}")
            content.append(r"")
    
    content.append(r"\end{document}")
    
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
        
    print(f"LaTeX tables saved to: {latex_path}")


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
    parser = argparse.ArgumentParser(description="Generate graphs from FMBench results.")
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
            output_dir = first_input.parent / f"{safe_stem}_graphs"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # print(f"Devices: {device_info}")
    
    # Generate plots
    generate_scenario_plots(df, output_dir, device_info, timestamp)
    generate_summary_plots(df, output_dir, device_info, timestamp)
    generate_idle_power_table(df, output_dir, device_info, timestamp)
    generate_latex_table(df, output_dir, device_info, timestamp)
    
    print(f"Graphs saved to: {output_dir}")


if __name__ == '__main__':
    main()
