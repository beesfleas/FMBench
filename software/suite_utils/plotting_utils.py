"""Plotting utilities for FMBench."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from suite_config import AXIS_LABELS

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
            'Summary: Average Accuracy vs Average Energy',
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
