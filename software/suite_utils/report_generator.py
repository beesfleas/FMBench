"""LaTeX report generation for FMBench."""
import pandas as pd
from pathlib import Path
from suite_config import SCENARIO_CATEGORIES

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
