"""Configuration constants for FMBench."""

# Device capability limits (in billions of parameters)
DEVICE_LIMITS = {
    "SoC": 1.0,
    "Mobile": 3.0,
    "Server": float("inf"),  # No limit
}

# Known model parameter counts (for models without size in name)
KNOWN_MODELS = {
    "smolvlm": 0.256,
    "tinyllama": 0.638,
    "distilgpt2": 0.082,
    "chronos-t5-tiny": 0.020,
    "chronos-t5-small": 0.046,
    "arima": 0.0,
    "llava": 7.0,
    "minicpm-v": 2.4,
    "patchtst": 0.0,
}

# Benchmark Configuration
BENCHMARK_CONFIG = {
    "LLM": {
        "models": [
            "qwen3-0.6b",
            "llama3.2-1b",
            "llama3.2-1b-quantized",
            "qwen2.5-1.5b",
            "qwen2.5-1.5b-quantized",
            "qwen3-4b",
            "qwen3-4b-quantized",
            "qwen2.5-7b",
            "qwen2.5-7b-quantized",
        ],
        "scenarios": {
            "idle": {"scenario.idle_duration": "10", "_skip_num_samples": True},
            "arc_easy": {},
            "arc_challenge": {},
            "classification": {},
            "ner": {},
            "perplexity_c4": {},
            "perplexity_wikitext2": {},
            "sentiment": {},
            "summarization": {},
            "translation": {},
        },
    },
    "VLM": {
        "models": ["smolvlm", "llava"],
        "scenarios": {
            "idle": {"scenario.idle_duration": "10", "_skip_num_samples": True},
            "hagrid": {},
            "gtsrb": {},
            "countbenchqa": {},
            "docvqa": {},
            "vqa": {},
        },
    },
    "TIME_SERIES": {
        "models": ["patchtst", "chronos-t5-small", "arima"],
        "scenarios": {
            "idle": {"scenario.idle_duration": "10", "_skip_num_samples": True},
            "fev_bench": {"_skip_num_samples": True},
            "gift_eval": {"_skip_num_samples": True},
            "m3_monthly": {},
        },
    },
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

DEVICE_LEVEL = "Server"  # Default: "SoC", "Mobile", or "Server"
GLOBAL_SETTINGS = {}
DEFAULT_NUM_SAMPLES = "10"
