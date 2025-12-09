# FMBench

A flexible benchmarking framework for evaluating foundation models across different hardware platforms. Supports LLMs, VLMs, and time-series models with automatic device detection and comprehensive performance profiling.

## Key Features

- **Rapid Model/Scenario Swapping**: Switch between models and scenarios with a single command-line flag—no code changes required
- **Comprehensive Test Suites**: Run extensive benchmark suites across multiple models and scenarios automatically
- **Easy Extensibility**: Add new models or scenarios by simply adding YAML config files to `conf/model/` or `conf/scenario/`
- **Cross-Platform**: Works on any home PC or server running Windows, macOS, or Linux, plus embedded devices like Raspberry Pi and NVIDIA Jetson
- **Performance Profiling**: Real-time monitoring of CPU, GPU, memory, power consumption, and thermal metrics

## Setup

```bash
git clone https://github.com/beesfleas/FMBench
cd FMBench/software
pip install -r requirements.txt
pip install bitsandbytes  # Optional: for quantization
```

**Requirements**: Python 3.8+

## Quick Start

```bash
# Basic run
python run.py

# Swap models instantly
python run.py model=qwen2.5
python run.py model=tinyllama
python run.py model=smolvlm

# Swap scenarios just as easily
python run.py model=qwen2.5 scenario=sentiment
python run.py model=qwen2.5 scenario=summarization

# Combine with options
python run.py model=tinyllama scenario=classification scenario.num_samples=100
```

> [!NOTE]
> To capture power metrics, `sudo` is required:
> - **Mac**: For `powermetrics` access
> - **Linux (AMD/Intel)**: For CPU power usage via RAPL

## Benchmark Suite

Run comprehensive test suites across multiple model/scenario combinations:

```bash
python benchmark_suite.py --summary-only  # Preview what will run
python benchmark_suite.py                  # Run with confirmation
python benchmark_suite.py -y               # Skip confirmation
```

### Device Level Filtering

```bash
python benchmark_suite.py --device-level SoC     # ≤1B params (Raspberry Pi, etc.)
python benchmark_suite.py --device-level Mobile  # ≤3B params
python benchmark_suite.py --device-level Server  # No limit (default)
```

Edit `BENCHMARK_CONFIG` in `benchmark_suite.py` to define your test matrix.

### Auto-Generated Graphs

After running a suite, comparison graphs are automatically generated in `suite_logs/<log_name>_graphs/`:

**Summary Plots** (averaged across all scenarios):
- `summary_latency_vs_accuracy.png` - Model latency vs accuracy tradeoffs
- `summary_latency_vs_energy.png` - Latency vs energy consumption
- `summary_accuracy_vs_energy.png` - Accuracy vs energy efficiency

**Per-Scenario Plots**:
- `<scenario>_latency_vs_accuracy.png` - Per-scenario latency/accuracy
- `<scenario>_latency_vs_energy.png` - Per-scenario latency/energy
- `<scenario>_accuracy_vs_energy.png` - Per-scenario accuracy/energy

**Idle Power Table**:
- `idle_power_table.png` - Power consumption when model is loaded but idle

<!-- Example graphs (TODO: add actual images)
![Summary: Latency vs Accuracy](docs/example_summary_latency_vs_accuracy.png)
![Idle Power Table](docs/example_idle_power_table.png)
-->

## Available Models

### LLMs
| Model | Config File |
|-------|-------------|
| DistilGPT-2 | `distilgpt2` |
| TinyLlama | `tinyllama` |
| Llama 2 7B | `llama2-7b`, `llama2-7b-quantized` |
| Llama 2 13B | `llama2-13b` |
| Llama 3 1B | `llama3-1b` |
| Llama 3.2 1B | `llama3.2-1b`, `llama3.2-1b-quantized` |
| Llama 3.2 3B | `llama3.2-3b`, `llama3.2-3b-quantized` |
| Qwen 2.5 | `qwen2.5`, `qwen2.5-1.5b`, `qwen2.5-1.5b-quantized`, `qwen2.5-7b`, `qwen2.5-7b-quantized` |
| Qwen 3 | `qwen3-0.6b`, `qwen3-0.6b-quantized`, `qwen3-4b`, `qwen3-4b-quantized`, `qwen3-8b`, `qwen3-8b-quantized` |
| Falcon 7B | `falcon-7b`, `falcon-7b-quantized` |
| SmolLM2 | `smollm2` |
| DeepSeek V3 | `deepseek-v3` |

### VLMs (Vision-Language Models)
| Model | Config File |
|-------|-------------|
| SmolVLM | `smolvlm` |
| LLaVA | `llava` |
| Llama Vision | `llama-vision` |
| MiniCPM-V | `minicpm-v` |
| Moondream2 | `moondream2` |
| Molmo | `molmo` |
| Qwen 2.5 VL | `qwen2.5-vl` |

### Time-Series Models
| Model | Config File |
|-------|-------------|
| TimeGPT-1 | `timegpt1` |
| Chronos T5 | `chronos-t5-tiny`, `chronos-t5-small` |
| PatchTST | `patchtst` |
| MOIRAI | `moirai-small` |
| Moment | `moment-1-large` |
| ARIMA | `arima` |

## Available Scenarios

### LLM Scenarios
| Scenario | Description |
|----------|-------------|
| `sentiment` | Sentiment analysis on IMDB |
| `summarization` | Abstractive summarization (CNN/DailyMail) |
| `ner` | Named Entity Recognition (WikiANN) |
| `classification` | Text classification (AG News) |
| `translation` | German→English translation (WMT16) |
| `arc_challenge` | AI2 Reasoning Challenge (hard) |
| `arc_easy` | AI2 Reasoning Challenge (easy) |
| `perplexity_c4` | Perplexity on C4 dataset |
| `perplexity_wikitext2` | Perplexity on WikiText-2 |

### VLM Scenarios
| Scenario | Description |
|----------|-------------|
| `countbenchqa` | Visual counting QA |
| `docvqa` | Document OCR QA |
| `vqa` | Visual Question Answering v2 |
| `gtsrb` | German Traffic Sign Recognition |
| `hagrid` | Hand Gesture Recognition |

### Time-Series Scenarios
| Scenario | Description |
|----------|-------------|
| `fev_bench` | Forecasting evaluation benchmark |
| `gift_eval` | Zero-shot forecasting |
| `m3_monthly` | M3 competition monthly forecasting |

### Utility Scenarios
| Scenario | Description |
|----------|-------------|
| `idle` | Idle power measurement |
| `simple_llm` | Basic LLM test |
| `simple_vlm` | Basic VLM test |
| `simple_timeseries` | Basic time-series test |

## Adding New Models/Scenarios

Simply add a YAML config file:

**New Model** (`conf/model/my-model.yaml`):
```yaml
model_id: organization/model-name
model_category: LLM  # or VLM, TIME_SERIES
max_tokens: 64
```

**New Scenario** (`conf/scenario/my-scenario.yaml`):
```yaml
name: my_scenario
type: llm  # or vlm, timeseries
num_samples: 100
```

## Device Support

| Platform | Device Types |
|----------|--------------|
| **Desktop/Server** | Windows, macOS, Linux (CPU, NVIDIA GPU, Apple Silicon) |
| **Embedded** | NVIDIA Jetson (Orin, Xavier, Nano), Raspberry Pi |

### Manual Device Selection
```bash
python run.py device=cpu      # Force CPU
python run.py device=gpu      # Force NVIDIA GPU
python run.py device=mac      # Force Apple Silicon MPS
python run.py device=jetson   # NVIDIA Jetson
python run.py device=pi       # Raspberry Pi
```

### NVIDIA Jetson Setup
```bash
sudo -H pip install -U jetson-stats  # Required for monitoring
```

## Configuration Options

```bash
# Quantization (4-bit or 8-bit)
python run.py model=qwen2.5 model.quantization=4

# Logging
python run.py log_level=DEBUG save_logs=true

# Profiling interval
python run.py sampling_interval=0.5

# Number of samples
python run.py scenario.num_samples=100
```

## Output

Results include device metrics (CPU, GPU, memory, power, thermal), model performance (tokens/sec, latency, TTFT), and scenario evaluation scores. Use `save_logs=true` to save detailed logs.
