# FMBench

FMBench is a flexible benchmarking framework for evaluating foundation models across different hardware platforms. It supports LLMs, VLMs, and time-series models with automatic device detection and comprehensive performance profiling.

## Features

- **Multi-Model Support**: LLMs, Vision-Language Models (VLMs), and Time-Series models
- **Cross-Platform**: Automatic device detection for CPU, CUDA (NVIDIA GPU), MPS (Apple Silicon), Jetson, and Raspberry Pi
- **Performance Profiling**: Real-time monitoring of CPU, GPU, memory, power consumption, and thermal metrics
- **Model Quantization**: 4-bit and 8-bit quantization support for memory-efficient inference
- **Flexible Configuration**: Hydra-based configuration system with command-line overrides
- **Scenario-Based Testing**: Support for custom benchmarking scenarios

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FMBench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For quantization support:
```bash
pip install bitsandbytes
```

## Quick Start

### Basic Usage

Run a benchmark with default settings:
```bash
python run.py
```

Run with a specific model:
```bash
python run.py model=qwen2.5
```

Run with a scenario:
```bash
python run.py model=qwen2.5 scenario=sentiment
```

## Available Scenarios

### Language & Knowledge (LLM)
- `sentiment`: Sentiment analysis on IMDB movie reviews
- `summarization`: Abstractive summarization using CNN/DailyMail
- `ner`: Named Entity Recognition on WikiANN
- `classification`: Text classification on AG News
- `translation`: Machine translation (German to English) on WMT16
- `arc_challenge` / `arc_easy`: AI2 Reasoning Challenge
- `mmlu`: Massive Multitask Language Understanding benchmark
- `helm`: Holistic Evaluation of Language Models
- `perplexity_c4` / `perplexity_wikitext2`: Perplexity evaluation
- `long_context`: Long context window evaluation

### Vision & Multimodal (VLM)
- `countbenchqa`: Visual counting question answering
- `docvqa`: Document Visual Question Answering
- `vqa`: Visual Question Answering v2
- `gtsrb`: German Traffic Sign Recognition
- `hagrid`: Hand Gesture Recognition
- `simple_vlm`: Basic VLM test scenario

### Time Series
- `ett`: Electricity Transformer Temperature forecasting
- `gifteval`: Zero-shot time series forecasting
- `m3_monthly`: M3 competition monthly forecasting
- `simple_timeseries`: Basic time series test scenario

## Available Flags

### Model Selection

```bash
# Select a model (see Available Models section)
python run.py model=<model_name>

# Examples:
python run.py model=distilgpt2
python run.py model=qwen2.5
python run.py model=tinyllama
```

### Scenario Selection

```bash
# Run with a scenario
python run.py scenario=<scenario_name>

# Examples:
python run.py scenario=sentiment
python run.py scenario=countbenchqa
python run.py scenario=ett
```

### Common Configuration Flags

```bash
# Number of samples to run
python run.py num_samples=100

# Set device
python run.py device=auto  # or cpu, gpu, mac, jetson, pi

# Model-specific options
python run.py model.loader_type=vllm
python run.py model.quantization=4  # or 8
python run.py model.max_tokens=128
python run.py model.device_preference=cuda  # or cpu, mps, auto

# Logging
python run.py log_level=DEBUG  # or INFO, WARNING, ERROR
python run.py save_logs=true

# Profiling
python run.py sampling_interval=0.5

# Scenario-specific options
python run.py scenario.slo_threshold=2  # For CountBenchQA SLO violation tracking
```

## Available Models

### LLMs (Large Language Models)
- `distilgpt2` - DistilGPT-2 (small, fast)
- `tinyllama` - TinyLlama 1.1B
- `qwen2.5` - Qwen2.5 1.5B Instruct
- `llama3-1b` - Llama 3 1B
- `smollm2` - SmolLM2 1.7B
- `llama2-7b` - Llama 2 7B
- `falcon-7b` - Falcon 7B
- `deepseek-v3` - DeepSeek V3

### VLMs (Vision-Language Models)
- `smolvlm` - SmolVLM 256M

### Time-Series Models
- `timegpt1` - TimeGPT-1

## Device Support

### Automatic Detection
By default, FMBench automatically detects and uses the best available device:
- **CUDA** (NVIDIA GPU) - if available
- **MPS** (Apple Silicon) - if on macOS with Apple Silicon
- **CPU** - fallback option

### Manual Device Selection
You can override device selection:
- `device=cpu` - Force CPU execution
- `device=gpu` - Force NVIDIA GPU (requires CUDA)
- `device=mac` - Force Apple Silicon MPS
- `device=jetson` - NVIDIA Jetson devices
- `device=pi` - Raspberry Pi devices

### Apple Silicon (MPS) Notes
- Small models (<1B parameters) work directly on MPS
- Large models (>1B parameters) require quantization or CPU:
  ```bash
  # Use quantization for large models on MPS
  python run.py model=qwen2.5 model.quantization=4
  
  # Or use CPU
  python run.py model=qwen2.5 model.device_preference=cpu
  ```
- **Sampling Behavior**: The `MacProfiler` relies on the system's `powermetrics` tool, which buffers output. This means that for very short benchmarks, you may see fewer samples than expected (e.g., 2 samples instead of 4 for a 2-second run with 0.5s interval) because samples are only recorded when `powermetrics` flushes its buffer. This is normal behavior and ensures data accuracy over longer runs.

### NVIDIA Jetson Setup

Running FMBench on NVIDIA Jetson (Orin, Xavier, Nano) requires specific setup:

1.  **System Monitoring**: Install `jetson-stats` to enable detailed power, GPU, and thermal monitoring:
    ```bash
    sudo -H pip install -U jetson-stats
    ```
    (You may need to restart your shell or reboot after installation)

2.  **PyTorch & Python**:
    - Python 3.8+ is supported (standard on JetPack 5.x/6.x).
    - FMBench automatically handles missing `torch.distributed` modules often found in Jetson PyTorch builds.

3.  **Running**:
    - Use `device=jetson` to force Jetson-specific profiling if auto-detection fails.
    ```bash
    python run.py device=jetson
    ```

## Model Quantization

Quantization reduces model memory usage, enabling larger models to run on limited hardware.

### Usage

**Command Line:**
```bash
# 4-bit quantization (reduces size by ~4x)
python run.py model=qwen2.5 model.quantization=4

# 8-bit quantization (reduces size by ~2x)
python run.py model=qwen2.5 model.quantization=8
```

**Config File:**
Add to model config file (e.g., `conf/model/qwen2.5.yaml`):
```yaml
model_id: Qwen/Qwen2.5-1.5B-Instruct
model_category: LLM
max_tokens: 64
quantization: 4  # 4-bit or 8-bit
```

### Requirements
- Requires `bitsandbytes`: `pip install bitsandbytes`
- Quantized models run on CPU (quantization doesn't support MPS device_map)
- Best for large models that exceed MPS memory limits

## Scenarios

Scenarios define structured benchmarking tasks:

- `simple_llm` - Basic LLM text generation tasks
- `simple_vlm` - Vision-language model tasks
- `simple_timeseries` - Time-series prediction tasks

## Configuration Files

FMBench uses Hydra for configuration management. Config files are located in `conf/`:

- `conf/config.yaml` - Main configuration
- `conf/model/` - Model configurations
- `conf/device/` - Device configurations
- `conf/scenario/` - Scenario configurations

You can override any config value via command line using dot notation:
```bash
python run.py model.max_tokens=128 device.type=cpu
```

## Output

Benchmark results are displayed in JSON format, including:
- Device metrics (CPU, GPU, memory, power, thermal)
- Model performance metrics
- Scenario evaluation results (if applicable)

Logs can be saved to files with `save_logs=true`.

## Examples

### Run a small model on auto-detected device
```bash
python run.py model=distilgpt2
```

### Run sentiment analysis
```bash
python run.py model=distilgpt2 scenario=sentiment num_samples=100
```

### Run VLM counting task with SLO tracking
```bash
python run.py model=smolvlm scenario=countbenchqa scenario.slo_threshold=2
```

### Run a large model with quantization
```bash
python run.py model=qwen2.5 model.quantization=4
```

### Run with custom settings and debug logging
```bash
python run.py model=tinyllama log_level=DEBUG save_logs=true
```

### Run a VLM with a scenario
```bash
python run.py model=smolvlm scenario=simple_vlm
```

### Run time series forecasting
```bash
python run.py model=timegpt1 scenario=ett num_samples=50
```

### Force CPU execution
```bash
python run.py model=qwen2.5 device=cpu
```

## Troubleshooting

### MPS Size Limit Error
If you see "Model too large for MPS" error:
- Use quantization: `model.quantization=4`
- Or use CPU: `model.device_preference=cpu`

### Missing bitsandbytes
If quantization is requested but fails:
```bash
pip install bitsandbytes
```

### CUDA Not Available
If CUDA is requested but not available:
- Use `device=cpu` or `device=auto`
- Check NVIDIA drivers and CUDA installation
