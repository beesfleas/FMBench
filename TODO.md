fm_bench/
├── __init__.py
├── run.py                 # Hydra-decorated main entry point (@hydra.main)
|
├── core/
│   ├── __init__.py
│   ├── runner.py          # The Benchmark Runner
│   └── registry.py        # Discovers and registers components
|
├── components/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── base.py        # Abstract base class
│   │   └── huggingface.py # Formerly hf_loader.py
│   │
│   ├── devices/
│   │   ├── base.py        # Abstract base class
│   │   ├── local_cpu.py   # Formerly cpu_bench.py
│   │   ├── nvidia_gpu.py  # Formerly gpu_bench.py
│   │   └── jetson.py
│   │
│   └── scenarios/         
│       ├── base.py
│       ├── helm/
│       │   ├── __init__.py
│       │   └── adapter.py
│       └── reasoning/
│           ├── __init__.py
│           └── mmlu.py
|
└── reporting/
    ├── base.py
    ├── csv_reporter.py
    └── json_reporter.py

conf/
├── config.yaml
│
├── model/
│   └── llama2-7b.yaml
│
├── device/
│   └── cpu.yaml
│
└── scenario/
    └── mmlu_ethics.yaml

setup.py
README.md

TO-DOs
1. load up llama 
2. Devices --> Need to be able to profile the device and get metrics on it
3. Get Benchmmarking/Testing by loaindg up HELM and other reasoning stuff from online
4. make modular by allowing it to load whatever model
5. get it to work on any device

Hydra --> config tool useful from Meta 