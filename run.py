# run.py
import os
import hydra
from omegaconf import DictConfig
from core.runner import run_benchmark
from core.logging_setup import setup_logging

# Set tokenizers parallelism to false to avoid fork warnings when subprocesses are created
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg) 
    run_benchmark(cfg)

if __name__ == "__main__":
    main()
