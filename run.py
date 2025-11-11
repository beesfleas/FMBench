# run.py
import hydra
from omegaconf import DictConfig
from core.runner import run_benchmark
from core.logging_setup import setup_logging

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_logging(cfg) 
    run_benchmark(cfg)

if __name__ == "__main__":
    main()
