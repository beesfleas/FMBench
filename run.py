# run.py
import hydra
from omegaconf import DictConfig
from core.runner import run_benchmark

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_benchmark(cfg)

if __name__ == "__main__":
    main()
