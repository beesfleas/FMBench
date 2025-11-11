import logging
import os
import time
from omegaconf import DictConfig
import sys

def setup_logging(cfg: DictConfig):
    """
    Configures the root logger based on the Hydra config.
    
    Feature 1: Saves all logs to a file in the 'logs/' directory.
    Feature 2: Controls the console log level via the 'log_level' config key.
    """
    # 1. Get log level from config
    log_level_str = cfg.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # 2. Create log directory
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, f"benchmark_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # 3. Get the root logger.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 4. Create formatters
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 5. Create File Handler
    root_logger.handlers = []
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG) # Log everything to the file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 6. Create Console/Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level) # Use level from config
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging configured. Console level: {log_level_str}, File logs at: {log_file_path}")