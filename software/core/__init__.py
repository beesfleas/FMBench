# core/__init__.py
"""
Core FMBench functionality.
"""
from .runner import run_benchmark, run_scenario
from .logging_setup import setup_logging

__all__ = [
    "run_benchmark",
    "run_scenario",
    "setup_logging",
]
