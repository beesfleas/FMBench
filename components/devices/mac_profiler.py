import time
from .base import BaseDeviceProfiler

class MacProfiler(BaseDeviceProfiler):
    """
    (Placeholder) Profiler for macOS devices (Intel and Apple Silicon).
    
    This profiler would be responsible for:
    1. Collecting CPU/RAM info from psutil (the safe parts).
    2. Parsing the 'powermetrics' command-line tool for GPU and
       energy data on Apple Silicon.
    
    For now, it is a placeholder and collects nothing.
    """
    def __init__(self, config):
        super().__init__(config)
        print("Initialized macOS Profiler (Placeholder).")
        self.samples = []

    def _monitor_process(self):
        print("macOS monitoring is a placeholder. Sleeping...")
        while self._is_monitoring:
            time.sleep(1.0)

    def get_metrics(self):
        return {
            "device_name": "macOS (Placeholder)",
            "error": "This profiler is a placeholder and collected no data."
        }