import time
from .base import BaseDeviceProfiler

class PiProfiler(BaseDeviceProfiler):
    """
    (Placeholder) Profiler for Raspberry Pi devices.
    
    This profiler would be responsible for:
    1. Reading temperature from '/sys/class/thermal/thermal_zone0/temp'.
    2. Reading power/throttling info using 'vcgencmd'.
    3. Collecting CPU/RAM info from psutil.
    
    For now, it is a placeholder and collects nothing.
    """
    def __init__(self, config):
        super().__init__(config)
        print("Initialized Raspberry Pi Profiler (Placeholder).")
        self.samples = []

    def _monitor_process(self):
        print("Raspberry Pi monitoring is a placeholder. Sleeping...")
        while self._is_monitoring:
            time.sleep(1.0)

    def get_metrics(self):
        return {
            "device_name": "Raspberry Pi (Placeholder)",
            "error": "This profiler is a placeholder and collected no data."
        }