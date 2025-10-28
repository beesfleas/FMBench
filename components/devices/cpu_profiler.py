import time
import psutil
import threading
from .base import BaseDeviceProfiler

class LocalCpuProfiler(BaseDeviceProfiler):
    """
    Profiler for local CPU and RAM using psutil.
    """
    def __init__(self, config):
        super().__init__(config)
        self.process = psutil.Process()
        self.metrics = {
            "peak_cpu_percent": 0.0,
            "peak_memory_mb": 0.0,
            "average_cpu_percent": 0.0,
            "measurements": 0
        }

    def _start_monitoring_thread(self):
        thread = threading.Thread(target=self._monitor_process, daemon=True)
        thread.start()
        return thread

    def _monitor_process(self):
        """
        The core monitoring loop for CPU/RAM.
        Runs in a separate thread.
        """
        while self._is_monitoring:
            try:
                # Get CPU percentage for the current process
                cpu_percent = self.process.cpu_percent(interval=None)
                
                # Get Memory usage for the current process
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024) # Resident Set Size in MB

                # Update metrics
                self.metrics["peak_cpu_percent"] = max(self.metrics["peak_cpu_percent"], cpu_percent)
                self.metrics["peak_memory_mb"] = max(self.metrics["peak_memory_mb"], memory_mb)
                self.metrics["average_cpu_percent"] += cpu_percent
                self.metrics["measurements"] += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._is_monitoring = False
                print("Monitoring process lost or access denied.")
            
            # Sample every 0.1 seconds
            time.sleep(0.1)

    def get_metrics(self):
        """
        Returns the collected metrics.
        """
        if self.metrics["measurements"] > 0:
            self.metrics["average_cpu_percent"] /= self.metrics["measurements"]
        
        # Clean up transient values
        self.metrics.pop("measurements", None)
        
        return self.metrics