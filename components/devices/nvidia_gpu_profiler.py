import time
import threading
from .base import BaseDeviceProfiler
import pynvml


class NvidiaGpuProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA GPUs using pynvml (nvidia-smi).
    
    Monitors:
    - VRAM (Video RAM) usage
    - Power consumption
    - GPU Temperature
    - GPU Utilization
    """
    def __init__(self, config):
        super().__init__(config)
        if pynvml is None:
            raise ImportError("pynvml library not installed.")
        
        try:
            pynvml.nvmlInit()
            self.device_index = self.config.get("cuda_device", 0)
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            print(f"Monitoring GPU: {pynvml.nvmlDeviceGetName(self.handle)}")
        except pynvml.NVMLError as e:
            print(f"Failed to initialize pynvml: {e}")
            raise
            
        self.metrics = {
            "peak_memory_mb": 0.0,
            "peak_power_watts": 0.0,
            "peak_temp_c": 0.0,
            "average_utilization_percent": 0.0,
            "measurements": 0
        }

    def _monitor_process(self):
        """
        The core monitoring loop for NVIDIA GPU.
        Runs in a separate thread.
        """
        while self._is_monitoring:
            try:
                # Get Power (in milliwatts, convert to watts)
                power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                
                # Get Temperature
                temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Get Memory (in bytes, convert to MB)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_mb = memory_info.used / (1024 * 1024)
                
                # Get Utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                util_percent = util_info.gpu

                # Update metrics
                self.metrics["peak_memory_mb"] = max(self.metrics["peak_memory_mb"], memory_mb)
                self.metrics["peak_power_watts"] = max(self.metrics["peak_power_watts"], power_watts)
                self.metrics["peak_temp_c"] = max(self.metrics["peak_temp_c"], temp_c)
                self.metrics["average_utilization_percent"] += util_percent
                self.metrics["measurements"] += 1

            except pynvml.NVMLError as e:
                print(f"NVMLError in monitoring thread: {e}")
                self._is_monitoring = False
            
            # Sample every 0.1 seconds
            time.sleep(0.1)

    # def get_metrics(self):
    #     """
    #     Returns the collected metrics.
    #     """
    #     if self.metrics["measurements"] > 0:
    #         self.metrics["average_utilization_percent"] /= self.metrics["measurements"]
        
    #     # Clean up transient values
    #     self.metrics.pop("measurements", None)
        
    #     return self.metrics

    def get_metrics(self):
        """
        Returns a copy of the collected metrics.
        """
        final_metrics = self.metrics.copy()
        measurements = final_metrics.get("measurements", 0)
        if measurements > 0:
            final_metrics["average_utilization_percent"] /= measurements
        

        final_metrics.pop("measurements", None)
        
        return final_metrics

    def stop_monitoring(self):
        """Stops monitoring and shuts down pynvml."""
        super().stop_monitoring()