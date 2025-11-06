import time
import threading
from .base import BaseDeviceProfiler
import pynvml


class NvidiaGpuProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA GPUs using pynvml (nvidia-smi).
    
    Stores raw timestamped metrics during monitoring.
    Analysis is deferred until get_metrics() is called.
    """
    def __init__(self, config):
        super().__init__(config)
        if pynvml is None:
            raise ImportError("pynvml library not installed.")
        try:
            pynvml.nvmlInit()
            self.device_index = config.get("cuda_device", 0)
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"Monitoring GPU: {device_name}")
        except pynvml.NVMLError as e:
            print(f"Failed to initialize pynvml: {e}")
            raise
        
        # Configurable sampling rate (default: 1.0 second)
        self.sampling_interval = config.get("sampling_interval", 1.0)
        
        # Store only raw samples - minimal overhead
        self.samples = []
        self._start_time = None

    def _monitor_process(self):
        """
        Lightweight monitoring loop - just collect raw data.
        No calculations or aggregations during monitoring.
        """
        self._start_time = time.time()
        
        while self._is_monitoring:
            try:
                current_time = time.time()
                
                # Collect raw metrics
                power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    self.handle, 
                    pynvml.NVML_TEMPERATURE_GPU
                )
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_mb = memory_info.used / (1024 * 1024)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                util_percent = util_info.gpu
                
                # Store raw sample with minimal processing
                self.samples.append({
                    "timestamp": current_time - self._start_time,
                    "power_watts": power_watts,
                    "temp_c": temp_c,
                    "memory_mb": memory_mb,
                    "utilization_percent": util_percent
                })

            except pynvml.NVMLError as e:
                print(f"NVMLError in monitoring thread: {e}")
                self._is_monitoring = False
            
            time.sleep(self.sampling_interval)

    def get_metrics(self):
        """
        Analyze raw samples after monitoring is complete.
        Returns both raw data and computed statistics.
        """
        if not self.samples:
            return {
                "raw_samples": [],
                "error": "No samples collected"
            }
        
        # Extract arrays for analysis
        power_values = [s["power_watts"] for s in self.samples]
        temp_values = [s["temp_c"] for s in self.samples]
        memory_values = [s["memory_mb"] for s in self.samples]
        util_values = [s["utilization_percent"] for s in self.samples]
        
        # Calculate statistics
        num_samples = len(self.samples)
        monitoring_duration = self.samples[-1]["timestamp"]
        
        # Energy calculation (trapezoidal integration for accuracy)
        total_energy_joules = 0.0
        for i in range(1, num_samples):
            time_delta = self.samples[i]["timestamp"] - self.samples[i-1]["timestamp"]
            avg_power = (self.samples[i]["power_watts"] + self.samples[i-1]["power_watts"]) / 2.0
            total_energy_joules += avg_power * time_delta
        
        metrics = {
            # Raw data for detailed analysis
            "raw_samples": self.samples,
            
            # Summary statistics
            "num_samples": num_samples,
            "monitoring_duration_seconds": monitoring_duration,
            "sampling_interval": self.sampling_interval,
            
            # Power statistics
            "peak_power_watts": max(power_values),
            "average_power_watts": sum(power_values) / num_samples,
            "min_power_watts": min(power_values),
            
            # Energy consumption
            "total_energy_joules": total_energy_joules,
            "total_energy_wh": total_energy_joules / 3600.0,
            
            # Temperature statistics
            "peak_temp_c": max(temp_values),
            "average_temp_c": sum(temp_values) / num_samples,
            
            # Memory statistics
            "peak_memory_mb": max(memory_values),
            "average_memory_mb": sum(memory_values) / num_samples,
            
            # Utilization statistics
            "peak_utilization_percent": max(util_values),
            "average_utilization_percent": sum(util_values) / num_samples,
        }
        
        return metrics

    def stop_monitoring(self):
        """
        Stop the monitoring thread and clean up.
        """
        super().stop_monitoring()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            print(f"Error during pynvml shutdown: {e}")
