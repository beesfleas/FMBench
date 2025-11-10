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
            self.device_name = device_name
        except pynvml.NVMLError as e:
            print(f"Failed to initialize pynvml: {e}")
            raise
        
        # Configurable sampling rate (default: 1.0 second)
        self.sampling_interval = config.get("gpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        
        # Store only raw samples - minimal overhead
        self.samples = []
        self._start_time = None
        
        # Add graceful failure flags, just like cpu_profiler.py
        self.power_available = True
        self.temp_available = True
        self.memory_available = True
        self.util_available = True

    def _monitor_process(self):
        """
        Lightweight monitoring loop - just collect raw data.
        No calculations or aggregations during monitoring.
        Matches cpu_profiler.py robustness with per-metric error handling.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        
        while self._is_monitoring:
            monitor_start_time = time.perf_counter()
            
            # --- 1. Power (Watts) ---
            power_watts = None
            if self.power_available:
                try:
                    power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        print(f"Warning: Could not read GPU power: {e}. Disabling power monitoring.")
                    self.power_available = False

            # --- 2. Temperature (Celsius) ---
            temp_c = None
            if self.temp_available:
                try:
                    temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        print(f"Warning: Could not read GPU temp: {e}. Disabling temp monitoring.")
                    self.temp_available = False

            # --- 3. Memory (MiB) ---
            memory_mb = None
            if self.memory_available:
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    memory_mb = memory_info.used / (1024 * 1024)
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        print(f"Warning: Could not read GPU memory: {e}. Disabling memory monitoring.")
                    self.memory_available = False

            # --- 4. Utilization (%) ---
            util_percent = None
            if self.util_available:
                try:
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    util_percent = util_info.gpu
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        print(f"Warning: Could not read GPU util: {e}. Disabling util monitoring.")
                    self.util_available = False
            
            # Get timestamp relative to the start
            timestamp = time.perf_counter() - self._start_time
            
            # Store raw sample with minimal processing
            self.samples.append({
                "timestamp": timestamp,
                "power_watts": power_watts,
                "temp_c": temp_c,
                "memory_mb": memory_mb,
                "utilization_percent": util_percent
            })
            
            # Use precise sleep to maintain sampling interval
            elapsed = time.perf_counter() - monitor_start_time
            sleep_duration = self.sampling_interval - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

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
        
        # Extract arrays for analysis, gracefully skipping None values
        power_values = [s["power_watts"] for s in self.samples if s["power_watts"] is not None]
        temp_values = [s["temp_c"] for s in self.samples if s["temp_c"] is not None]
        memory_values = [s["memory_mb"] for s in self.samples if s["memory_mb"] is not None]
        util_values = [s["utilization_percent"] for s in self.samples if s["utilization_percent"] is not None]
        
        # Calculate statistics
        num_samples = len(self.samples)
        monitoring_duration = self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]
        
        metrics = {
            # Device Info
            "device_name": self.device_name,
            
            # Raw data for detailed analysis
            "raw_samples": self.samples,
            
            # Summary statistics
            "num_samples": num_samples,
            "monitoring_duration_seconds": monitoring_duration,
            "sampling_interval": self.sampling_interval,
        }
        
        # --- Power statistics (only if available) ---
        if power_values:
            metrics.update({
                "peak_power_watts": max(power_values),
                "average_power_watts": sum(power_values) / len(power_values),
                "min_power_watts": min(power_values),
            })
            # Energy calculation (trapezoidal integration for accuracy)
            total_energy_joules = 0.0
            for i in range(1, num_samples):
                if self.samples[i]["power_watts"] is not None and self.samples[i-1]["power_watts"] is not None:
                    time_delta = self.samples[i]["timestamp"] - self.samples[i-1]["timestamp"]
                    avg_power = (self.samples[i]["power_watts"] + self.samples[i-1]["power_watts"]) / 2.0
                    total_energy_joules += avg_power * time_delta
            
            metrics.update({
                "total_energy_joules": total_energy_joules,
                "total_energy_wh": total_energy_joules / 3600.0,
            })

        # --- Temperature statistics (only if available) ---
        if temp_values:
            metrics.update({
                "peak_temp_c": max(temp_values),
                "average_temp_c": sum(temp_values) / len(temp_values),
                "min_temp_c": min(temp_values),
            })
            
        # --- Memory statistics (only if available) ---
        if memory_values:
            metrics.update({
                "peak_memory_mb": max(memory_values),
                "average_memory_mb": sum(memory_values) / len(memory_values),
                "min_memory_mb": min(memory_values),
            })
            
        # --- Utilization statistics (only if available) ---
        if util_values:
            metrics.update({
                "peak_utilization_percent": max(util_values),
                "average_utilization_percent": sum(util_values) / len(util_values),
                "min_utilization_percent": min(util_values),
            })
        
        return metrics

    def stop_monitoring(self):
        """
        Stop the monitoring thread and clean up.
        """
        # Get metrics *before* pynvml.nvmlShutdown() is called
        metrics = super().stop_monitoring()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            print(f"Error during pynvml shutdown: {e}")
        
        return metrics