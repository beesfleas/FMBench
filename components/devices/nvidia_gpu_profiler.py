import time
from .base import BaseDeviceProfiler
import pynvml
import logging

log = logging.getLogger(__name__)

class NvidiaGpuProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA GPUs using pynvml (nvidia-smi).
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
            self.device_name = device_name
        except pynvml.NVMLError as e:
            log.error(f"Failed to initialize pynvml: {e}")
            raise
        
        self.sampling_interval = config.get("gpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        self.samples = []
        self._start_time = None
        self.power_available = False
        self.temp_available = False
        self.memory_available = False
        self.util_available = False
        
        self._check_metric_availability()

    def get_device_info(self) -> str:
        """Return the device name set during initialization."""
        return self.device_name

    def _check_metric_availability(self):
        """
        Performs a test-read for each metric to set availability flags.
        """
        try:
            pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.power_available = True
        except pynvml.NVMLError:
            log.warning("Could not read GPU power. Disabling power monitoring.")
            
        try:
            pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            self.temp_available = True
        except pynvml.NVMLError:
            log.warning("Could not read GPU temperature. Disabling temp monitoring.")
            
        try:
            pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.memory_available = True
        except pynvml.NVMLError:
            log.warning("Could not read GPU memory. Disabling memory monitoring.")
            
        try:
            pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.util_available = True
        except pynvml.NVMLError:
            log.warning("Could not read GPU utilization. Disabling util monitoring.")

    def _monitor_process(self):
        """
        Lightweight monitoring loop - just collect raw data.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
        
        while self._is_monitoring:
            monitor_start_time = time.perf_counter()
            
            power_watts = None
            if self.power_available:
                try:
                    power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        log.error(f"Could not read GPU power: {e}. Disabling power monitoring.")
                    self.power_available = False

            temp_c = None
            if self.temp_available:
                try:
                    temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        log.error(f"Could not read GPU temp: {e}. Disabling temp monitoring.")
                    self.temp_available = False

            memory_mb = None
            if self.memory_available:
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    memory_mb = memory_info.used / (1024 * 1024)
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        log.error(f"Could not read GPU memory: {e}. Disabling memory monitoring.")
                    self.memory_available = False

            util_percent = None
            if self.util_available:
                try:
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    util_percent = util_info.gpu
                except pynvml.NVMLError as e:
                    if self._is_monitoring:
                        log.error(f"Could not read GPU util: {e}. Disabling util monitoring.")
                    self.util_available = False
            
            timestamp = time.perf_counter() - self._start_time
            
            self.samples.append({
                "timestamp": timestamp,
                "power_watts": power_watts,
                "temp_c": temp_c,
                "memory_mb": memory_mb,
                "utilization_percent": util_percent
            })
            
            elapsed = time.perf_counter() - monitor_start_time
            sleep_duration = self.sampling_interval - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def get_metrics(self):
        """
        Analyze raw samples after monitoring is complete.
        """
        if not self.samples:
            return {
                "device_name": self.device_name,
                "raw_samples": [],
                "error": "No samples collected"
            }
        
        power_values = [s["power_watts"] for s in self.samples if s["power_watts"] is not None]
        temp_values = [s["temp_c"] for s in self.samples if s["temp_c"] is not None]
        memory_values = [s["memory_mb"] for s in self.samples if s["memory_mb"] is not None]
        util_values = [s["utilization_percent"] for s in self.samples if s["utilization_percent"] is not None]
        
        num_samples = len(self.samples)
        if num_samples > 1:
            monitoring_duration = self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]
        else:
            monitoring_duration = 0.0
        
        metrics = {
            "device_name": self.device_name,
            # "raw_samples": self.samples,
            "num_samples": num_samples,
            "monitoring_duration_seconds": monitoring_duration,
            "sampling_interval": self.sampling_interval,
        }
        
        if power_values:
            metrics.update({
                "peak_power_watts": max(power_values),
                "average_power_watts": sum(power_values) / len(power_values),
                "min_power_watts": min(power_values),
            })
            total_energy_joules = 0.0
            for i in range(1, num_samples):
                if self.samples[i]["power_watts"] is not None and self.samples[i-1]["power_watts"] is not None:
                    time_delta = self.samples[i]["timestamp"] - self.samples[i-1]["timestamp"]
                    avg_power = (self.samples[i]["power_watts"] + self.samples[i-1]["power_watts"]) / 2.0
                    total_energy_joules += avg_power * time_delta
            
            metrics.update({
                "total_energy_joules": total_energy_joules,
            })

        if temp_values:
            metrics.update({
                "peak_temp_c": max(temp_values),
                "average_temp_c": sum(temp_values) / len(temp_values),
                "min_temp_c": min(temp_values),
            })
            
        if memory_values:
            metrics.update({
                "peak_memory_mb": max(memory_values),
                "average_memory_mb": sum(memory_values) / len(memory_values),
                "min_memory_mb": min(memory_values),
            })
            
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
        metrics = super().stop_monitoring()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            log.error(f"Error during pynvml shutdown: {e}")
        
        return metrics