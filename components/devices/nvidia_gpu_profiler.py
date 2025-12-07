import time
import os
import csv
import tempfile
from .base import BaseDeviceProfiler
import pynvml
import logging

log = logging.getLogger(__name__)

class NvidiaGpuProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA GPUs using pynvml (nvidia-smi).
    Writes samples to CSV and calculates metrics in real-time.
    """
    def __init__(self, config, device_index: int, profiler_manager=None):
        super().__init__(config)
        self.profiler_manager = profiler_manager
        if pynvml is None:
            raise ImportError("pynvml library not installed.")
        try:
            # pynvml.nvmlInit() # should be initialized in manager
            self.device_index = device_index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            self.device_name = f"{device_name} (GPU {self.device_index})"
        except pynvml.NVMLError as e:
            log.error(f"Failed to initialize pynvml for GPU {device_index}: {e}")
            raise
        
        self.sampling_interval = config.get("gpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        self.csv_filepath = None
        
        # Cached metrics
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        
        self.power_available = False
        self.temp_available = False
        self.memory_available = False
        self.util_available = False
        
        log.info(f"Initialized Nvidia GPU Profiler for {self.device_name}")

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
        Collect GPU metrics and write to CSV.
        """
        # Setup CSV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        self.csv_filepath = os.path.join(temp_dir, f"nvidia_gpu_profiler_gpu{self.device_index}_{timestamp}.csv")
        self.metrics["csv_filepath"] = self.csv_filepath
        
        log.info(f"Writing NVIDIA GPU {self.device_index} samples to: {self.csv_filepath}")
        
        start_time = time.perf_counter()
        csv_file = None
        csv_writer = None
        
        # Initialize stats
        stats = {
            "power": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf'), "nonzero_count": 0, "nonzero_sum": 0.0, "nonzero_min": float('inf'), "nonzero_max": 0.0},
            "temp": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf')},
            "memory": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf')},
            "util": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf'), "nonzero_count": 0, "nonzero_sum": 0.0, "nonzero_min": float('inf'), "nonzero_max": 0.0}
        }
        total_energy_joules = 0.0
        
        try:
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                
                sample = {"timestamp": rel_timestamp}
                
                # Power
                if self.power_available:
                    try:
                        power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                        sample["power_watts"] = power_watts
                        total_energy_joules += power_watts * self.sampling_interval
                        
                        s = stats["power"]
                        s["count"] += 1
                        s["sum"] += power_watts
                        s["max"] = max(s["max"], power_watts)
                        s["min"] = min(s["min"], power_watts)
                        if power_watts != 0:
                            s["nonzero_count"] += 1
                            s["nonzero_sum"] += power_watts
                            s["nonzero_max"] = max(s["nonzero_max"], power_watts)
                            s["nonzero_min"] = min(s["nonzero_min"], power_watts)
                            
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error(f"Could not read GPU power: {e}. Disabling power monitoring.")
                        self.power_available = False

                # Temperature
                if self.temp_available:
                    try:
                        temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        sample["temp_c"] = temp_c
                        
                        s = stats["temp"]
                        s["count"] += 1
                        s["sum"] += temp_c
                        s["max"] = max(s["max"], temp_c)
                        s["min"] = min(s["min"], temp_c)
                        
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error(f"Could not read GPU temp: {e}. Disabling temp monitoring.")
                        self.temp_available = False

                # Memory
                if self.memory_available:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                        memory_mb = memory_info.used / (1024 * 1024)
                        sample["memory_mb"] = memory_mb
                        
                        s = stats["memory"]
                        s["count"] += 1
                        s["sum"] += memory_mb
                        s["max"] = max(s["max"], memory_mb)
                        s["min"] = min(s["min"], memory_mb)
                        
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error(f"Could not read GPU memory: {e}. Disabling memory monitoring.")
                        self.memory_available = False

                # Utilization
                if self.util_available:
                    try:
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        util_percent = util_info.gpu
                        sample["utilization_percent"] = util_percent
                        
                        s = stats["util"]
                        s["count"] += 1
                        s["sum"] += util_percent
                        s["max"] = max(s["max"], util_percent)
                        s["min"] = min(s["min"], util_percent)
                        if util_percent != 0:
                            s["nonzero_count"] += 1
                            s["nonzero_sum"] += util_percent
                            s["nonzero_max"] = max(s["nonzero_max"], util_percent)
                            s["nonzero_min"] = min(s["nonzero_min"], util_percent)
                        
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error(f"Could not read GPU util: {e}. Disabling util monitoring.")
                        self.util_available = False
                
                # Write to CSV
                if csv_file is None:
                    try:
                        csv_file = open(self.csv_filepath, 'w', newline='')
                        csv_writer = csv.DictWriter(csv_file, fieldnames=sample.keys())
                        csv_writer.writeheader()
                    except Exception as e:
                        log.error(f"Failed to create CSV file {self.csv_filepath}: {e}")
                        csv_file = None
                        csv_writer = None
                
                if csv_writer is not None:
                    try:
                        csv_writer.writerow(sample)
                        csv_file.flush()
                    except Exception as e:
                        log.warning(f"Failed to write CSV sample: {e}")
                
                # Update cached metrics
                self.metrics["num_samples"] = max(stats["power"]["count"], stats["util"]["count"], stats["memory"]["count"])
                
                s = stats["power"]
                if s["nonzero_count"] > 0:
                    self.metrics["peak_power_watts"] = s["nonzero_max"]
                    self.metrics["average_power_watts"] = s["nonzero_sum"] / s["nonzero_count"]
                    self.metrics["min_power_watts"] = s["nonzero_min"]
                else:
                    self.metrics["peak_power_watts"] = 0
                    self.metrics["average_power_watts"] = 0
                    self.metrics["min_power_watts"] = 0
                self.metrics["total_energy_joules"] = total_energy_joules
                
                s = stats["temp"]
                if s["count"] > 0:
                    self.metrics["peak_temp_c"] = s["max"]
                    self.metrics["average_temp_c"] = s["sum"] / s["count"]
                    self.metrics["min_temp_c"] = s["min"]
                
                s = stats["memory"]
                if s["count"] > 0:
                    self.metrics["peak_memory_mb"] = s["max"]
                    self.metrics["average_memory_mb"] = s["sum"] / s["count"]
                    self.metrics["min_memory_mb"] = s["min"]
                
                s = stats["util"]
                if s["nonzero_count"] > 0:
                    self.metrics["peak_utilization_percent"] = s["nonzero_max"]
                    self.metrics["average_utilization_percent"] = s["nonzero_sum"] / s["nonzero_count"]
                    self.metrics["min_utilization_percent"] = s["nonzero_min"]
                else:
                    self.metrics["peak_utilization_percent"] = 0
                    self.metrics["average_utilization_percent"] = 0
                    self.metrics["min_utilization_percent"] = 0
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    # Use event.wait() instead of time.sleep() to allow immediate interruption
                    self._stop_event.wait(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()

    def stop_monitoring(self):
        """
        Stop the monitoring thread and clean up.
        Note: pynvml shutdown is handled by ProfilerManager to avoid double shutdown.
        """
        return super().stop_monitoring()
