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
        
        # Metric accumulators
        power_values = []
        temp_values = []
        memory_values = []
        util_values = []
        
        try:
            while self._is_monitoring:
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                
                sample = {"timestamp": rel_timestamp}
                
                # Power
                if self.power_available:
                    try:
                        power_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                        sample["power_watts"] = power_watts
                    except pynvml.NVMLError as e:
                        if self._is_monitoring:
                            log.error(f"Could not read GPU power: {e}. Disabling power monitoring.")
                        self.power_available = False

                # Temperature
                if self.temp_available:
                    try:
                        temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        sample["temp_c"] = temp_c
                    except pynvml.NVMLError as e:
                        if self._is_monitoring:
                            log.error(f"Could not read GPU temp: {e}. Disabling temp monitoring.")
                        self.temp_available = False

                # Memory
                if self.memory_available:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                        memory_mb = memory_info.used / (1024 * 1024)
                        sample["memory_mb"] = memory_mb
                    except pynvml.NVMLError as e:
                        if self._is_monitoring:
                            log.error(f"Could not read GPU memory: {e}. Disabling memory monitoring.")
                        self.memory_available = False

                # Utilization
                if self.util_available:
                    try:
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        util_percent = util_info.gpu
                        sample["utilization_percent"] = util_percent
                    except pynvml.NVMLError as e:
                        if self._is_monitoring:
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
                
                # Update accumulators
                if "power_watts" in sample:
                    power_values.append(sample["power_watts"])
                if "temp_c" in sample:
                    temp_values.append(sample["temp_c"])
                if "memory_mb" in sample:
                    memory_values.append(sample["memory_mb"])
                if "utilization_percent" in sample:
                    util_values.append(sample["utilization_percent"])
                
                # Update cached metrics
                # Use the most reliable accumulator for sample count
                self.metrics["num_samples"] = len(power_values) if power_values else (
                    len(util_values) if util_values else (
                        len(memory_values) if memory_values else 0
                    )
                )
                
                if power_values:
                    self.metrics["peak_power_watts"] = max(power_values)
                    self.metrics["average_power_watts"] = sum(power_values) / len(power_values)
                    self.metrics["min_power_watts"] = min(power_values)
                
                if temp_values:
                    self.metrics["peak_temp_c"] = max(temp_values)
                    self.metrics["average_temp_c"] = sum(temp_values) / len(temp_values)
                    self.metrics["min_temp_c"] = min(temp_values)
                
                if memory_values:
                    self.metrics["peak_memory_mb"] = max(memory_values)
                    self.metrics["average_memory_mb"] = sum(memory_values) / len(memory_values)
                    self.metrics["min_memory_mb"] = min(memory_values)
                
                if util_values:
                    self.metrics["peak_utilization_percent"] = max(util_values)
                    self.metrics["average_utilization_percent"] = sum(util_values) / len(util_values)
                    self.metrics["min_utilization_percent"] = min(util_values)
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()

    def stop_monitoring(self):
        """
        Stop the monitoring thread and clean up.
        Note: pynvml shutdown is handled by ProfilerManager to avoid double shutdown.
        """
        return super().stop_monitoring()
