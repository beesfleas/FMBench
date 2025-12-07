import time
from pathlib import Path
from typing import Optional
from .base import BaseDeviceProfiler
from .profiler_utils import CSVWriter, MetricAccumulator, generate_csv_filepath, get_results_directory
import pynvml
import logging

log = logging.getLogger(__name__)


class NvidiaGpuProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA GPUs using pynvml (nvidia-smi).
    Writes samples to CSV and calculates metrics in real-time.
    """
    
    def __init__(self, config, device_index: int, profiler_manager=None, results_dir: Optional[Path] = None):
        super().__init__(config, results_dir)
        self.profiler_manager = profiler_manager
        
        if pynvml is None:
            raise ImportError("pynvml library not installed.")
        
        try:
            self.device_index = device_index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            device_name = pynvml.nvmlDeviceGetName(self.handle)
            self.device_name = f"{device_name} (GPU {self.device_index})"
        except pynvml.NVMLError as e:
            log.error("Failed to initialize pynvml for GPU %d: %s", device_index, e)
            raise
        
        self.sampling_interval = config.get("gpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        
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
        
        log.info("Initialized Nvidia GPU Profiler for %s", self.device_name)
        self._check_metric_availability()

    def get_device_info(self) -> str:
        """Return the device name set during initialization."""
        return self.device_name

    def _check_metric_availability(self):
        """Performs a test-read for each metric to set availability flags."""
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
        """Collect GPU metrics and write to CSV."""
        # Setup CSV file in results directory
        suffix = f"_gpu{self.device_index}"
        if self.results_dir:
            self.csv_filepath = generate_csv_filepath(self.results_dir, "nvidia_gpu_profiler", suffix)
        else:
            results_dir = get_results_directory()
            self.csv_filepath = generate_csv_filepath(results_dir, "nvidia_gpu_profiler", suffix)
        
        self.metrics["csv_filepath"] = self.csv_filepath
        log.info("Writing NVIDIA GPU %d samples to: %s", self.device_index, self.csv_filepath)
        
        start_time = time.perf_counter()
        
        # Initialize accumulators
        power_acc = MetricAccumulator(track_nonzero=True)
        temp_acc = MetricAccumulator()
        memory_acc = MetricAccumulator()
        util_acc = MetricAccumulator(track_nonzero=True)
        
        total_energy_joules = 0.0
        
        with CSVWriter(self.csv_filepath) as csv_writer:
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
                        power_acc.add(power_watts)
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error("Could not read GPU power: %s. Disabling power monitoring.", e)
                        self.power_available = False

                # Temperature
                if self.temp_available:
                    try:
                        temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        sample["temp_c"] = temp_c
                        temp_acc.add(temp_c)
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error("Could not read GPU temp: %s. Disabling temp monitoring.", e)
                        self.temp_available = False

                # Memory
                if self.memory_available:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                        memory_mb = memory_info.used / (1024 * 1024)
                        sample["memory_mb"] = memory_mb
                        memory_acc.add(memory_mb)
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error("Could not read GPU memory: %s. Disabling memory monitoring.", e)
                        self.memory_available = False

                # Utilization
                if self.util_available:
                    try:
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                        util_percent = util_info.gpu
                        sample["utilization_percent"] = util_percent
                        util_acc.add(util_percent)
                    except pynvml.NVMLError as e:
                        if not self._stop_event.is_set():
                            log.error("Could not read GPU util: %s. Disabling util monitoring.", e)
                        self.util_available = False
                
                # Write sample to CSV
                csv_writer.write_sample(sample)
                
                # Update cached metrics
                self.metrics["num_samples"] = max(power_acc.count, util_acc.count, memory_acc.count)
                
                power_stats = power_acc.get_stats(use_nonzero=True)
                self.metrics["peak_power_watts"] = power_stats["peak"]
                self.metrics["average_power_watts"] = power_stats["average"]
                self.metrics["min_power_watts"] = power_stats["min"]
                self.metrics["total_energy_joules"] = total_energy_joules
                
                if temp_acc.count > 0:
                    temp_stats = temp_acc.get_stats()
                    self.metrics["peak_temp_c"] = temp_stats["peak"]
                    self.metrics["average_temp_c"] = temp_stats["average"]
                    self.metrics["min_temp_c"] = temp_stats["min"]
                
                if memory_acc.count > 0:
                    memory_stats = memory_acc.get_stats()
                    self.metrics["peak_memory_mb"] = memory_stats["peak"]
                    self.metrics["average_memory_mb"] = memory_stats["average"]
                    self.metrics["min_memory_mb"] = memory_stats["min"]
                
                util_stats = util_acc.get_stats(use_nonzero=True)
                self.metrics["peak_utilization_percent"] = util_stats["peak"]
                self.metrics["average_utilization_percent"] = util_stats["average"]
                self.metrics["min_utilization_percent"] = util_stats["min"]
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    self._stop_event.wait(sleep_duration)

    def stop_monitoring(self):
        """
        Stop the monitoring thread and clean up.
        Note: pynvml shutdown is handled by ProfilerManager to avoid double shutdown.
        """
        return super().stop_monitoring()
