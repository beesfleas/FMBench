import time
import psutil
import os
import csv
import logging
import tempfile
from .base import BaseDeviceProfiler

log = logging.getLogger(__name__)

class JetsonProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA Jetson devices.
    Writes samples to CSV and calculates metrics in real-time.
    """
    def __init__(self, config, profiler_manager=None):
        super().__init__(config)
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.device_name = "NVIDIA Jetson"
        self.csv_filepath = None
        
        # Cached metrics
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_available = True
            log.info("Jetson GPU (NVIDIA) initialized via pynvml")
        except Exception as e:
            log.warning(f"Jetson GPU not available: {e}")
            self.pynvml = None
            self.gpu_available = False
        
        psutil.cpu_percent(interval=None)
        log.info(f"Initialized Jetson Profiler")

    def get_device_info(self) -> str:
        return self.device_name

    def _monitor_process(self):
        """Collect CPU, GPU, and power metrics from Jetson."""
        # Setup CSV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        self.csv_filepath = os.path.join(temp_dir, f"jetson_profiler_{timestamp}.csv")
        self.metrics["csv_filepath"] = self.csv_filepath
        
        log.info(f"Writing Jetson samples to: {self.csv_filepath}")
        
        start_time = time.perf_counter()
        csv_file = None
        csv_writer = None
        
        # Metric accumulators
        cpu_values = []
        mem_values = []
        mem_pct_values = []
        gpu_util_values = []
        gpu_mem_values = []
        gpu_temp_values = []
        
        try:
            while self._is_monitoring:
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                
                sample = {"timestamp": rel_timestamp}
                
                # CPU and Memory
                try:
                    cpu_util = psutil.cpu_percent(interval=None)
                    vmem = psutil.virtual_memory()
                    sample["cpu_utilization_percent"] = cpu_util
                    sample["memory_used_mb"] = vmem.used / (1024 * 1024)
                    sample["memory_utilization_percent"] = vmem.percent
                    
                    cpu_values.append(cpu_util)
                    mem_values.append(sample["memory_used_mb"])
                    mem_pct_values.append(vmem.percent)
                except Exception as e:
                    log.warning(f"Failed to read CPU/memory: {e}")
                
                # GPU metrics (if available)
                if self.gpu_available and self.pynvml:
                    try:
                        handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        sample["gpu_utilization_percent"] = util.gpu
                        sample["gpu_memory_mb"] = mem_info.used / (1024 * 1024)
                        
                        gpu_util_values.append(util.gpu)
                        gpu_mem_values.append(sample["gpu_memory_mb"])
                        
                        try:
                            temp = self.pynvml.nvmlDeviceGetTemperature(handle, 0)
                            sample["gpu_temp_c"] = temp
                            gpu_temp_values.append(temp)
                        except Exception:
                            # Temperature reading failed, continue without it
                            pass
                    except Exception as e:
                        log.debug(f"GPU read failed: {e}")
                
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
                # Use the most reliable accumulator (CPU) for sample count
                self.metrics["num_samples"] = len(cpu_values) if cpu_values else 0
                if cpu_values:
                    self.metrics["average_cpu_utilization_percent"] = sum(cpu_values) / len(cpu_values)
                    self.metrics["peak_cpu_utilization_percent"] = max(cpu_values)
                    self.metrics["min_cpu_utilization_percent"] = min(cpu_values)
                if mem_values:
                    self.metrics["average_memory_mb"] = sum(mem_values) / len(mem_values)
                    self.metrics["peak_memory_mb"] = max(mem_values)
                    self.metrics["min_memory_mb"] = min(mem_values)
                if mem_pct_values:
                    self.metrics["average_memory_utilization_percent"] = sum(mem_pct_values) / len(mem_pct_values)
                    self.metrics["peak_memory_utilization_percent"] = max(mem_pct_values)
                    self.metrics["min_memory_utilization_percent"] = min(mem_pct_values)
                if gpu_util_values:
                    self.metrics["average_gpu_utilization_percent"] = sum(gpu_util_values) / len(gpu_util_values)
                    self.metrics["peak_gpu_utilization_percent"] = max(gpu_util_values)
                    self.metrics["min_gpu_utilization_percent"] = min(gpu_util_values)
                if gpu_mem_values:
                    self.metrics["average_gpu_memory_mb"] = sum(gpu_mem_values) / len(gpu_mem_values)
                    self.metrics["peak_gpu_memory_mb"] = max(gpu_mem_values)
                    self.metrics["min_gpu_memory_mb"] = min(gpu_mem_values)
                if gpu_temp_values:
                    self.metrics["average_gpu_temp_c"] = sum(gpu_temp_values) / len(gpu_temp_values)
                    self.metrics["peak_gpu_temp_c"] = max(gpu_temp_values)
                    self.metrics["min_gpu_temp_c"] = min(gpu_temp_values)
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()
