import time
import psutil
import os
import csv
import logging
import tempfile
from .base import BaseDeviceProfiler

log = logging.getLogger(__name__)

class PiProfiler(BaseDeviceProfiler):
    """
    Profiler for Raspberry Pi devices.
    Writes samples to CSV and calculates metrics in real-time.
    """
    def __init__(self, config, profiler_manager=None):
        super().__init__(config)
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.device_name = "Raspberry Pi"
        self.csv_filepath = None
        
        # Cached metrics
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        
        # Check for thermal zone
        self.thermal_zone_path = "/sys/class/thermal/thermal_zone0/temp"
        self.temp_available = os.path.exists(self.thermal_zone_path)
        
        # Set initial availability flag
        self.metrics["temperature_monitoring_available"] = self.temp_available
        
        psutil.cpu_percent(interval=None)
        log.info(f"Initialized Raspberry Pi Profiler")

    def get_device_info(self) -> str:
        return self.device_name

    def _read_temp(self) -> float | None:
        """Read CPU temperature from thermal zone (millidegrees Celsius)."""
        if not self.temp_available:
            return None
        
        try:
            with open(self.thermal_zone_path, 'r') as f:
                temp_m = int(f.read().strip())
            return temp_m / 1000.0  # Convert to Celsius
        except Exception as e:
            log.debug(f"Failed to read temperature: {e}")
            return None

    def _monitor_process(self):
        """Collect CPU, memory, and thermal metrics from Raspberry Pi."""
        # Setup CSV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        self.csv_filepath = os.path.join(temp_dir, f"pi_profiler_{timestamp}.csv")
        self.metrics["csv_filepath"] = self.csv_filepath
        
        log.info(f"Writing Pi samples to: {self.csv_filepath}")
        
        start_time = time.perf_counter()
        csv_file = None
        csv_writer = None
        
        # Metric accumulators
        cpu_values = []
        mem_values = []
        mem_pct_values = []
        temp_values = []
        
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
                
                # Temperature
                temp = self._read_temp()
                if temp is not None:
                    sample["cpu_temp_c"] = temp
                    temp_values.append(temp)
                
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
                self.metrics["num_samples"] = len(cpu_values) if cpu_values else 0
                if cpu_values:
                    self.metrics["average_cpu_utilization_percent"] = sum(cpu_values) / len(cpu_values)
                    self.metrics["peak_cpu_utilization_percent"] = max(cpu_values)
                if mem_values:
                    self.metrics["average_memory_mb"] = sum(mem_values) / len(mem_values)
                    self.metrics["peak_memory_mb"] = max(mem_values)
                if mem_pct_values:
                    self.metrics["average_memory_utilization_percent"] = sum(mem_pct_values) / len(mem_pct_values)
                    self.metrics["peak_memory_utilization_percent"] = max(mem_pct_values)
                if temp_values:
                    self.metrics["average_cpu_temp_c"] = sum(temp_values) / len(temp_values)
                    self.metrics["peak_cpu_temp_c"] = max(temp_values)
                    self.metrics["min_cpu_temp_c"] = min(temp_values)
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                # Update availability flag (may change during monitoring)
                self.metrics["temperature_monitoring_available"] = self.temp_available
                
                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()
