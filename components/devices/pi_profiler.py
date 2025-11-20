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
        
        # Check for power monitoring
        self.power_monitoring_available = False
        self.power_path = None
        self._check_power_availability()
        
        psutil.cpu_percent(interval=None)
        log.info(f"Initialized Raspberry Pi Profiler")

    def get_device_info(self) -> str:
        return self.device_name

    def _check_power_availability(self):
        """
        Check for power monitoring via hwmon interface.
        Raspberry Pi may expose power through hwmon devices.
        """
        base_path = "/sys/class/hwmon/"
        if not os.path.exists(base_path):
            return
        
        try:
            for dir_name in os.listdir(base_path):
                if not dir_name.startswith("hwmon"):
                    continue
                
                hwmon_dir = os.path.join(base_path, dir_name)
                name_path = os.path.join(hwmon_dir, "name")
                power_path = os.path.join(hwmon_dir, "power1_input")
                
                if os.path.exists(name_path) and os.path.exists(power_path):
                    try:
                        with open(name_path, 'r') as f:
                            name = f.read().strip()
                        # Check if readable
                        with open(power_path, 'r') as f:
                            f.read()
                        
                        # Accept common Pi power sensor names
                        if any(keyword in name.lower() for keyword in ['rpi', 'ina219', 'ina260', 'power', 'volt']):
                            self.power_path = power_path
                            self.power_monitoring_available = True
                            log.info(f"[Pi] Power monitoring enabled via {name} at: {power_path}")
                            return
                    except PermissionError:
                        log.debug(f"Permission denied reading power file: {power_path}")
                    except Exception as e:
                        log.debug(f"Error checking power path: {e}")
        except Exception as e:
            log.debug(f"Error while searching for Pi power path: {e}")
        
        if not self.power_monitoring_available:
            log.debug("[Pi] No power monitoring interface found")

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

    def _read_power_watts(self) -> float | None:
        """Read power consumption in watts from hwmon interface."""
        if not self.power_monitoring_available or not self.power_path:
            return None
        
        try:
            with open(self.power_path, 'r') as f:
                power_uw = int(f.read().strip())
            return power_uw / 1_000_000.0  # Convert microwatts to watts
        except PermissionError:
            log.debug(f"Permission denied reading power file: {self.power_path}")
            self.power_monitoring_available = False
            return None
        except Exception as e:
            log.debug(f"Failed to read power: {e}")
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
        power_values = []
        
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
                
                # Power monitoring
                if self.power_monitoring_available:
                    power = self._read_power_watts()
                    if power is not None:
                        sample["power_watts"] = power
                        power_values.append(power)
                
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
                if power_values:
                    self.metrics["average_power_watts"] = sum(power_values) / len(power_values)
                    self.metrics["peak_power_watts"] = max(power_values)
                    self.metrics["min_power_watts"] = min(power_values)
                
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
