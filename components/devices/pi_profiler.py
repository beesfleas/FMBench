import time
import psutil
import os
import csv
import logging
import tempfile
import subprocess
from typing import Optional
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
        
        # Check for power monitoring (Pi 5 PMIC only)
        self.power_monitoring_available = False
        self._check_power_availability()
        
        psutil.cpu_percent(interval=None)
        log.info(f"Initialized Raspberry Pi Profiler")

    def get_device_info(self) -> str:
        return self.device_name

    def _check_power_availability(self):
        """Check for Raspberry Pi 5 PMIC power monitoring."""
        try:
            result = subprocess.run(
                ["vcgencmd", "pmic_read_adc"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and "VDD_CORE_A" in result.stdout and "VDD_CORE_V" in result.stdout:
                self.power_monitoring_available = True
                log.info("[Pi] Power monitoring enabled via Raspberry Pi 5 PMIC")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            log.debug("[Pi] Pi 5 PMIC not available")
        except Exception as e:
            log.debug(f"Error while searching for Pi power path: {e}")
        
        if not self.power_monitoring_available:
            log.warning("[Pi] No power monitoring interface found")

    def _read_temp(self) -> Optional[float]:
        """Read CPU temperature from thermal zone (millidegrees Celsius)."""
        if not self.temp_available:
            return None
        
        try:
            with open(self.thermal_zone_path, 'r') as f:
                temp_m = int(f.read().strip())
            return temp_m / 1000.0  # Convert to Celsius
        except Exception as e:
            log.warning(f"Failed to read temperature: {e}")
            return None

    def _read_power_watts(self) -> Optional[float]:
        """Read CPU power consumption in watts using Raspberry Pi 5 PMIC."""
        if not self.power_monitoring_available:
            return None
        
        try:
            result = subprocess.run(
                ["vcgencmd", "pmic_read_adc"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode != 0:
                return None
            
            current = None
            voltage = None
            
            for line in result.stdout.splitlines():
                if "VDD_CORE_A" in line:
                    current_str = line.split("=")[1].replace("A", "").strip()
                    current = float(current_str)
                elif "VDD_CORE_V" in line:
                    voltage_str = line.split("=")[1].replace("V", "").strip()
                    voltage = float(voltage_str)
            
            if current is not None and voltage is not None:
                return current * voltage  # Power (W) = Current (A) × Voltage (V)
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, IndexError):
            pass
        except Exception as e:
            log.debug(f"Error reading power: {e}")
        
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
        
        # Initialize stats
        stats = {
            "cpu": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf'), "nonzero_count": 0, "nonzero_sum": 0.0, "nonzero_min": float('inf'), "nonzero_max": 0.0},
            "mem": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf')},
            "mem_pct": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf')},
            "temp": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf')},
            "power": {"count": 0, "sum": 0.0, "max": 0.0, "min": float('inf'), "nonzero_count": 0, "nonzero_sum": 0.0, "nonzero_min": float('inf'), "nonzero_max": 0.0}
        }
        total_energy_joules = 0.0
        
        try:
            while not self._stop_event.is_set():
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
                    
                    s = stats["cpu"]
                    s["count"] += 1
                    s["sum"] += cpu_util
                    s["max"] = max(s["max"], cpu_util)
                    s["min"] = min(s["min"], cpu_util)
                    if cpu_util != 0:
                        s["nonzero_count"] += 1
                        s["nonzero_sum"] += cpu_util
                        s["nonzero_max"] = max(s["nonzero_max"], cpu_util)
                        s["nonzero_min"] = min(s["nonzero_min"], cpu_util)

                    s = stats["mem"]
                    s["count"] += 1
                    s["sum"] += sample["memory_used_mb"]
                    s["max"] = max(s["max"], sample["memory_used_mb"])
                    s["min"] = min(s["min"], sample["memory_used_mb"])

                    s = stats["mem_pct"]
                    s["count"] += 1
                    s["sum"] += vmem.percent
                    s["max"] = max(s["max"], vmem.percent)
                    s["min"] = min(s["min"], vmem.percent)
                    
                except Exception as e:
                    log.warning(f"Failed to read CPU/memory: {e}")
                
                # Temperature
                temp = self._read_temp()
                if temp is not None:
                    sample["cpu_temp_c"] = temp
                    s = stats["temp"]
                    s["count"] += 1
                    s["sum"] += temp
                    s["max"] = max(s["max"], temp)
                    s["min"] = min(s["min"], temp)
                
                # Power monitoring
                if self.power_monitoring_available:
                    power = self._read_power_watts()
                    if power is not None:
                        sample["power_watts"] = power
                        total_energy_joules += power * self.sampling_interval
                        
                        s = stats["power"]
                        s["count"] += 1
                        s["sum"] += power
                        s["max"] = max(s["max"], power)
                        s["min"] = min(s["min"], power)
                        if power != 0:
                            s["nonzero_count"] += 1
                            s["nonzero_sum"] += power
                            s["nonzero_max"] = max(s["nonzero_max"], power)
                            s["nonzero_min"] = min(s["nonzero_min"], power)
                
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
                self.metrics["num_samples"] = stats["cpu"]["count"]
                
                s = stats["cpu"]
                if s["nonzero_count"] > 0:
                    self.metrics["average_cpu_utilization_percent"] = s["nonzero_sum"] / s["nonzero_count"]
                    self.metrics["peak_cpu_utilization_percent"] = s["nonzero_max"]
                else:
                    self.metrics["average_cpu_utilization_percent"] = 0
                    self.metrics["peak_cpu_utilization_percent"] = 0
                
                s = stats["mem"]
                if s["count"] > 0:
                    self.metrics["average_memory_mb"] = s["sum"] / s["count"]
                    self.metrics["peak_memory_mb"] = s["max"]
                
                s = stats["mem_pct"]
                if s["count"] > 0:
                    self.metrics["average_memory_utilization_percent"] = s["sum"] / s["count"]
                    self.metrics["peak_memory_utilization_percent"] = s["max"]
                
                s = stats["temp"]
                if s["count"] > 0:
                    self.metrics["average_cpu_temp_c"] = s["sum"] / s["count"]
                    self.metrics["peak_cpu_temp_c"] = s["max"]
                    self.metrics["min_cpu_temp_c"] = s["min"]
                
                s = stats["power"]
                if s["nonzero_count"] > 0:
                    self.metrics["average_power_watts"] = s["nonzero_sum"] / s["nonzero_count"]
                    self.metrics["peak_power_watts"] = s["nonzero_max"]
                    self.metrics["min_power_watts"] = s["nonzero_min"]
                else:
                    self.metrics["average_power_watts"] = 0
                    self.metrics["peak_power_watts"] = 0
                    self.metrics["min_power_watts"] = 0
                self.metrics["total_energy_joules"] = total_energy_joules
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    # Use event.wait() instead of time.sleep() to allow immediate interruption
                    self._stop_event.wait(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()
