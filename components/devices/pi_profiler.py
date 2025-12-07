import time
import psutil
import os
import subprocess
from pathlib import Path
from typing import Optional
from .base import BaseDeviceProfiler
from .profiler_utils import CSVWriter, MetricAccumulator, generate_csv_filepath, get_results_directory
import logging

log = logging.getLogger(__name__)


class PiProfiler(BaseDeviceProfiler):
    """
    Profiler for Raspberry Pi devices.
    Writes samples to CSV and calculates metrics in real-time.
    """
    
    def __init__(self, config, profiler_manager=None, results_dir: Optional[Path] = None):
        super().__init__(config, results_dir)
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.device_name = "Raspberry Pi"
        
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
        
        # Prime psutil
        psutil.cpu_percent(interval=None)
        log.info("Initialized Raspberry Pi Profiler")

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
            log.debug("Error while searching for Pi power path: %s", e)
        
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
            log.warning("Failed to read temperature: %s", e)
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
            log.debug("Error reading power: %s", e)
        
        return None

    def _monitor_process(self):
        """Collect CPU, memory, and thermal metrics from Raspberry Pi."""
        # Setup CSV file in results directory
        if self.results_dir:
            self.csv_filepath = generate_csv_filepath(self.results_dir, "pi_profiler")
        else:
            results_dir = get_results_directory()
            self.csv_filepath = generate_csv_filepath(results_dir, "pi_profiler")
        
        self.metrics["csv_filepath"] = self.csv_filepath
        log.info("Writing Pi samples to: %s", self.csv_filepath)
        
        start_time = time.perf_counter()
        
        # Initialize accumulators
        cpu_acc = MetricAccumulator(track_nonzero=True)
        mem_acc = MetricAccumulator()
        mem_pct_acc = MetricAccumulator()
        temp_acc = MetricAccumulator()
        power_acc = MetricAccumulator(track_nonzero=True)
        
        total_energy_joules = 0.0
        
        fieldnames = [
            "timestamp", "cpu_utilization_percent", "memory_used_mb", 
            "memory_utilization_percent", "cpu_temp_c", "power_watts"
        ]
        
        with CSVWriter(self.csv_filepath, fieldnames=fieldnames) as csv_writer:
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
                    
                    cpu_acc.add(cpu_util)
                    mem_acc.add(sample["memory_used_mb"])
                    mem_pct_acc.add(vmem.percent)
                    
                except Exception as e:
                    log.warning("Failed to read CPU/memory: %s", e)
                
                # Temperature
                temp = self._read_temp()
                if temp is not None:
                    sample["cpu_temp_c"] = temp
                    temp_acc.add(temp)
                
                # Power monitoring
                if self.power_monitoring_available:
                    power = self._read_power_watts()
                    if power is not None:
                        sample["power_watts"] = power
                        total_energy_joules += power * self.sampling_interval
                        power_acc.add(power)
                
                # Write sample to CSV
                csv_writer.write_sample(sample)
                
                # Update cached metrics
                self.metrics["num_samples"] = cpu_acc.count
                
                cpu_stats = cpu_acc.get_stats(use_nonzero=True)
                self.metrics["average_cpu_utilization_percent"] = cpu_stats["average"]
                self.metrics["peak_cpu_utilization_percent"] = cpu_stats["peak"]
                
                if mem_acc.count > 0:
                    mem_stats = mem_acc.get_stats()
                    self.metrics["average_memory_mb"] = mem_stats["average"]
                    self.metrics["peak_memory_mb"] = mem_stats["peak"]
                
                if mem_pct_acc.count > 0:
                    mem_pct_stats = mem_pct_acc.get_stats()
                    self.metrics["average_memory_utilization_percent"] = mem_pct_stats["average"]
                    self.metrics["peak_memory_utilization_percent"] = mem_pct_stats["peak"]
                
                if temp_acc.count > 0:
                    temp_stats = temp_acc.get_stats()
                    self.metrics["average_cpu_temp_c"] = temp_stats["average"]
                    self.metrics["peak_cpu_temp_c"] = temp_stats["peak"]
                    self.metrics["min_cpu_temp_c"] = temp_stats["min"]
                
                power_stats = power_acc.get_stats(use_nonzero=True)
                self.metrics["average_power_watts"] = power_stats["average"]
                self.metrics["peak_power_watts"] = power_stats["peak"]
                self.metrics["min_power_watts"] = power_stats["min"]
                self.metrics["total_energy_joules"] = total_energy_joules
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    self._stop_event.wait(sleep_duration)
