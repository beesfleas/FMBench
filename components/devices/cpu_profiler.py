import time
import psutil
import platform
import os
from .base import BaseDeviceProfiler
from collections import defaultdict
import logging

log = logging.getLogger(__name__)

class LocalCpuProfiler(BaseDeviceProfiler):
    """
    Profiler for local CPU and RAM using psutil.
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.sampling_interval = config.get("cpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        
        self.samples = []
        self._start_time = None
        self.device_name = f"{platform.processor()}"
        
        self.energy_counter_path = None # Generic path for Intel or AMD
        
        # Set availability flags
        self.power_monitoring_available = False
        self.temp_monitoring_available = False
        self._check_metric_availability()

        # Initialize psutil for CPU percent.
        psutil.cpu_percent(interval=None)

    def get_device_info(self) -> str:
        """Return the device name set during initialization."""
        return self.device_name

    def _find_intel_rapl_path(self) -> str | None:
        """
        Find the path to the Intel RAPL package-0 energy counter.
        """
        base_path = "/sys/class/powercap/"
        if not os.path.exists(base_path):
            return None
        
        try:
            for dir_name in os.listdir(base_path):
                if dir_name.startswith("intel-rapl:"):
                    energy_path = os.path.join(base_path, dir_name, "energy_uj")
                    if os.path.exists(energy_path):
                        try:
                            with open(energy_path, 'r') as f: f.read()
                            return energy_path
                        except PermissionError:
                            log.warning(f"Intel RAPL file found ({energy_path}) but permission denied.")
                            return None
        except Exception as e:
            log.error(f"Error while searching for Intel RAPL path: {e}")
        return None

    def _find_amd_energy_path(self) -> str | None:
        """
        Find the path to the AMD energy counter via hwmon.
        """
        base_path = "/sys/class/hwmon/"
        if not os.path.exists(base_path):
            return None
            
        try:
            for dir_name in os.listdir(base_path):
                if not dir_name.startswith("hwmon"):
                    continue
                
                name_path = os.path.join(base_path, dir_name, "name")
                energy_path = os.path.join(base_path, dir_name, "energy1_input")
                
                if os.path.exists(name_path) and os.path.exists(energy_path):
                    with open(name_path, 'r') as f:
                        name = f.read().strip()
                    
                    if name == "amd_energy":
                        try:
                            with open(energy_path, 'r') as f: f.read()
                            return energy_path
                        except PermissionError:
                            log.warning(f"AMD Energy file found ({energy_path}) but permission denied.")
                            return None
        except Exception as e:
            log.error(f"Error while searching for AMD Energy path: {e}")
        return None

    def _check_metric_availability(self):
        """
        Performs a test-read for each metric to set availability flags.
        Tries psutil, then Intel RAPL, then AMD Energy.
        """
        # 1. Check for psutil.sensors_power()
        self.power_monitoring_available = hasattr(psutil, 'sensors_power')
        
        if self.power_monitoring_available:
            log.info("psutil.sensors_power() found. Using psutil for power monitoring.")
        else:
            log.warning("'psutil.sensors_power' not found. Trying kernel file fallbacks...")
            
            # 2. Try Intel RAPL
            self.energy_counter_path = self._find_intel_rapl_path()
            if self.energy_counter_path:
                log.info(f"Intel RAPL fallback enabled. Found energy counter at: {self.energy_counter_path}")
                self.power_monitoring_available = True
            else:
                # 3. Try AMD Energy
                self.energy_counter_path = self._find_amd_energy_path()
                if self.energy_counter_path:
                    log.info(f"AMD Energy fallback enabled. Found energy counter at: {self.energy_counter_path}")
                    self.power_monitoring_available = True
                else:
                    log.warning("All power fallbacks failed: No readable Intel RAPL or AMD Energy file found.")
            
        # 4. Check for psutil.sensors_temperatures()
        self.temp_monitoring_available = hasattr(psutil, 'sensors_temperatures')
        if not self.temp_monitoring_available:
            log.warning("'psutil.sensors_temperatures' not found. Disabling temperature monitoring.")

    def _read_energy_uj(self) -> int | None:
        """
        Reads the raw energy counter from the determined path.
        This is stateLESS and just returns the current microjoule value.
        """
        try:
            with open(self.energy_counter_path, 'r') as f:
                current_energy_uj = int(f.read().strip())
            return current_energy_uj
            
        except PermissionError:
            log.error(f"Permission denied reading energy file: {self.energy_counter_path}. Disabling power monitoring.")
            self.power_monitoring_available = False
            return None
        except Exception as e:
            log.error(f"Failed to read energy file: {e}. Disabling power monitoring.")
            self.power_monitoring_available = False
            return None

    def _monitor_process(self):
        """
        Lightweight monitoring loop - just collect raw data.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
            
        while self._is_monitoring:
            monitor_start_time = time.perf_counter()
            
            # 1. System-wide CPU Utilization
            cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
            
            # 2. System-wide Physical (Hardware) RAM
            vmem = psutil.virtual_memory()
            memory_used_mb = vmem.used / (1024 * 1024)
            memory_percent = vmem.percent
            
            # 3. System-wide CPU Power (Watts)
            power_watts = None
            energy_uj = None
            
            if self.power_monitoring_available:
                if self.energy_counter_path:
                    # Use Kernel File Fallback
                    energy_uj = self._read_energy_uj()
                    # power_watts remains None
                else:
                    # Use psutil (Collects Power)
                    try:
                        power_info = psutil.sensors_power()
                        if not power_info:
                            raise Exception("No power sensors found by psutil.")
                        if hasattr(power_info, 'core') and power_info.core:
                             power_watts = power_info.core.current
                        elif power_info:
                            power_watts = power_info[0].current
                    except Exception as e:
                        if self._is_monitoring:
                            log.error(f"Could not read CPU power: {e}. Disabling power monitoring.")
                        self.power_monitoring_available = False
            
            # 4. System-wide CPU Temperature (Celsius)
            cpu_temp_c = None
            if self.temp_monitoring_available:
                try:
                    temps = psutil.sensors_temperatures()
                    if not temps:
                        raise Exception("No temperature sensors found by psutil.")
                    found_temp = False
                    for sensor_group, readings in temps.items():
                        for sensor in readings:
                            label = sensor.label.lower()
                            if 'package' in label or 'cpu' in label or 'tdie' in label:
                                cpu_temp_c = sensor.current
                                found_temp = True
                                break
                        if found_temp:
                            break
                    if not found_temp:
                        cpu_temp_c = list(temps.values())[0][0].current
                except Exception as e:
                    if self._is_monitoring:
                        log.error(f"Could not read CPU temperature: {e}. Disabling temperature monitoring.")
                    self.temp_monitoring_available = False

            # 5. Get timestamp relative to the start
            timestamp = time.perf_counter() - self._start_time
            
            self.samples.append({
                "timestamp": timestamp,
                "cpu_utilization_percent": cpu_percent,
                "memory_used_mb": memory_used_mb,
                "memory_utilization_percent": memory_percent,
                "power_watts": power_watts, # Will be None in RAPL/Energy mode
                "energy_uj": energy_uj,     # Will be None in psutil mode
                "cpu_temp_c": cpu_temp_c
            })
            
            # Sleep to maintain the desired sampling interval
            elapsed = time.perf_counter() - monitor_start_time
            sleep_duration = self.sampling_interval - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def get_metrics(self):
        """
        Process all raw samples and return a structured dictionary.
        """
        if not self.samples:
            return {"device_name": self.device_name, "error": "No metrics collected."}

        num_samples = len(self.samples)
        
        if num_samples > 1:
            monitoring_duration = self.samples[-1]['timestamp'] - self.samples[0]['timestamp']
        else:
            monitoring_duration = 0.0
        
        stats = defaultdict(list)
        power_values = []
        temp_values = []
        energy_uj_values = []
        
        for sample in self.samples:
            stats["cpu_utilization_percent"].append(sample["cpu_utilization_percent"])
            stats["memory_used_mb"].append(sample["memory_used_mb"])
            stats["memory_utilization_percent"].append(sample["memory_utilization_percent"])
            if sample["power_watts"] is not None:
                power_values.append(sample["power_watts"])
            if sample["energy_uj"] is not None:
                energy_uj_values.append(sample["energy_uj"])
            if sample["cpu_temp_c"] is not None:
                temp_values.append(sample["cpu_temp_c"])

        if not stats["cpu_utilization_percent"]:
             return {"device_name": self.device_name, "error": "No valid samples collected."}
        
        num_valid_samples = len(stats["cpu_utilization_percent"])

        metrics = {
            "device_name": self.device_name,
            # "raw_samples": self.samples,
            "num_samples": num_samples,
            "monitoring_duration_seconds": monitoring_duration,
            "sampling_interval": self.sampling_interval,
            "peak_cpu_utilization_percent": max(stats["cpu_utilization_percent"]),
            "average_cpu_utilization_percent": sum(stats["cpu_utilization_percent"]) / num_valid_samples,
            "peak_memory_mb": max(stats["memory_used_mb"]),
            "average_memory_mb": sum(stats["memory_used_mb"]) / num_valid_samples,
            "peak_memory_utilization_percent": max(stats["memory_utilization_percent"]),
            "average_memory_utilization_percent": sum(stats["memory_utilization_percent"]) / num_valid_samples,
        }
        
        if energy_uj_values:
            # Use the raw energy counter
            first_energy = energy_uj_values[0]
            last_energy = energy_uj_values[-1]
            total_energy_joules = (last_energy - first_energy) / 1_000_000.0
            
            metrics["total_energy_joules"] = total_energy_joules
            if monitoring_duration > 0:
                metrics["average_power_watts"] = total_energy_joules / monitoring_duration
            
        elif power_values:
            # Fallback to trapezoidal integration of power (W)
            metrics["peak_power_watts"] = max(power_values)
            metrics["average_power_watts"] = sum(power_values) / len(power_values)
            metrics["min_power_watts"] = min(power_values)
            
            total_energy_joules = 0.0
            for i in range(1, len(self.samples)):
                if self.samples[i]['power_watts'] is not None and self.samples[i-1]['power_watts'] is not None:
                    dt = self.samples[i]['timestamp'] - self.samples[i-1]['timestamp']
                    avg_power = (self.samples[i]['power_watts'] + self.samples[i-1]['power_watts']) / 2
                    total_energy_joules += avg_power * dt    
            metrics["total_energy_joules"] = total_energy_joules

        if temp_values:
            metrics["peak_temp_c"] = max(temp_values)
            metrics["average_temp_c"] = sum(temp_values) / len(temp_values)
            metrics["min_temp_c"] = min(temp_values)

        return metrics