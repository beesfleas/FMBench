import time
import psutil
import platform
import os
import csv
from .base import BaseDeviceProfiler
import logging
import tempfile

log = logging.getLogger(__name__)

class LocalCpuProfiler(BaseDeviceProfiler):
    """
    Profiler for local CPU and RAM using psutil.
    Writes samples to CSV and calculates metrics in real-time.
    """
    def __init__(self, config, profiler_manager=None):
        super().__init__(config)
        
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("cpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        
        self.device_name = f"{platform.processor()}"
        self.cpu_type = None
        self.energy_counter_path = None
        self.power_monitoring_available = False
        self.temp_monitoring_available = False
        
        # CSV file path
        self.csv_filepath = None
        
        # Cached metrics (updated during monitoring)
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        
        self._detect_cpu_type()
        self._check_metric_availability()

        log.info(f"Initialized CPU Profiler for {self.device_name}")
        psutil.cpu_percent(interval=None)

    def get_device_info(self) -> str:
        """Return the device name set during initialization."""
        return self.device_name

    def _detect_cpu_type(self) -> str:
        """
        Detects the CPU type and stores in self.cpu_type.
        Returns one of: 'intel', 'amd', 'arm', 'unknown'
        """
        system_info = self.profiler_manager.get_system_info() if self.profiler_manager else {}
        processor_info = system_info.get('processor_info', platform.processor().lower())
        
        # Detect CPU type
        if "intel" in processor_info or "x86" in processor_info:
            self.cpu_type = "intel"
        elif "amd" in processor_info or "ryzen" in processor_info or "epyc" in processor_info:
            self.cpu_type = "amd"
        elif "arm" in processor_info or "aarch64" in processor_info:
            self.cpu_type = "arm"
        else:
            self.cpu_type = "unknown"
        
        log.info(f"Detected CPU type: {self.cpu_type}")
        return self.cpu_type

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
        Checks available metrics for this CPU type and sets availability flags.
        For all CPUs: checks psutil for utilization, temperature, memory (universal).
        """
        self.utilization_available = True
        self.temp_monitoring_available = hasattr(psutil, 'sensors_temperatures')
        if not self.temp_monitoring_available:
            log.warning("'psutil.sensors_temperatures' not found. Disabling temperature monitoring.")
        
        # Check for power monitoring (CPU-type specific routing)
        self._check_power_availability()

    def _check_power_availability(self):
        """
        Routes power metric availability check based on detected CPU type.
        Sets self.power_monitoring_available and self.energy_counter_path as needed.
        
        Only Intel and AMD (Linux/Windows PCs) are supported.
        macOS is handled by MacProfiler using powermetrics.
        """
        # No universal power monitoring via psutil
        self.power_monitoring_available = False
        
        # CPU-specific power metric collection via kernel interfaces
        if self.cpu_type == "intel":
            self.energy_counter_path = self._find_intel_rapl_path()
            if self.energy_counter_path:
                log.info(f"[Intel] Intel RAPL enabled at: {self.energy_counter_path}")
                self.power_monitoring_available = True
            else:
                log.warning("[Intel] No Intel RAPL found. Power monitoring disabled.")
        
        elif self.cpu_type == "amd":
            self.energy_counter_path = self._find_amd_energy_path()
            if self.energy_counter_path:
                log.info(f"[AMD] AMD Energy enabled at: {self.energy_counter_path}")
                self.power_monitoring_available = True
            else:
                log.warning("[AMD] No AMD Energy file found. Power monitoring disabled.")
        
        elif self.cpu_type == "arm":
            log.warning("[ARM] No power metric collection implemented for generic ARM CPUs.")
            self.power_monitoring_available = False
        
        else:
            log.warning(f"[{self.cpu_type}] Unknown CPU type. Power monitoring disabled.")
            self.power_monitoring_available = False

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
        """Main monitoring loop - write to CSV and update metrics in real-time."""
        # Setup CSV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        self.csv_filepath = os.path.join(temp_dir, f"cpu_profiler_{timestamp}.csv")
        self.metrics["csv_filepath"] = self.csv_filepath
        
        log.info(f"Writing CPU samples to: {self.csv_filepath}")
        
        start_time = time.perf_counter()
        csv_file = None
        csv_writer = None
        
        # Initialize metric accumulators
        cpu_values = []
        mem_values = []
        mem_pct_values = []
        temp_values = []
        energy_values = []
        
        try:
            while self._is_monitoring:
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                
                # Collect metrics
                cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
                vmem = psutil.virtual_memory()
                
                sample = {
                    "timestamp": rel_timestamp,
                    "cpu_utilization_percent": cpu_percent,
                    "memory_used_mb": vmem.used / (1024 * 1024),
                    "memory_utilization_percent": vmem.percent,
                }
                
                # Accumulate for real-time metrics
                cpu_values.append(cpu_percent)
                mem_values.append(sample["memory_used_mb"])
                mem_pct_values.append(vmem.percent)
                
                # Power monitoring
                if self.power_monitoring_available:
                    energy_uj = self._read_energy_uj()
                    if energy_uj is not None:
                        sample["energy_uj"] = energy_uj
                        energy_values.append(energy_uj)
                
                # Temperature monitoring
                if self.temp_monitoring_available:
                    try:
                        temps = psutil.sensors_temperatures()
                        if temps:
                            cpu_temp = None
                            for sensor_group, readings in temps.items():
                                for sensor in readings:
                                    label = sensor.label.lower()
                                    if 'package' in label or 'cpu' in label or 'tdie' in label:
                                        cpu_temp = sensor.current
                                        break
                                if cpu_temp is not None:
                                    break
                            
                            if cpu_temp is not None:
                                sample["cpu_temp_c"] = cpu_temp
                                temp_values.append(cpu_temp)
                    except Exception as e:
                        log.warning(f"Temperature read failed: {e}")
                        self.temp_monitoring_available = False
                
                # Write to CSV (open on first sample, close on last)
                if csv_file is None:
                    try:
                        csv_file = open(self.csv_filepath, 'w', newline='')
                        csv_writer = csv.DictWriter(csv_file, fieldnames=sample.keys())
                        csv_writer.writeheader()
                    except Exception as e:
                        log.error(f"Failed to create CSV file {self.csv_filepath}: {e}")
                        # Continue monitoring but without CSV logging
                        csv_file = None
                        csv_writer = None
                
                if csv_writer is not None:
                    try:
                        csv_writer.writerow(sample)
                        csv_file.flush()
                    except Exception as e:
                        log.warning(f"Failed to write CSV sample: {e}")
                
                # Update cached metrics in real-time
                self.metrics["num_samples"] = len(cpu_values) if cpu_values else 0
                if len(cpu_values) > 0:
                    self.metrics["average_cpu_utilization_percent"] = sum(cpu_values) / len(cpu_values)
                    self.metrics["peak_cpu_utilization_percent"] = max(cpu_values)
                    self.metrics["min_cpu_utilization_percent"] = min(cpu_values)
                if len(mem_values) > 0:
                    self.metrics["average_memory_mb"] = sum(mem_values) / len(mem_values)
                    self.metrics["peak_memory_mb"] = max(mem_values)
                    self.metrics["min_memory_mb"] = min(mem_values)
                if len(mem_pct_values) > 0:
                    self.metrics["average_memory_utilization_percent"] = sum(mem_pct_values) / len(mem_pct_values)
                    self.metrics["peak_memory_utilization_percent"] = max(mem_pct_values)
                    self.metrics["min_memory_utilization_percent"] = min(mem_pct_values)
                if temp_values:
                    self.metrics["average_cpu_temp_c"] = sum(temp_values) / len(temp_values)
                    self.metrics["peak_cpu_temp_c"] = max(temp_values)
                    self.metrics["min_cpu_temp_c"] = min(temp_values)
                
                # Calculate energy metrics
                if len(energy_values) >= 2:
                    total_energy_joules = (energy_values[-1] - energy_values[0]) / 1_000_000.0
                    self.metrics["total_energy_joules"] = total_energy_joules
                    if rel_timestamp > 0:
                        self.metrics["average_power_watts"] = total_energy_joules / rel_timestamp
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                # Sleep to maintain sampling interval
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()
