import time
import psutil
import platform
import os
from typing import Optional
from pathlib import Path
from .base import BaseDeviceProfiler
from .profiler_utils import CSVWriter, MetricAccumulator, generate_csv_filepath, get_results_directory
import logging

log = logging.getLogger(__name__)


class LocalCpuProfiler(BaseDeviceProfiler):
    """
    Profiler for local CPU and RAM using psutil.
    Writes samples to CSV and calculates metrics in real-time.
    """
    
    def __init__(self, config, profiler_manager=None, results_dir: Optional[Path] = None):
        super().__init__(config, results_dir)
        
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("cpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        
        self.device_name = f"{platform.processor()}"
        self.cpu_type = None
        self.energy_counter_path = None
        self.power_monitoring_available = False
        self.temp_monitoring_available = False
        
        # Cached metrics (updated during monitoring)
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        
        self._detect_cpu_type()
        self._check_metric_availability()

        log.info("Initialized CPU Profiler for %s", self.device_name)
        # Prime psutil to avoid initial 0.0 reading
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
        
        if "intel" in processor_info or "x86" in processor_info:
            self.cpu_type = "intel"
        elif "amd" in processor_info or "ryzen" in processor_info or "epyc" in processor_info:
            self.cpu_type = "amd"
        elif "arm" in processor_info or "aarch64" in processor_info:
            self.cpu_type = "arm"
        else:
            self.cpu_type = "unknown"
        
        log.info("Detected CPU type: %s", self.cpu_type)
        return self.cpu_type

    def _find_intel_rapl_path(self) -> Optional[str]:
        """Find the path to the Intel RAPL package-0 energy counter."""
        base_path = "/sys/class/powercap/"
        if not os.path.exists(base_path):
            return None
        
        try:
            for dir_name in os.listdir(base_path):
                if dir_name.startswith("intel-rapl:"):
                    energy_path = os.path.join(base_path, dir_name, "energy_uj")
                    if os.path.exists(energy_path):
                        try:
                            with open(energy_path, 'r') as f:
                                f.read()
                            return energy_path
                        except PermissionError:
                            log.warning("Intel RAPL file found (%s) but permission denied.", energy_path)
                            return None
        except Exception as e:
            log.error("Error while searching for Intel RAPL path: %s", e)
        return None

    def _find_amd_energy_path(self) -> Optional[str]:
        """Find the path to the AMD energy counter via hwmon."""
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
                            with open(energy_path, 'r') as f:
                                f.read()
                            return energy_path
                        except PermissionError:
                            log.warning("AMD Energy file found (%s) but permission denied.", energy_path)
                            return None
        except Exception as e:
            log.error("Error while searching for AMD Energy path: %s", e)
        return None

    def _check_metric_availability(self):
        """Check available metrics for this CPU type and sets availability flags."""
        self.utilization_available = True
        self.temp_monitoring_available = hasattr(psutil, 'sensors_temperatures')
        if not self.temp_monitoring_available:
            log.warning("psutil.sensors_temperatures not found. Disabling temperature monitoring.")
        
        self._check_power_availability()

    def _check_power_availability(self):
        """
        Routes power metric availability check based on detected CPU type.
        Sets self.power_monitoring_available and self.energy_counter_path as needed.
        """
        self.power_monitoring_available = False
        
        if self.cpu_type == "intel":
            self.energy_counter_path = self._find_intel_rapl_path()
            if self.energy_counter_path:
                log.info("[Intel] Intel RAPL enabled at: %s", self.energy_counter_path)
                self.power_monitoring_available = True
            else:
                log.warning("[Intel] No Intel RAPL found. Power monitoring disabled.")
        
        elif self.cpu_type == "amd":
            self.energy_counter_path = self._find_amd_energy_path()
            if self.energy_counter_path:
                log.info("[AMD] AMD Energy enabled at: %s", self.energy_counter_path)
                self.power_monitoring_available = True
            else:
                log.warning("[AMD] No AMD Energy file found. Power monitoring disabled.")
        
        elif self.cpu_type == "arm":
            log.warning("[ARM] No power metric collection implemented for generic ARM CPUs.")
        
        else:
            log.warning("[%s] Unknown CPU type. Power monitoring disabled.", self.cpu_type)

    def _read_energy_uj(self) -> Optional[int]:
        """Read the raw energy counter from the determined path."""
        try:
            with open(self.energy_counter_path, 'r') as f:
                return int(f.read().strip())
        except PermissionError:
            log.error("Permission denied reading energy file: %s. Disabling power monitoring.", 
                     self.energy_counter_path)
            self.power_monitoring_available = False
            return None
        except Exception as e:
            log.error("Failed to read energy file: %s. Disabling power monitoring.", e)
            self.power_monitoring_available = False
            return None

    def _monitor_process(self):
        """Main monitoring loop - write to CSV and update metrics in real-time."""
        # Setup CSV file in results directory
        if self.results_dir:
            self.csv_filepath = generate_csv_filepath(self.results_dir, "cpu_profiler")
        else:
            results_dir = get_results_directory()
            self.csv_filepath = generate_csv_filepath(results_dir, "cpu_profiler")
        
        self.metrics["csv_filepath"] = self.csv_filepath
        log.info("Writing CPU samples to: %s", self.csv_filepath)
        
        start_time = time.perf_counter()
        
        # Initialize metric accumulators
        cpu_acc = MetricAccumulator(track_nonzero=True)
        mem_acc = MetricAccumulator()
        mem_pct_acc = MetricAccumulator()
        temp_acc = MetricAccumulator()
        
        total_energy_joules = 0.0
        prev_energy_uj = None
        
        fieldnames = [
            "timestamp", "cpu_utilization_percent", "memory_used_mb", 
            "memory_utilization_percent", "energy_uj", "cpu_temp_c"
        ]
        
        with CSVWriter(self.csv_filepath, fieldnames=fieldnames) as csv_writer:
            while not self._stop_event.is_set():
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
                
                # Update accumulators
                cpu_acc.add(cpu_percent)
                mem_acc.add(sample["memory_used_mb"])
                mem_pct_acc.add(vmem.percent)
                
                # Power monitoring
                if self.power_monitoring_available:
                    energy_uj = self._read_energy_uj()
                    if energy_uj is not None:
                        sample["energy_uj"] = energy_uj
                        if prev_energy_uj is not None:
                            energy_delta_j = (energy_uj - prev_energy_uj) / 1_000_000.0
                            if energy_delta_j > 0:
                                total_energy_joules += energy_delta_j
                        prev_energy_uj = energy_uj
                
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
                                temp_acc.add(cpu_temp)
                    except Exception as e:
                        log.warning("Temperature read failed: %s", e)
                        self.temp_monitoring_available = False
                
                # Write sample to CSV
                csv_writer.write_sample(sample)
                
                # Update cached metrics in real-time
                cpu_stats = cpu_acc.get_stats(use_nonzero=True)
                self.metrics["num_samples"] = cpu_acc.count
                self.metrics["average_cpu_utilization_percent"] = cpu_stats["average"]
                self.metrics["peak_cpu_utilization_percent"] = cpu_stats["peak"]
                self.metrics["min_cpu_utilization_percent"] = cpu_stats["min"]
                
                mem_stats = mem_acc.get_stats()
                self.metrics["average_memory_mb"] = mem_stats["average"]
                self.metrics["peak_memory_mb"] = mem_stats["peak"]
                self.metrics["min_memory_mb"] = mem_stats["min"]
                
                mem_pct_stats = mem_pct_acc.get_stats()
                self.metrics["average_memory_utilization_percent"] = mem_pct_stats["average"]
                self.metrics["peak_memory_utilization_percent"] = mem_pct_stats["peak"]
                self.metrics["min_memory_utilization_percent"] = mem_pct_stats["min"]

                if temp_acc.count > 0:
                    temp_stats = temp_acc.get_stats()
                    self.metrics["average_cpu_temp_c"] = temp_stats["average"]
                    self.metrics["peak_cpu_temp_c"] = temp_stats["peak"]
                    self.metrics["min_cpu_temp_c"] = temp_stats["min"]
                
                if self.power_monitoring_available and prev_energy_uj is not None:
                    self.metrics["total_energy_joules"] = total_energy_joules
                    if rel_timestamp > 0:
                        self.metrics["average_power_watts"] = total_energy_joules / rel_timestamp
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                self.metrics["sampling_interval"] = self.sampling_interval
                
                # Sleep to maintain sampling interval
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    self._stop_event.wait(sleep_duration)
