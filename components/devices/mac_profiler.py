import subprocess
import time
import psutil
import os
import csv
import logging
import tempfile
import re
from .base import BaseDeviceProfiler

log = logging.getLogger(__name__)

class MacProfiler(BaseDeviceProfiler):
    """
    Profiler for macOS (Intel and Apple Silicon).
    Uses powermetrics for CPU/GPU power and thermal data.
    Writes samples to CSV and calculates metrics in real-time.
    """
    def __init__(self, config, profiler_manager=None):
        super().__init__(config)
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.sampling_interval_ms = int(self.sampling_interval * 1000)
        self.device_name = "macOS"
        self.csv_filepath = None
        
        # Cached metrics
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        
        self.powermetrics_process = None
        self._can_read_powermetrics = True
        self.last_known_metrics = {}
        
        psutil.cpu_percent(interval=None)
        log.info("Initialized macOS Profiler (powermetrics-based)")

    def get_device_info(self) -> str:
        system_info = ""
        try:
            result = subprocess.run(["system_profiler", "SPHardwareDataType"], 
                                  capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if 'Model' in line or 'Processor' in line or 'Memory' in line:
                    system_info += line.strip() + " | "
        except Exception as e:
            log.debug("Could not get detailed system info: %s", e)
        
        return f"{self.device_name} {system_info}".strip()

    def _start_powermetrics_process(self):
        """Starts the long-running powermetrics process."""
        if not self._can_read_powermetrics:
            return

        # Check if we're already running as root (e.g., via sudo)
        is_root = os.geteuid() == 0 if hasattr(os, 'geteuid') else False
        cmd = ['powermetrics'] if is_root else ['sudo', 'powermetrics']
        cmd.extend([
            '--samplers', 'cpu_power,gpu_power,thermal',
            '-i', f'{self.sampling_interval_ms}'
        ])
        log.debug("Starting powermetrics: %s", ' '.join(cmd))
        
        try:
            self.powermetrics_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                bufsize=1
            )
            log.debug("powermetrics PID: %s", self.powermetrics_process.pid)
            
            time.sleep(0.1)
            poll_result = self.powermetrics_process.poll()
            if poll_result is not None:
                stderr_output = self.powermetrics_process.stderr.read()
                if 'Operation not permitted' in stderr_output:
                    log.warning("powermetrics requires sudo, power metrics unavailable")
                else:
                    log.warning("powermetrics failed to start: %s", stderr_output)
                self._can_read_powermetrics = False
                self.powermetrics_process = None

        except Exception as e:
            log.warning("powermetrics command failed: %s, disabling power metrics", e)
            self._can_read_powermetrics = False
            self.powermetrics_process = None

    def _parse_powermetrics_block(self, block: str) -> dict:
        """Parse a powermetrics output block using regex."""
        metrics = {}
        try:
            # CPU power (mW -> W)
            e_power_match = re.search(r'E-Cluster Power:\s*(\d+)\s*mW', block)
            if e_power_match:
                metrics['e_cluster_power_watts'] = float(e_power_match.group(1)) / 1000.0
            
            p_power_match = re.search(r'P-Cluster Power:\s*(\d+)\s*mW', block)
            if p_power_match:
                metrics['p_cluster_power_watts'] = float(p_power_match.group(1)) / 1000.0

            # GPU power (mW -> W)
            gpu_power_match = re.search(r'GPU Power:\s*(\d+)\s*mW', block)
            if gpu_power_match:
                metrics['gpu_power_watts'] = float(gpu_power_match.group(1)) / 1000.0

            # GPU utilization (%)
            gpu_util_match = re.search(r'GPU HW active residency:\s*([\d\.]+)%', block)
            if gpu_util_match:
                metrics['gpu_utilization_percent'] = float(gpu_util_match.group(1))
            
            # Temperature
            temp_match = re.search(r'CPU die temperature:\s*([\d\.]+)\s*C', block)
            if temp_match:
                metrics['cpu_temp_c'] = float(temp_match.group(1))
                
        except Exception as e:
            log.debug("Error parsing powermetrics: %s", e)
            
        return metrics

    def _monitor_process(self):
        """Collect macOS system metrics and write to CSV."""
        # Setup CSV file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        self.csv_filepath = os.path.join(temp_dir, f"mac_profiler_{timestamp}.csv")
        self.metrics["csv_filepath"] = self.csv_filepath
        
        log.info("Writing samples to: %s", self.csv_filepath)
        
        # Start powermetrics process
        self._start_powermetrics_process()
        
        start_time = time.perf_counter()
        csv_file = None
        csv_writer = None
        current_block = ""
        loop_count = 0
        
        # Metric accumulators
        cpu_util_values = []
        mem_values = []
        mem_pct_values = []
        e_power_values = []
        p_power_values = []
        gpu_power_values = []
        gpu_util_values = []
        temp_values = []
        
        try:
            while self._is_monitoring:
                loop_count += 1
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                
                sample = {"timestamp": rel_timestamp}
                
                # Read from powermetrics
                if self.powermetrics_process and self.powermetrics_process.stdout:
                    try:
                        line = self.powermetrics_process.stdout.readline()
                        if not line:
                            if self._is_monitoring:
                                log.warning("powermetrics process closed unexpectedly")
                            self._can_read_powermetrics = False
                            self.powermetrics_process = None
                        else:
                            current_block += line
                            if '***' in line:
                                power_metrics = self._parse_powermetrics_block(current_block)
                                self.last_known_metrics.update(power_metrics)
                                current_block = ""
                    except Exception as e:
                        if self._is_monitoring:
                            log.debug("Error reading powermetrics: %s", e, exc_info=True)
                        self._can_read_powermetrics = False
                        self.powermetrics_process = None
                
                # CPU and memory via psutil
                try:
                    cpu_util = psutil.cpu_percent(interval=None)
                    vmem = psutil.virtual_memory()
                    sample["cpu_utilization_percent"] = cpu_util
                    sample["memory_used_mb"] = vmem.used / (1024 * 1024)
                    sample["memory_utilization_percent"] = vmem.percent
                except Exception as e:
                    log.debug("Failed to read CPU/memory: %s", e, exc_info=True)
                
                # Add powermetrics data to sample
                sample.update(self.last_known_metrics)
                
                # Write to CSV
                if csv_file is None:
                    csv_file = open(self.csv_filepath, 'w', newline='')
                    # Get all possible fieldnames from sample and known metrics
                    all_fieldnames = set(sample.keys())
                    # Add common powermetrics fields that might appear later
                    all_fieldnames.update([
                        'e_cluster_power_watts', 'p_cluster_power_watts', 
                        'gpu_power_watts', 'gpu_utilization_percent', 'cpu_temp_c'
                    ])
                    csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(all_fieldnames))
                    csv_writer.writeheader()
                    log.debug("CSV header written (%d fields)", len(all_fieldnames))
                
                csv_writer.writerow(sample)
                csv_file.flush()
                
                # Update accumulators
                if "cpu_utilization_percent" in sample:
                    cpu_util_values.append(sample["cpu_utilization_percent"])
                if "memory_used_mb" in sample:
                    mem_values.append(sample["memory_used_mb"])
                if "memory_utilization_percent" in sample:
                    mem_pct_values.append(sample["memory_utilization_percent"])
                if "e_cluster_power_watts" in sample:
                    e_power_values.append(sample["e_cluster_power_watts"])
                if "p_cluster_power_watts" in sample:
                    p_power_values.append(sample["p_cluster_power_watts"])
                if "gpu_power_watts" in sample:
                    gpu_power_values.append(sample["gpu_power_watts"])
                if "gpu_utilization_percent" in sample:
                    gpu_util_values.append(sample["gpu_utilization_percent"])
                if "cpu_temp_c" in sample:
                    temp_values.append(sample["cpu_temp_c"])
                
                # Update cached metrics
                self.metrics["num_samples"] = len(cpu_util_values)
                if cpu_util_values:
                    self.metrics["average_cpu_utilization_percent"] = sum(cpu_util_values) / len(cpu_util_values)
                    self.metrics["peak_cpu_utilization_percent"] = max(cpu_util_values)
                if mem_values:
                    self.metrics["average_memory_mb"] = sum(mem_values) / len(mem_values)
                    self.metrics["peak_memory_mb"] = max(mem_values)
                if mem_pct_values:
                    self.metrics["average_memory_utilization_percent"] = sum(mem_pct_values) / len(mem_pct_values)
                    self.metrics["peak_memory_utilization_percent"] = max(mem_pct_values)
                if e_power_values:
                    self.metrics["average_e_cluster_power_watts"] = sum(e_power_values) / len(e_power_values)
                    self.metrics["peak_e_cluster_power_watts"] = max(e_power_values)
                if p_power_values:
                    self.metrics["average_p_cluster_power_watts"] = sum(p_power_values) / len(p_power_values)
                    self.metrics["peak_p_cluster_power_watts"] = max(p_power_values)
                if gpu_power_values:
                    self.metrics["average_gpu_power_watts"] = sum(gpu_power_values) / len(gpu_power_values)
                    self.metrics["peak_gpu_power_watts"] = max(gpu_power_values)
                if gpu_util_values:
                    self.metrics["average_gpu_utilization_percent"] = sum(gpu_util_values) / len(gpu_util_values)
                    self.metrics["peak_gpu_utilization_percent"] = max(gpu_util_values)
                if temp_values:
                    self.metrics["average_cpu_temp_c"] = sum(temp_values) / len(temp_values)
                    self.metrics["peak_cpu_temp_c"] = max(temp_values)
                    self.metrics["min_cpu_temp_c"] = min(temp_values)
                
                self.metrics["monitoring_duration_seconds"] = rel_timestamp
                
                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        
        except Exception as e:
            log.error("Exception in monitoring loop: %s", e, exc_info=True)
            raise
        finally:
            if csv_file:
                csv_file.close()
            if self.powermetrics_process:
                try:
                    self.powermetrics_process.terminate()
                    self.powermetrics_process.wait(timeout=1.0)
                except Exception as e:
                    log.debug("Error terminating powermetrics: %s", e, exc_info=True)

    def get_metrics(self) -> dict:
        """Return cached metrics."""
        return self.metrics

