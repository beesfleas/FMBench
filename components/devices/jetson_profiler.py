import time
import psutil
from pathlib import Path
from typing import Optional
from .base import BaseDeviceProfiler
from .profiler_utils import CSVWriter, MetricAccumulator, generate_csv_filepath, get_results_directory
import logging

log = logging.getLogger(__name__)


class JetsonProfiler(BaseDeviceProfiler):
    """
    Profiler for NVIDIA Jetson devices using 'jetson-stats' (jtop).
    Supports JetPack 5 & 6 across all Jetson models (Orin, Xavier, Nano).
    
    Falls back to psutil for CPU/RAM if jetson-stats is not available.
    """
    
    def __init__(self, config, profiler_manager=None, results_dir: Optional[Path] = None):
        super().__init__(config, results_dir)
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.device_name = "NVIDIA Jetson"
        
        # Check for jtop availability
        self.jtop_wrapper = None
        self.has_jtop = False
        try:
            from jtop import jtop
            self.jtop_wrapper = jtop
            self.has_jtop = True
            log.debug("Jetson-stats (jtop) library found")
        except ImportError:
            log.warning("jetson-stats not found. Run 'pip install jetson-stats'. "
                       "Only CPU/RAM metrics will be available.")

        # Cached metrics
        self.metrics = {
            "device_name": self.device_name,
            "num_samples": 0,
            "csv_filepath": None,
        }
        self.total_energy_joules = 0.0

    def get_device_info(self) -> str:
        if self.has_jtop:
            return f"{self.device_name} (jtop)"
        return f"{self.device_name} (psutil-only)"

    def _monitor_process(self):
        """Collect metrics using jtop if available, otherwise fallback to psutil."""
        # Setup CSV file in results directory
        if self.results_dir:
            self.csv_filepath = generate_csv_filepath(self.results_dir, "jetson_profiler")
        else:
            results_dir = get_results_directory()
            self.csv_filepath = generate_csv_filepath(results_dir, "jetson_profiler")
        
        self.metrics["csv_filepath"] = self.csv_filepath
        log.info("Writing Jetson samples to: %s", self.csv_filepath)
        
        self.total_energy_joules = 0.0
        start_time = time.perf_counter()
        
        # Initialize accumulators
        accumulators = {
            "cpu": MetricAccumulator(),
            "mem": MetricAccumulator(),
            "gpu": MetricAccumulator(),
            "gpu_power": MetricAccumulator(),
            "sys_power": MetricAccumulator(),
            "temp": MetricAccumulator(),
            "cpu_freq": MetricAccumulator(),
            "gpu_freq": MetricAccumulator(),
        }
        
        # If jtop is available, we use its context manager
        if self.has_jtop and self.jtop_wrapper:
            try:
                with self.jtop_wrapper() as jetson:
                    if jetson.ok():
                        board_info = jetson.board.get('info', {}).get('machine', 'Unknown Jetson')
                        log.debug("jtop connected: %s", board_info)
                        self._collection_loop(start_time, jetson, accumulators)
                    else:
                        log.warning("jtop failed to initialize (check permissions?), falling back to psutil")
                        self._collection_loop(start_time, None, accumulators)
            except Exception as e:
                log.warning("Error running jtop: %s. Falling back to psutil.", e)
                self._collection_loop(start_time, None, accumulators)
        else:
            self._collection_loop(start_time, None, accumulators)

    def _collection_loop(self, start_time, jetson, accumulators):
        """
        Core collection loop. 
        Args:
            start_time: Start time of monitoring
            jetson: initialized jtop object or None
            accumulators: dictionary of MetricAccumulator instances
        """
        last_sample_time = start_time
        
        with CSVWriter(self.csv_filepath) as csv_writer:
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                
                # Calculate time delta for energy integration
                time_delta = loop_start - last_sample_time
                last_sample_time = loop_start

                sample = {"timestamp": rel_timestamp}

                # --- CPU & Memory (Always available via psutil) ---
                try:
                    cpu_util = psutil.cpu_percent(interval=None)
                    vmem = psutil.virtual_memory()
                    sample["cpu_utilization_percent"] = cpu_util
                    sample["memory_used_mb"] = vmem.used / (1024 * 1024)
                    sample["memory_utilization_percent"] = vmem.percent
                    
                    accumulators["cpu"].add(cpu_util)
                    accumulators["mem"].add(sample["memory_used_mb"])
                except Exception as e:
                    log.debug("psutil error: %s", e)

                # --- GPU, Power, Temp (via jtop) ---
                if jetson and jetson.ok():
                    try:
                        # GPU Util
                        gpu_val = jetson.stats.get('GPU', 0)
                        sample["gpu_utilization_percent"] = gpu_val
                        accumulators["gpu"].add(gpu_val)

                        # GPU Frequency
                        gpu_freq_khz = 0
                        if hasattr(jetson, 'gpu'):
                            for arch in jetson.gpu:
                                freq_data = jetson.gpu[arch].get('freq', {})
                                if 'cur' in freq_data:
                                    gpu_freq_khz = freq_data['cur']
                                    break
                        
                        if gpu_freq_khz:
                            mhz = gpu_freq_khz / 1000.0
                            sample["gpu_frequency_mhz"] = mhz
                            accumulators["gpu_freq"].add(mhz)
                        
                        # CPU Frequency
                        cpu_freqs = []
                        if hasattr(jetson, 'cpu'):
                            cpu_info = jetson.cpu.get('cpu', [])
                            for core in cpu_info:
                                if core.get('online') and 'freq' in core:
                                    cpu_freqs.append(core['freq'].get('cur', 0))
                        
                        if cpu_freqs:
                            avg_cpu_freq = sum(cpu_freqs) / len(cpu_freqs)
                            mhz = avg_cpu_freq / 1000.0
                            sample["cpu_frequency_mhz"] = mhz
                            accumulators["cpu_freq"].add(mhz)

                        # Power
                        power_stats = getattr(jetson, 'power', {})
                        total_power_mw = 0
                        gpu_power_mw = 0
                        
                        if power_stats:
                            if 'tot' in power_stats:
                                total_power_mw = power_stats['tot'].get('power', 0)
                            
                            rails = power_stats.get('rail', {})
                            if 'VDD_GPU_SOC' in rails:
                                gpu_power_mw = rails['VDD_GPU_SOC'].get('power', 0)
                            
                        if total_power_mw:
                            watts = total_power_mw / 1000.0
                            sample["system_power_watts"] = watts
                            accumulators["sys_power"].add(watts)
                            
                        if gpu_power_mw:
                            watts = gpu_power_mw / 1000.0
                            sample["gpu_power_watts"] = watts
                            accumulators["gpu_power"].add(watts)

                        # Energy Integration
                        power_for_energy = sample.get("system_power_watts", 
                                                     sample.get("gpu_power_watts", 0))
                        if power_for_energy > 0 and time_delta > 0:
                            self.total_energy_joules += power_for_energy * time_delta

                        # Temp
                        temps = jetson.stats.get('Temp', {})
                        if 'GPU' in temps:
                            sample["gpu_temp_c"] = temps['GPU']
                            accumulators["temp"].add(temps['GPU'])
                            
                    except Exception as e:
                        log.warning("Error reading jtop stats: %s", e)

                # Write sample to CSV
                csv_writer.write_sample(sample)

                # Update aggregated metrics
                self._update_metrics(accumulators, rel_timestamp)

                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    self._stop_event.wait(sleep_duration)

    def _update_metrics(self, accumulators, duration):
        """Update self.metrics dictionary with aggregated stats."""
        self.metrics["num_samples"] = accumulators["cpu"].count
        self.metrics["monitoring_duration_seconds"] = duration
        
        mappings = [
            ("cpu_utilization", "cpu", "_percent"),
            ("memory", "mem", "_mb"),
            ("gpu_utilization", "gpu", "_percent"),
            ("gpu_power", "gpu_power", "_watts"),
            ("system_power", "sys_power", "_watts"),
            ("gpu_temp", "temp", "_c"),
            ("cpu_frequency", "cpu_freq", "_mhz"),
            ("gpu_frequency", "gpu_freq", "_mhz"),
        ]
        
        for metric_prefix, acc_key, unit_suffix in mappings:
            acc = accumulators[acc_key]
            if acc.count > 0:
                stats = acc.get_stats()
                self.metrics[f"average_{metric_prefix}{unit_suffix}"] = stats["average"]
                self.metrics[f"peak_{metric_prefix}{unit_suffix}"] = stats["peak"]

        self.metrics["total_energy_joules"] = self.total_energy_joules
