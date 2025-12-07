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
    Profiler for NVIDIA Jetson devices using 'jetson-stats' (jtop).
    Supports JetPack 5 & 6 across all Jetson models (Orin, Xavier, Nano).
    
    Falls back to psutil for CPU/RAM if jetson-stats is not available.
    """
    def __init__(self, config, profiler_manager=None):
        super().__init__(config)
        self.profiler_manager = profiler_manager
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.device_name = "NVIDIA Jetson"
        self.csv_filepath = None
        
        # Check for jtop availability
        self.jtop_wrapper = None
        self.has_jtop = False
        try:
            from jtop import jtop
            self.jtop_wrapper = jtop
            self.has_jtop = True
            log.info("Jetson-stats (jtop) library found.")
        except ImportError:
            log.warning("jetson-stats not found. Run 'pip install jetson-stats'. Only CPU/RAM metrics will be available.")

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
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        self.csv_filepath = os.path.join(temp_dir, f"jetson_profiler_{timestamp}.csv")
        self.metrics["csv_filepath"] = self.csv_filepath
        
        log.info(f"Writing Jetson samples to: {self.csv_filepath}")
        
        self.total_energy_joules = 0.0
        start_time = time.perf_counter()
        
        # Initialize stats container
        stats = {
            "cpu": {"count": 0, "sum": 0.0, "max": 0.0},
            "mem": {"count": 0, "sum": 0.0, "max": 0.0},
            "gpu": {"count": 0, "sum": 0.0, "max": 0.0},
            "gpu_power": {"count": 0, "sum": 0.0, "max": 0.0},
            "sys_power": {"count": 0, "sum": 0.0, "max": 0.0},
            "temp": {"count": 0, "sum": 0.0, "max": 0.0},
            "cpu_freq": {"count": 0, "sum": 0.0, "max": 0.0},
            "gpu_freq": {"count": 0, "sum": 0.0, "max": 0.0}
        }
        
        # If jtop is available, we use its context manager
        if self.has_jtop and self.jtop_wrapper:
            try:
                with self.jtop_wrapper() as jetson:
                    if jetson.ok():
                        board_info = jetson.board.get('info', {}).get('machine', 'Unknown Jetson')
                        log.info(f"jtop connected: {board_info}")
                        self._collection_loop(start_time, jetson, stats)
                    else:
                        log.error("jtop failed to initialize (check permissions?). Falling back to psutil.")
                        self._collection_loop(start_time, None, stats)
            except Exception as e:
                log.error(f"Error running jtop: {e}. Falling back to psutil.")
                self._collection_loop(start_time, None, stats)
        else:
            self._collection_loop(start_time, None, stats)

    def _collection_loop(self, start_time, jetson, stats):
        """
        Core collection loop. 
        Args:
            start_time: Start time of monitoring
            jetson: initialized jtop object or None
            stats: dictionary of running statistics
        """
        csv_file = None
        csv_writer = None
        last_sample_time = start_time
        
        try:
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
                    
                    # Update stats
                    stats["cpu"]["count"] += 1
                    stats["cpu"]["sum"] += cpu_util
                    stats["cpu"]["max"] = max(stats["cpu"]["max"], cpu_util)
                    
                    stats["mem"]["count"] += 1
                    stats["mem"]["sum"] += sample["memory_used_mb"]
                    stats["mem"]["max"] = max(stats["mem"]["max"], sample["memory_used_mb"])
                except Exception as e:
                    log.debug(f"psutil error: {e}")

                # --- GPU, Power, Temp (via jtop) ---
                if jetson and jetson.ok():
                    try:
                        # GPU Util
                        # jetson.stats['GPU'] is usually percent integer
                        gpu_val = jetson.stats.get('GPU', 0)
                        sample["gpu_utilization_percent"] = gpu_val
                        
                        stats["gpu"]["count"] += 1
                        stats["gpu"]["sum"] += gpu_val
                        stats["gpu"]["max"] = max(stats["gpu"]["max"], gpu_val)

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
                            stats["gpu_freq"]["count"] += 1
                            stats["gpu_freq"]["sum"] += mhz
                            stats["gpu_freq"]["max"] = max(stats["gpu_freq"]["max"], mhz)
                        
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
                            stats["cpu_freq"]["count"] += 1
                            stats["cpu_freq"]["sum"] += mhz
                            stats["cpu_freq"]["max"] = max(stats["cpu_freq"]["max"], mhz)

                        # Power
                        power_stats = getattr(jetson, 'power', {})
                        total_power_mw = 0
                        gpu_power_mw = 0
                        
                        if power_stats:
                            # Total System Power
                            if 'tot' in power_stats:
                                total_power_mw = power_stats['tot'].get('power', 0)
                            
                            # GPU Power
                            rails = power_stats.get('rail', {})
                            if 'VDD_GPU_SOC' in rails:
                                gpu_power_mw = rails['VDD_GPU_SOC'].get('power', 0)
                            
                        if total_power_mw:
                            watts = total_power_mw / 1000.0
                            sample["system_power_watts"] = watts
                            stats["sys_power"]["count"] += 1
                            stats["sys_power"]["sum"] += watts
                            stats["sys_power"]["max"] = max(stats["sys_power"]["max"], watts)
                            
                        if gpu_power_mw:
                             watts = gpu_power_mw / 1000.0
                             sample["gpu_power_watts"] = watts
                             stats["gpu_power"]["count"] += 1
                             stats["gpu_power"]["sum"] += watts
                             stats["gpu_power"]["max"] = max(stats["gpu_power"]["max"], watts)

                        # Energy Integration
                        power_for_energy = sample.get("system_power_watts", sample.get("gpu_power_watts", 0))
                        if power_for_energy > 0 and time_delta > 0:
                            self.total_energy_joules += power_for_energy * time_delta

                        # Temp
                        temps = jetson.stats.get('Temp', {})
                        if 'GPU' in temps:
                            sample["gpu_temp_c"] = temps['GPU']
                            stats["temp"]["count"] += 1
                            stats["temp"]["sum"] += temps['GPU']
                            stats["temp"]["max"] = max(stats["temp"]["max"], temps['GPU'])
                            
                    except Exception as e:
                        log.warning(f"Error reading jtop stats: {e}")

                # --- Write to CSV ---
                if csv_file is None:
                    try:
                        csv_file = open(self.csv_filepath, 'w', newline='')
                        csv_writer = csv.DictWriter(csv_file, fieldnames=sample.keys())
                        csv_writer.writeheader()
                    except Exception as e:
                        log.error(f"Failed to open CSV file: {e}")
                        
                if csv_writer:
                    try:
                        csv_writer.writerow(sample)
                        csv_file.flush()
                    except Exception as e:
                        log.error(f"Failed to write row: {e}")

                # --- Update Aggregates ---
                self._update_metrics(stats, rel_timestamp)

                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    self._stop_event.wait(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()

    def _update_metrics(self, stats, duration):
        """Update self.metrics dictionary with aggregated stats."""
        self.metrics["num_samples"] = stats["cpu"]["count"]
        self.metrics["monitoring_duration_seconds"] = duration
        
        def update_stat(metric_prefix, stat_key, unit_suffix=""):
            s = stats[stat_key]
            if s["count"] > 0:
                self.metrics[f"average_{metric_prefix}{unit_suffix}"] = s["sum"] / s["count"]
                self.metrics[f"peak_{metric_prefix}{unit_suffix}"] = s["max"]

        update_stat("cpu_utilization", "cpu", "_percent")
        update_stat("memory", "mem", "_mb")
        update_stat("gpu_utilization", "gpu", "_percent")
        update_stat("gpu_power", "gpu_power", "_watts")
        update_stat("system_power", "sys_power", "_watts")
        update_stat("gpu_temp", "temp", "_c")
        update_stat("cpu_frequency", "cpu_freq", "_mhz")
        update_stat("gpu_frequency", "gpu_freq", "_mhz")

        # Use the integrated energy value
        self.metrics["total_energy_joules"] = self.total_energy_joules
