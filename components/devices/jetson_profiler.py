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
        
        start_time = time.perf_counter()
        
        # Metric accumulators
        cpu_values = []
        mem_values = []
        gpu_util_values = []
        gpu_power_values = []
        gpu_temp_values = []
        cpu_freq_values = []
        gpu_freq_values = []
        
        # If jtop is available, we use its context manager
        if self.has_jtop and self.jtop_wrapper:
            try:
                with self.jtop_wrapper() as jetson:
                    if jetson.ok():
                        board_info = jetson.board.get('info', {}).get('machine', 'Unknown Jetson')
                        log.info(f"jtop connected: {board_info}")
                        self._collection_loop(
                            start_time, 
                            jetson=jetson,
                            cpu_values=cpu_values,
                            mem_values=mem_values,
                            gpu_util_values=gpu_util_values,
                            gpu_power_values=gpu_power_values,
                            gpu_temp_values=gpu_temp_values,
                            cpu_freq_values=cpu_freq_values,
                            gpu_freq_values=gpu_freq_values
                        )
                    else:
                        log.error("jtop failed to initialize (check permissions?). Falling back to psutil.")
                        self._collection_loop(
                            start_time, 
                            jetson=None, 
                            cpu_values=cpu_values, 
                            mem_values=mem_values, 
                            gpu_util_values=gpu_util_values, 
                            gpu_power_values=gpu_power_values, 
                            gpu_temp_values=gpu_temp_values,
                            cpu_freq_values=cpu_freq_values,
                            gpu_freq_values=gpu_freq_values
                        )
            except Exception as e:
                log.error(f"Error running jtop: {e}. Falling back to psutil.")
                self._collection_loop(
                    start_time, 
                    jetson=None, 
                    cpu_values=cpu_values, 
                    mem_values=mem_values, 
                    gpu_util_values=gpu_util_values, 
                    gpu_power_values=gpu_power_values, 
                    gpu_temp_values=gpu_temp_values,
                    cpu_freq_values=cpu_freq_values,
                    gpu_freq_values=gpu_freq_values
                )
        else:
            self._collection_loop(
                start_time, 
                jetson=None, 
                cpu_values=cpu_values, 
                mem_values=mem_values, 
                gpu_util_values=gpu_util_values, 
                gpu_power_values=gpu_power_values, 
                gpu_temp_values=gpu_temp_values,
                cpu_freq_values=cpu_freq_values,
                gpu_freq_values=gpu_freq_values
            )

    def _collection_loop(self, start_time, jetson, cpu_values, mem_values, gpu_util_values, gpu_power_values, gpu_temp_values, cpu_freq_values, gpu_freq_values):
        """
        Core collection loop. 
        Args:
            start_time: Start time of monitoring
            jetson: initialized jtop object or None
            ... accumulators ...
        """
        csv_file = None
        csv_writer = None
        
        try:
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()
                rel_timestamp = loop_start - start_time
                sample = {"timestamp": rel_timestamp}

                # --- CPU & Memory (Always available via psutil) ---
                try:
                    cpu_util = psutil.cpu_percent(interval=None)
                    vmem = psutil.virtual_memory()
                    sample["cpu_utilization_percent"] = cpu_util
                    sample["memory_used_mb"] = vmem.used / (1024 * 1024)
                    sample["memory_utilization_percent"] = vmem.percent
                    
                    cpu_values.append(cpu_util)
                    mem_values.append(sample["memory_used_mb"])
                except Exception as e:
                    log.debug(f"psutil error: {e}")

                # --- GPU, Power, Temp (via jtop) ---
                if jetson and jetson.ok():
                    try:
                        # GPU Util
                        # jetson.stats['GPU'] is usually percent integer
                        gpu_val = jetson.stats.get('GPU', 0)
                        sample["gpu_utilization_percent"] = gpu_val
                        gpu_util_values.append(gpu_val)

                        # GPU Frequency
                        # jetson.gpu is a dict like {'ga10b': {'freq': {'cur': ...}}}
                        # We iterate to find the first available GPU architecture
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
                            gpu_freq_values.append(mhz)
                        
                        # CPU Frequency
                        # jetson.cpu['cpu'] is a list of dicts for each core
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
                            cpu_freq_values.append(mhz)

                        # Power
                        # jetson.power['tot']['power'] is total power in mW
                        # jetson.power['rail']['VDD_GPU_SOC']['power'] is GPU/SOC power in mW
                        power_stats = getattr(jetson, 'power', {})
                        total_power_mw = 0
                        gpu_power_mw = 0
                        
                        if power_stats:
                            # Total System Power
                            if 'tot' in power_stats:
                                total_power_mw = power_stats['tot'].get('power', 0)
                            
                            # GPU Power (approximate using VDD_GPU_SOC if available)
                            rails = power_stats.get('rail', {})
                            if 'VDD_GPU_SOC' in rails:
                                gpu_power_mw = rails['VDD_GPU_SOC'].get('power', 0)
                            
                        if total_power_mw:
                            sample["system_power_watts"] = total_power_mw / 1000.0
                            
                        if gpu_power_mw:
                             watts = gpu_power_mw / 1000.0
                             sample["gpu_power_watts"] = watts
                             gpu_power_values.append(watts)
                        elif total_power_mw:
                             # Fallback if no specific GPU rail found (common on some boards)
                             # But usually VDD_GPU_SOC is there for Orin.
                             # If not, we might leave gpu_power_watts empty or use a fraction?
                             # For now, let's only report if we found the specific rail to be accurate.
                             pass

                        # Temp
                        # jetson.stats['Temp'] -> {'GPU': 34.5, ...}
                        temps = jetson.stats.get('Temp', {})
                        if 'GPU' in temps:
                            sample["gpu_temp_c"] = temps['GPU']
                            gpu_temp_values.append(temps['GPU'])
                            
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
                        # Don't break loop, just skip writing
                        
                if csv_writer:
                    try:
                        csv_writer.writerow(sample)
                        csv_file.flush()
                    except Exception as e:
                        log.error(f"Failed to write row: {e}")

                # --- Update Aggregates ---
                self._update_metrics(cpu_values, mem_values, gpu_util_values, gpu_power_values, gpu_temp_values, cpu_freq_values, gpu_freq_values, rel_timestamp)

                # Sleep
                elapsed = time.perf_counter() - loop_start
                sleep_duration = self.sampling_interval - elapsed
                if sleep_duration > 0:
                    self._stop_event.wait(sleep_duration)
        
        finally:
            if csv_file:
                csv_file.close()

    def _update_metrics(self, cpu, mem, gpu, power, temp, cpu_freq, gpu_freq, duration):
        """Update self.metrics dictionary with aggregated stats."""
        self.metrics["num_samples"] = len(cpu)
        self.metrics["monitoring_duration_seconds"] = duration
        
        if cpu:
            self.metrics["average_cpu_utilization_percent"] = sum(cpu) / len(cpu)
            self.metrics["peak_cpu_utilization_percent"] = max(cpu)
        
        if mem:
            self.metrics["average_memory_mb"] = sum(mem) / len(mem)
            self.metrics["peak_memory_mb"] = max(mem)
            
        if gpu:
            self.metrics["average_gpu_utilization_percent"] = sum(gpu) / len(gpu)
            self.metrics["peak_gpu_utilization_percent"] = max(gpu)
            
        if power:
            avg_power = sum(power) / len(power)
            self.metrics["average_gpu_power_watts"] = avg_power
            self.metrics["peak_gpu_power_watts"] = max(power)
            # Energy = Avg Power (W) * Duration (s)
            self.metrics["total_energy_joules"] = avg_power * duration

        if temp:
            self.metrics["average_gpu_temp_c"] = sum(temp) / len(temp)
            self.metrics["peak_gpu_temp_c"] = max(temp)
            
        if cpu_freq:
            self.metrics["average_cpu_frequency_mhz"] = sum(cpu_freq) / len(cpu_freq)
            self.metrics["peak_cpu_frequency_mhz"] = max(cpu_freq)
            
        if gpu_freq:
            self.metrics["average_gpu_frequency_mhz"] = sum(gpu_freq) / len(gpu_freq)
            self.metrics["peak_gpu_frequency_mhz"] = max(gpu_freq)
