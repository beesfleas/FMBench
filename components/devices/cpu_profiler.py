import time
import psutil
import threading
import platform
from .base import BaseDeviceProfiler
from collections import defaultdict

class LocalCpuProfiler(BaseDeviceProfiler):
    """
    Profiler for local CPU and RAM using psutil.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Configurable sampling rate (default: 1.0 second)
        self.sampling_interval = config.get("cpu_sampling_interval", 
                                            config.get("sampling_interval", 1.0))
        
        # Store only raw samples
        self.samples = []
        self._start_time = None
        self.power_monitoring_available = True  # Assume available until proven otherwise
        self.temp_monitoring_available = True

        # Initialize psutil for CPU percent.
        # Call once before starting to get a baseline.
        psutil.cpu_percent(interval=None)
        print("Initialized CPU Profiler. Collecting system-wide metrics.")

    def _monitor_process(self):
        """
        Lightweight monitoring loop - just collect raw data.
        No calculations or aggregations happen here.
        """
        if self._start_time is None:
            self._start_time = time.perf_counter()
            
        while self._is_monitoring:
            monitor_start_time = time.perf_counter()
            
            # 1. System-wide CPU Utilization
            # Non-blocking call, compares to last time it was called
            cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
            
            # 2. System-wide Virtual Memory (RAM)
            vmem = psutil.virtual_memory()
            memory_used_mb = vmem.used / (1024 * 1024)
            memory_percent = vmem.percent
            
            # 3. System-wide CPU Power (Watts)
            power_watts = None
            if self.power_monitoring_available:
                try:
                    # 'psutil.sensors_power()' returns a list, we look for 'core'
                    power_info = psutil.sensors_power()
                    if power_info and hasattr(power_info, 'core'):
                         power_watts = power_info.core.current
                    # Fallback for systems without named fields (e.g., some Windows)
                    elif power_info:
                        power_watts = power_info[0].current
                        
                except Exception as e:
                    if self._is_monitoring: # Avoid printing error if stopping
                        print(f"Warning: Could not read CPU power: {e}. Disabling power monitoring.")
                    self.power_monitoring_available = False
            
            # 4. System-wide CPU Temperature (Celsius)
            cpu_temp_c = None
            if self.temp_monitoring_available:
                try:
                    temps = psutil.sensors_temperatures()
                    if not temps:
                        raise Exception("No temperature sensors found by psutil.")
                    
                    # Try to find the main CPU package temperature
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
                    
                    # Fallback: if no specific CPU temp, grab the first available
                    if not found_temp:
                        cpu_temp_c = list(temps.values())[0][0].current
                        
                except Exception as e:
                    if self._is_monitoring:
                        print(f"Warning: Could not read CPU temperature: {e}. Disabling temperature monitoring.")
                    self.temp_monitoring_available = False

            # 5. Get timestamp relative to the start
            timestamp = time.perf_counter() - self._start_time
            
            self.samples.append({
                "timestamp": timestamp,
                "cpu_utilization_percent": cpu_percent,
                "memory_used_mb": memory_used_mb,
                "memory_utilization_percent": memory_percent,
                "power_watts": power_watts,
                "cpu_temp_c": cpu_temp_c
            })
            
            # Sleep to maintain the desired sampling interval
            elapsed = time.perf_counter() - monitor_start_time
            sleep_duration = self.sampling_interval - elapsed
            time.sleep(sleep_duration)

    def get_metrics(self):
        """
        Process all raw samples and return a structured dictionary
        identical to the NvidiaGpuProfiler.
        """
        if not self.samples:
            return {"error": "No metrics collected."}

        num_samples = len(self.samples)
        
        # Calculate precise monitoring duration from samples
        monitoring_duration = self.samples[-1]['timestamp'] - self.samples[0]['timestamp']
        
        # Use defaultdict to simplify aggregation
        stats = defaultdict(list)
        power_values = []
        temp_values = []
        
        for sample in self.samples:
            stats["cpu_utilization_percent"].append(sample["cpu_utilization_percent"])
            stats["memory_used_mb"].append(sample["memory_used_mb"])
            stats["memory_utilization_percent"].append(sample["memory_utilization_percent"])
            if sample["power_watts"] is not None:
                power_values.append(sample["power_watts"])
            if sample["cpu_temp_c"] is not None:
                temp_values.append(sample["cpu_temp_c"])

        metrics = {
            # Device Info
            "device_name": f"{platform.processor()} (CPU/RAM)",
            
            # Raw data for detailed analysis
            "raw_samples": self.samples,
            
            # Summary statistics
            "num_samples": num_samples,
            "monitoring_duration_seconds": monitoring_duration,
            "sampling_interval": self.sampling_interval,

            # CPU Utilization statistics
            "peak_cpu_utilization_percent": max(stats["cpu_utilization_percent"]),
            "average_cpu_utilization_percent": sum(stats["cpu_utilization_percent"]) / num_samples,
            
            # Memory statistics
            "peak_memory_mb": max(stats["memory_used_mb"]),
            "average_memory_mb": sum(stats["memory_used_mb"]) / num_samples,
            "peak_memory_utilization_percent": max(stats["memory_utilization_percent"]),
            "average_memory_utilization_percent": sum(stats["memory_utilization_percent"]) / num_samples,
        }
        
        # Add power metrics only if they were successfully collected
        if power_values:
            metrics["peak_power_watts"] = max(power_values)
            metrics["average_power_watts"] = sum(power_values) / len(power_values)
            metrics["min_power_watts"] = min(power_values)
            
            # Estimate energy consumption (J = W * s)
            # This is a rough trapezoidal integration
            total_energy_joules = 0
            for i in range(1, len(self.samples)):
                if self.samples[i]['power_watts'] is not None and self.samples[i-1]['power_watts'] is not None:
                    dt = self.samples[i]['timestamp'] - self.samples[i-1]['timestamp']
                    avg_power = (self.samples[i]['power_watts'] + self.samples[i-1]['power_watts']) / 2
                    total_energy_joules += avg_power * dt
            
            metrics["total_energy_joules"] = total_energy_joules
            metrics["total_energy_wh"] = total_energy_joules / 3600.0

        if temp_values:
            metrics["peak_temp_c"] = max(temp_values)
            metrics["average_temp_c"] = sum(temp_values) / len(temp_values)
            metrics["min_temp_c"] = min(temp_values)

        return metrics