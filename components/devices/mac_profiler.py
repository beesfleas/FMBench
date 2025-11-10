import time
import threading
import subprocess
import re
from .base import BaseDeviceProfiler
import select  # Used for non-blocking reads

# We can now safely re-introduce psutil, as the 'bus error'
# will be fixed in run.py by setting the start_method to 'spawn'.
try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. MacProfiler will not collect CPU/RAM metrics.")
    psutil = None

class MacProfiler(BaseDeviceProfiler):
    """
    Profiler for macOS devices (Apple Silicon M1/M2/M3).
    
    This implementation is fork-safe and uses the best tool for each metric:
    - psutil: CPU Utilization, RAM Usage (safe now)
    - powermetrics: GPU, Power, and Temperature
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.sampling_interval_ms = int(self.sampling_interval * 1000)
        self.samples = []
        self._start_time = None
        self.device_name = self._get_device_name()
        
        # Flags for graceful failure
        self._can_read_psutil = (psutil is not None)
        self._can_read_powermetrics = True
        self.powermetrics_process = None
        self.last_known_metrics = {}

        print(f"Initialized macOS Profiler for: {self.device_name}")
        
        # Start the long-running powermetrics process
        self._start_powermetrics_process()

    def _get_device_name(self):
        """Get the Apple chip name."""
        try:
            model = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
            return f"macOS ({model})"
        except Exception as e:
            print(f"Couldn't get Mac device name: {e}")
            return "macOS (Unknown)"

    def _start_powermetrics_process(self):
        """Starts the long-running powermetrics process."""
        if not self._can_read_powermetrics:
            return

        # --- THIS IS THE FIX for the 'unrecognized sampler' warning ---
        # We remove 'cpu_load' as it's not universally available.
        # We will get CPU utilization from psutil instead.
        cmd = [
            'sudo', 'powermetrics', 
            '--samplers', 'cpu_power,gpu_power,thermal', 
            '-i', f'{self.sampling_interval_ms}'
        ]
        
        try:
            self.powermetrics_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                bufsize=1 # Line-buffered
            )
            
            time.sleep(0.1) # Give it a moment to fail
            if self.powermetrics_process.poll() is not None:
                stderr_output = self.powermetrics_process.stderr.read()
                if 'Operation not permitted' in stderr_output:
                    print("\n" + "*"*60)
                    print("Warning: 'powermetrics' requires sudo.")
                    print("Please re-run this script with 'sudo' for power/GPU metrics.")
                    print("*"*60 + "\n")
                else:
                    print(f"Warning: 'powermetrics' failed to start: {stderr_output}")
                self._can_read_powermetrics = False
                self.powermetrics_process = None

        except Exception as e:
            print(f"Warning: 'powermetrics' command failed: {e}. Disabling.")
            self._can_read_powermetrics = False
            self.powermetrics_process = None

    def _parse_powermetrics(self, block: str) -> dict:
        """
        Parses a text block from 'powermetrics' using regex.
        """
        metrics = {}
        try:
            # Power (mW -> W)
            e_power_match = re.search(r'E-Cluster Power: (\d+) mW', block)
            if e_power_match:
                metrics['e_cluster_power_watts'] = float(e_power_match.group(1)) / 1000.0
            
            p_power_match = re.search(r'P-Cluster Power: (\d+) mW', block)
            if p_power_match:
                metrics['p_cluster_power_watts'] = float(p_power_match.group(1)) / 1000.0

            gpu_power_match = re.search(r'GPU Power: (\d+) mW', block)
            if gpu_power_match:
                metrics['gpu_power_watts'] = float(gpu_power_match.group(1)) / 1000.0

            # GPU Utilization
            gpu_util_match = re.search(r'GPU HW active residency:\s*([\d\.]+)%', block)
            if gpu_util_match:
                metrics['gpu_utilization_percent'] = float(gpu_util_match.group(1))
            
            # Temperature
            temp_match = re.search(r'CPU die temperature: ([\d\.]+) C', block)
            if temp_match:
                metrics['cpu_temp_c'] = float(temp_match.group(1))
                
        except Exception as e:
            print(f"Error parsing powermetrics: {e}")
            
        return metrics

    def _monitor_process(self):
        """
        The core monitoring loop.
        Reads from powermetrics stdout and calls psutil.
        This is now safe because 'run.py' set the 'spawn' start method.
        """
        self._start_time = time.perf_counter()
        
        # Initialize psutil baseline
        if self._can_read_psutil:
            psutil.cpu_percent(interval=None)
            
        current_block = ""
        last_psutil_sample_time = 0
        
        while self._is_monitoring:
            monitor_start_time = time.perf_counter()
            
            # 1. Read from Powermetrics (if available)
            power_metrics = {}
            if self.powermetrics_process and self.powermetrics_process.stdout:
                # Use select for non-blocking read
                ready_to_read, _, _ = select.select([self.powermetrics_process.stdout], [], [], 0.0)
                
                if ready_to_read:
                    try:
                        line = self.powermetrics_process.stdout.readline()
                        if not line: # Process closed
                            if self._is_monitoring:
                                print("powermetrics process closed unexpectedly.")
                            self._can_read_powermetrics = False
                            self.powermetrics_process = None # Stop trying
                        
                        current_block += line
                        if '***' in line:
                            power_metrics = self._parse_powermetrics(current_block)
                            self.last_known_metrics.update(power_metrics)
                            current_block = "" # Reset for next block
                    except Exception as e:
                        if self._is_monitoring:
                            print(f"Error reading powermetrics output: {e}")
                        self._can_read_powermetrics = False
                        self.powermetrics_process = None # Stop trying

            # 2. Read from psutil
            # We do this on *every* loop to ensure we get CPU/RAM
            cpu_percent, mem_used_mb, mem_percent = None, None, None
            if self._can_read_psutil:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory()
                    mem_used_mb = (mem.total - mem.available) / (1024 * 1024)
                    mem_percent = mem.percent
                except Exception as e:
                    if self._is_monitoring:
                        print(f"Warning: Could not read psutil: {e}. Disabling psutil.")
                    self._can_read_psutil = False

            # 3. Collect Sample
            # We collect a sample on every loop, even if powermetrics
            # hasn't reported, to ensure we get timely CPU/RAM data.
            timestamp = time.perf_counter() - self._start_time
            self.samples.append({
                "timestamp": timestamp,
                "cpu_utilization_percent": cpu_percent,
                "memory_used_mb": mem_used_mb,
                "memory_utilization_percent": mem_percent,
                "e_cluster_power_watts": self.last_known_metrics.get('e_cluster_power_watts'),
                "p_cluster_power_watts": self.last_known_metrics.get('p_cluster_power_watts'),
                "gpu_power_watts": self.last_known_metrics.get('gpu_power_watts'),
                "gpu_utilization_percent": self.last_known_metrics.get('gpu_utilization_percent'),
                "cpu_temp_c": self.last_known_metrics.get('cpu_temp_c'),
            })

            # 4. Precise Sleep
            elapsed = time.perf_counter() - monitor_start_time
            sleep_duration = self.sampling_interval - elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def stop_monitoring(self):
        """Stop the monitoring thread and terminate the powermetrics process."""
        super().stop_monitoring() # This joins the thread
        
        # Now that the thread is joined, kill the process
        if self.powermetrics_process:
            try:
                self.powermetrics_process.terminate() # Send SIGTERM
                self.powermetrics_process.wait(timeout=1.0) # Wait for it to die
            except subprocess.TimeoutExpired:
                self.powermetrics_process.kill() # Force kill
            except Exception as e:
                print(f"Error terminating powermetrics: {e}")
            self.powermetrics_process = None

    def get_metrics(self):
        """
Note: The get_metrics logic from the file in the prompt is fine,
I will just copy it.
        """
        if not self.samples:
            return {"device_name": self.device_name, "error": "No samples collected"}
        
        def valid_values(key):
            return [s[key] for s in self.samples if s.get(key) is not None]

        metrics = {
            "device_name": self.device_name,
            "raw_samples": self.samples,
            "num_samples": len(self.samples),
        }
        
        cpu_vals = valid_values("cpu_utilization_percent")
        if cpu_vals:
            metrics["peak_cpu_utilization_percent"] = max(cpu_vals)
            metrics["average_cpu_utilization_percent"] = sum(cpu_vals) / len(cpu_vals)
        
        mem_vals = valid_values("memory_used_mb")
        if mem_vals:
            metrics["peak_memory_mb"] = max(mem_vals)
            metrics["average_memory_mb"] = sum(mem_vals) / len(mem_vals)
            
        mem_perc_vals = valid_values("memory_utilization_percent")
        if mem_perc_vals:
            metrics["peak_memory_utilization_percent"] = max(mem_perc_vals)
            metrics["average_memory_utilization_percent"] = sum(mem_perc_vals) / len(mem_perc_vals)

        gpu_power_vals = valid_values("gpu_power_watts")
        if gpu_power_vals:
            metrics["peak_gpu_power_watts"] = max(gpu_power_vals)
            metrics["average_gpu_power_watts"] = sum(gpu_power_vals) / len(gpu_power_vals)

        gpu_util_vals = valid_values("gpu_utilization_percent")
        if gpu_util_vals:
            metrics["peak_gpu_utilization_percent"] = max(gpu_util_vals)
            metrics["average_gpu_utilization_percent"] = sum(gpu_util_vals) / len(gpu_util_vals)

        e_vals = valid_values("e_cluster_power_watts")
        if e_vals:
            metrics["average_e_cluster_power_watts"] = sum(e_vals) / len(e_vals)
        
        p_vals = valid_values("p_cluster_power_watts")
        if p_vals:
            metrics["average_p_cluster_power_watts"] = sum(p_vals) / len(p_vals)
            
        temp_vals = valid_values("cpu_temp_c")
        if temp_vals:
            metrics["peak_cpu_temp_c"] = max(temp_vals)
            metrics["average_cpu_temp_c"] = sum(temp_vals) / len(temp_vals)
            
        return metrics