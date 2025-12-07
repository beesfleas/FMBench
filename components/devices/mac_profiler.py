import subprocess
import time
import psutil
import os
import re
from pathlib import Path
from typing import Optional
from .base import BaseDeviceProfiler
from .profiler_utils import CSVWriter, MetricAccumulator, generate_csv_filepath, get_results_directory
import logging

log = logging.getLogger(__name__)


class MacProfiler(BaseDeviceProfiler):
    """
    Profiler for macOS (Intel and Apple Silicon) using powermetrics.
    """
    
    # Regex patterns for parsing powermetrics output: (Pattern, Metric Key, Divisor)
    METRIC_PATTERNS = [
        (r'CPU Power:\s*(\d+)\s*mW', 'cpu_power_watts', 1000.0),
        (r'GPU Power:\s*(\d+)\s*mW', 'gpu_power_watts', 1000.0),
        (r'ANE Power:\s*(\d+)\s*mW', 'ane_power_watts', 1000.0),
        (r'Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW', 'combined_power_watts', 1000.0),
        (r'GPU HW active frequency:\s*(\d+)\s*MHz', 'gpu_active_frequency_mhz', 1.0),
        (r'GPU HW active residency:\s*([\d\.]+)%', 'gpu_utilization_percent', 1.0),
    ]

    def __init__(self, config, profiler_manager=None, results_dir: Optional[Path] = None):
        super().__init__(config, results_dir)
        self.sampling_interval = config.get("sampling_interval", 1.0)
        self.sampling_interval_ms = int(self.sampling_interval * 1000)
        self.device_name = "macOS"
        
        # State variables
        self.metrics = {"device_name": self.device_name, "num_samples": 0, "csv_filepath": None}
        self.powermetrics_process = None
        self._can_read_powermetrics = True
        self.last_known_metrics = {}
        self.total_energy_joules = 0.0
        self.start_time = 0
        self.last_sample_time = 0
        self.last_psutil_time = 0
        self.last_cpu_util = 0.0
        self.csv_writer = None
        
        # Initialize accumulators
        self.accumulators = {
            "cpu_util": MetricAccumulator(track_nonzero=True),
            "mem": MetricAccumulator(),
            "mem_pct": MetricAccumulator(),
            "cpu_power": MetricAccumulator(track_nonzero=True),
            "gpu_power": MetricAccumulator(track_nonzero=True),
            "ane_power": MetricAccumulator(track_nonzero=True),
            "combined_power": MetricAccumulator(track_nonzero=True),
            "gpu_freq": MetricAccumulator(),
            "gpu_util": MetricAccumulator(track_nonzero=True),
        }

        # Prime psutil to avoid initial 0.0
        psutil.cpu_percent(interval=None)
        log.debug("Initialized macOS Profiler")

    def get_device_info(self) -> str:
        try:
            res = subprocess.run(["system_profiler", "SPHardwareDataType"], 
                               capture_output=True, text=True, timeout=5)
            info = [l.strip() for l in res.stdout.split('\n') 
                   if any(x in l for x in ['Model', 'Processor', 'Memory'])]
            return f"{self.device_name} {' | '.join(info)}"
        except Exception:
            return self.device_name

    def _start_powermetrics_process(self):
        if not self._can_read_powermetrics:
            return
        cmd = ['powermetrics' if os.geteuid() == 0 else 'sudo', 'powermetrics']
        cmd.extend(['--samplers', 'cpu_power,gpu_power', '-i', str(self.sampling_interval_ms)])
        
        try:
            self.powermetrics_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
            time.sleep(0.1)  # Quick check for immediate failure
            if self.powermetrics_process.poll() is not None:
                err = self.powermetrics_process.stderr.read()
                log.warning("powermetrics failed: %s", err)
                self._can_read_powermetrics = False
                self.powermetrics_process = None
        except Exception as e:
            log.warning("Failed to start powermetrics: %s", e)
            self._can_read_powermetrics = False

    def _parse_powermetrics_block(self, block: str) -> dict:
        metrics = {}
        for pattern, key, divisor in self.METRIC_PATTERNS:
            match = re.search(pattern, block)
            if match:
                metrics[key] = float(match.group(1)) / divisor
        
        if not metrics and len(block) > 100 and "Machine model" not in block:
            log.debug("Parsed large block but found no metrics: %s...", block[:100])
        return metrics

    def _update_stats(self):
        """Update aggregate statistics in self.metrics based on accumulators."""
        mappings = {
            "cpu_util": "cpu_utilization_percent",
            "mem": "memory_mb",
            "mem_pct": "memory_utilization_percent",
            "cpu_power": "cpu_power_watts",
            "gpu_power": "gpu_power_watts",
            "ane_power": "ane_power_watts",
            "combined_power": "combined_power_watts",
            "gpu_freq": "gpu_active_frequency_mhz",
            "gpu_util": "gpu_utilization_percent"
        }
        
        for acc_key, metric_name in mappings.items():
            acc = self.accumulators[acc_key]
            use_nonzero = acc.track_nonzero
            stats = acc.get_stats(use_nonzero=use_nonzero)
            self.metrics[f"average_{metric_name}"] = stats["average"]
            self.metrics[f"peak_{metric_name}"] = stats["peak"]
            self.metrics[f"min_{metric_name}"] = stats["min"]
        
        self.metrics["total_energy_joules"] = self.total_energy_joules

    def _record_sample(self, power_metrics=None):
        """Process a single sample point: gather psutil, merge power, write CSV, update stats."""
        if power_metrics:
            self.last_known_metrics.update(power_metrics)
        
        # Throttle psutil (min 50ms)
        now = time.perf_counter()
        if now - self.last_psutil_time > 0.05:
            self.last_cpu_util = psutil.cpu_percent(interval=None)
            self.last_psutil_time = now
            
        try:
            vmem = psutil.virtual_memory()
            sample = {
                "timestamp": now - self.start_time,
                "cpu_utilization_percent": self.last_cpu_util,
                "memory_used_mb": vmem.used / 1048576,
                "memory_utilization_percent": vmem.percent
            }
        except Exception:
            sample = {"timestamp": now - self.start_time}

        sample.update(self.last_known_metrics)

        # Energy Integration
        total_power = sample.get("combined_power_watts", 0)
        if total_power <= 0:
            total_power = sum([
                sample.get("cpu_power_watts", 0),
                sample.get("gpu_power_watts", 0),
                sample.get("ane_power_watts", 0)
            ])
        
        # Use actual time delta between samples for accurate energy integration
        if self.last_sample_time > 0:
            time_delta = now - self.last_sample_time
        else:
            time_delta = self.sampling_interval

        if total_power > 0 and time_delta > 0:
            self.total_energy_joules += total_power * time_delta
        
        self.last_sample_time = now

        # Write sample to CSV
        if self.csv_writer:
            self.csv_writer.write_sample(sample)

        # Update Accumulators
        acc_map = {
            "cpu_util": "cpu_utilization_percent",
            "mem": "memory_used_mb",
            "mem_pct": "memory_utilization_percent",
            "cpu_power": "cpu_power_watts",
            "gpu_power": "gpu_power_watts",
            "ane_power": "ane_power_watts",
            "combined_power": "combined_power_watts",
            "gpu_freq": "gpu_active_frequency_mhz",
            "gpu_util": "gpu_utilization_percent"
        }
        for acc_key, sample_k in acc_map.items():
            if sample_k in sample:
                self.accumulators[acc_key].add(sample[sample_k])

        self.metrics["num_samples"] = self.accumulators["cpu_util"].count
        self.metrics["monitoring_duration_seconds"] = sample["timestamp"]
        self._update_stats()

    def _monitor_process(self):
        # Setup CSV file in results directory
        if self.results_dir:
            self.csv_filepath = generate_csv_filepath(self.results_dir, "mac_profiler")
        else:
            results_dir = get_results_directory()
            self.csv_filepath = generate_csv_filepath(results_dir, "mac_profiler")
        
        self.metrics["csv_filepath"] = self.csv_filepath
        log.info("Writing samples to: %s", self.csv_filepath)
        
        # Define all possible fields for consistent CSV structure
        all_fields = [
            'timestamp', 'cpu_utilization_percent', 'memory_used_mb', 'memory_utilization_percent',
            'cpu_power_watts', 'gpu_power_watts', 'ane_power_watts', 'combined_power_watts',
            'gpu_utilization_percent', 'gpu_active_frequency_mhz'
        ]
        
        self._start_powermetrics_process()
        self.start_time = time.perf_counter()
        self.last_sample_time = self.start_time
        current_block = ""

        try:
            with CSVWriter(self.csv_filepath) as csv_writer:
                self.csv_writer = csv_writer
                
                while not self._stop_event.is_set():
                    if self.powermetrics_process and self.powermetrics_process.stdout:
                        line = self.powermetrics_process.stdout.readline()
                        if not line:
                            break
                        
                        if '***' in line:
                            if current_block.strip():
                                metrics = self._parse_powermetrics_block(current_block)
                                if metrics:
                                    self._record_sample(metrics)
                            current_block = line
                        else:
                            current_block += line
                    else:
                        time.sleep(0.1)
                        
        except Exception as e:
            log.error("Monitor loop error: %s", e)
        finally:
            self.csv_writer = None
            
            # Graceful Shutdown & Flush
            if self.accumulators["cpu_power"].count == 0 and self.powermetrics_process:
                if self.powermetrics_process.poll() is None:
                    log.debug("Waiting for powermetrics flush...")
                    wait_time = max(self.sampling_interval * 2, 2.0)
                    time.sleep(wait_time)

            if self.powermetrics_process:
                try:
                    self.powermetrics_process.terminate()
                    try:
                        self.powermetrics_process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self.powermetrics_process.kill()
                    
                    if self.powermetrics_process.stdout:
                        current_block += self.powermetrics_process.stdout.read()
                except Exception:
                    pass
                self.powermetrics_process = None

            if current_block.strip():
                metrics = self._parse_powermetrics_block(current_block)
                if metrics:
                    self._record_sample(metrics)
