# components/devices/profiler_manager.py
import torch
import platform
import os
import logging
from typing import Dict, List
from .base import BaseDeviceProfiler
from .cpu_profiler import LocalCpuProfiler
from .nvidia_gpu_profiler import NvidiaGpuProfiler
from .mac_profiler import MacProfiler
from .jetson_profiler import JetsonProfiler
from .pi_profiler import PiProfiler

log = logging.getLogger(__name__)

def is_jetson():
    """Check if we are running on an NVIDIA Jetson device."""
    return os.path.exists('/etc/nv_tegra_release')

def is_raspberry_pi():
    """Check if we are running on a Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Model') and 'Raspberry Pi' in line:
                    return True
    except Exception:
        pass
    return False

def get_platform_profiler_classes() -> List:
    """
    Detects the current hardware platform and returns the
    appropriate profiler classes to use.
    """
    system = platform.system()
    
    if system == "Darwin":
        # macOS (Apple Silicon or Intel)
        return [MacProfiler]
        
    if system == "Linux":
        if is_jetson():
            # NVIDIA Jetson
            return [JetsonProfiler]
        if is_raspberry_pi():
            # Raspberry Pi
            return [PiProfiler]
        
        # Standard Linux PC (Intel/AMD)
        # It will have a CPU and *maybe* an NVIDIA GPU
        profilers = [LocalCpuProfiler]
        if torch.cuda.is_available():
            profilers.append(NvidiaGpuProfiler)
        return profilers
        
    if system == "Windows":
        # Standard Windows PC (Intel/AMD)
        profilers = [LocalCpuProfiler]
        if torch.cuda.is_available():
            profilers.append(NvidiaGpuProfiler)
        return profilers

    # Fallback for unknown systems
    return [LocalCpuProfiler]


class ProfilerManager:
    """
    Coordinates multiple profilers based on device configuration.
    Manages lifecycle and ensures synchronized metrics collection.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.profilers: List[BaseDeviceProfiler] = []
        self.all_metrics: Dict[str, Dict] = {}
        self._initialize_profilers()
    
    def _initialize_profilers(self):
        """Detect available devices and initialize appropriate profilers."""
        
        profiler_classes = get_platform_profiler_classes()
        
        # Override with config if "cpu_only" is forced
        if self.config.get("device", {}).get("type", "auto") == "cpu":
            profiler_classes = [LocalCpuProfiler]

        for profiler_class in profiler_classes:
            try:
                profiler_instance = profiler_class(self.config)
                self.profilers.append(profiler_instance)
            except Exception as e:
                log.error(f"Failed to initialize profiler {profiler_class.__name__}: {e}", exc_info=True)
 
    def get_hardware_info(self):
        """
        Gets the hardware info from all initialized profilers. Returns a string for printing
        """
        str = "\n--- Hardware Info ---\n"
        if not self.profilers:
            str += "  No profilers initialized.\n"
            return str
            
        for profiler in self.profilers:
            str += f"  - {profiler.get_device_info()}\n"
        str += "---------------------\n"

        return str

    def start_all(self):
        """Start all profilers simultaneously."""
        for profiler in self.profilers:
            profiler.start_monitoring()
    
    def stop_all(self):
        """
        Stop all profilers and *immediately* collect their metrics.
        This is called by the __exit__ of the 'with' block.
        """
        log.info("Stopping profilers and collecting metrics...")
        for profiler in self.profilers:
            metrics = profiler.stop_monitoring()
            profiler_name = profiler.__class__.__name__.lower()
            self.all_metrics[profiler_name] = metrics
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """
        Returns the collected metrics from all profilers.
        """
        if not self.all_metrics:
            log.warning("get_all_metrics() called before profilers were stopped. No data to return.")
        return self.all_metrics

    def __enter__(self):
        """Start profilers when entering a 'with' block."""
        self.start_all()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Stop profilers when exiting a 'with' block."""
        self.stop_all()