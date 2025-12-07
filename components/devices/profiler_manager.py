# components/devices/profiler_manager.py
import torch
import platform
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from .base import BaseDeviceProfiler
from .cpu_profiler import LocalCpuProfiler
from .nvidia_gpu_profiler import NvidiaGpuProfiler
from .mac_profiler import MacProfiler
from .jetson_profiler import JetsonProfiler
from .pi_profiler import PiProfiler
from .profiler_utils import get_results_directory

try:
    import pynvml
except ImportError:
    pynvml = None

log = logging.getLogger(__name__)

def get_system_info() -> Dict[str, str]:
    """
    Returns: Dict with keys 'system' and 'processor_info'
    """
    return {
        'system': platform.system(),
        'processor_info': platform.processor().lower()
    }

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

def is_soc_device():
    if is_jetson():
        return True
    if is_raspberry_pi():
        return True
    return False

def get_platform_profiler_classes(device_override: str = None) -> List:
    """
    Detects the current hardware platform and returns the appropriate profiler classes.
    Returns:
        List of profiler classes to instantiate.
    """
    
    # Handle explicit device overrides
    if device_override:
        device_override = device_override.lower()
        
        if device_override == "jetson":
            log.info("Device override: using Jetson profiler")
            return [JetsonProfiler]
        elif device_override == "pi":
            log.info("Device override: using Raspberry Pi profiler")
            return [PiProfiler]
        elif device_override == "mac":
            log.info("Device override: using Mac profiler")
            return [MacProfiler]
        elif device_override == "cuda":
            log.info("Device override: using CUDA profiler (and CPU)")
            profilers = [LocalCpuProfiler]
            if torch.cuda.is_available():
                profilers.append(NvidiaGpuProfiler)
            return profilers
        elif device_override == "cuda-only":
            log.info("Device override: using CUDA profiler ONLY")
            if torch.cuda.is_available():
                return [NvidiaGpuProfiler]
            else:
                log.warning("CUDA requested but not available. Falling back to CPU.")
                return [LocalCpuProfiler]
        elif device_override == "cpu":
            log.info("Device override: using CPU profiler only")
            return [LocalCpuProfiler]
        elif device_override != "auto":
            log.warning("Unknown device override '%s', falling back to auto-detection", device_override)
    
    system = platform.system()

    # Auto-detection based on platform
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
    
    def __init__(self, config: Dict, run_name: Optional[str] = None):
        """
        Args:
            config: Configuration dictionary or DictConfig.
            run_name: Optional name for this benchmark run. Used for results directory.
        """
        self.config = config
        self.profilers: List[BaseDeviceProfiler] = []
        self.all_metrics: Dict[str, Dict] = {}
        self._pynvml_initialized = False
        self._system_info = get_system_info()
        
        # Create results directory for this run
        self.results_dir = get_results_directory(run_name)
        log.info("Results will be saved to: %s", self.results_dir)
        
        log.debug("System: %s (%s)", self._system_info['system'], self._system_info['processor_info'])
        self._initialize_profilers()
        log.info("Initialized %d profiler(s)", len(self.profilers))
    
    def _initialize_profilers(self):
        """
        Detect available devices and initialize appropriate profilers.
        Uses device.type from config, or falls back to device_override for testing.
        """
        device_type = self.config.get("device", {}).get("type", None)
        log.debug("Device type: %s", device_type)
        profiler_classes = get_platform_profiler_classes(device_override=device_type)
        log.debug("Detected profiler classes: %s", [cls.__name__ for cls in profiler_classes])

        # Special handling for NVIDIA GPUs: Initialize pynvml ONCE
        if NvidiaGpuProfiler in profiler_classes:
            if pynvml is None:
                log.warning("NVIDIA GPU profiling requested but pynvml not installed, skipping GPU profilers")
                profiler_classes.remove(NvidiaGpuProfiler)
            else:
                try:
                    pynvml.nvmlInit()
                    self._pynvml_initialized = True
                    log.debug("pynvml initialized")
                except pynvml.NVMLError as e:
                    log.warning("Failed to initialize pynvml: %s. GPU profiling disabled", e)
                    profiler_classes.remove(NvidiaGpuProfiler)

        for profiler_class in profiler_classes:
            try:
                # Special case: Instantiate one profiler per GPU
                if profiler_class == NvidiaGpuProfiler and self._pynvml_initialized:
                    device_count = pynvml.nvmlDeviceGetCount()
                    log.info("Detected %d NVIDIA GPU(s)", device_count)
                    for i in range(device_count):
                        try:
                            profiler_instance = profiler_class(
                                self.config, 
                                device_index=i, 
                                profiler_manager=self,
                                results_dir=self.results_dir
                            )
                            self.profilers.append(profiler_instance)
                            log.debug("GPU %d profiler initialized", i)
                        except Exception as e:
                            log.error("Failed to initialize GPU %d profiler: %s", i, e, exc_info=True)
                
                elif profiler_class != NvidiaGpuProfiler:
                    # Standard instantiation for all other profilers
                    profiler_instance = profiler_class(
                        self.config, 
                        profiler_manager=self,
                        results_dir=self.results_dir
                    )
                    self.profilers.append(profiler_instance)
                    log.debug("%s profiler initialized", profiler_class.__name__)
                    
            except Exception as e:
                log.error("Failed to initialize %s profiler: %s", profiler_class.__name__, e, exc_info=True)
    
    def get_system_info(self) -> Dict[str, str]:
        """Return cached system info."""
        return self._system_info

    def start_all(self):
        """Start all profilers simultaneously."""
        log.debug("Starting %d profiler(s)", len(self.profilers))
        for profiler in self.profilers:
            profiler.start_monitoring()
    
    def stop_all(self):
        """
        Stop all profilers and *immediately* collect their metrics.
        This is called by the __exit__ of the 'with' block.
        """
        log.info("Stopping profilers and collecting metrics")
        for profiler in self.profilers:
            metrics = profiler.stop_monitoring()
            profiler_name = profiler.__class__.__name__.lower()
            
            # Create unique key for multi-GPU
            if isinstance(profiler, NvidiaGpuProfiler):
                profiler_name = "%s_gpu%d" % (profiler_name, profiler.device_index)
            
            if profiler_name in self.all_metrics:
                log.warning("Duplicate profiler name '%s', overwriting metrics", profiler_name)
            
            self.all_metrics[profiler_name] = metrics
        
        # Shut down pynvml after all profilers are stopped
        if self._pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as e:
                log.warning("Failed to shut down pynvml: %s", e)
    
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