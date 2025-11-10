import torch
import platform  # OS/hardware detection
import os # Jetson/Pi detection
from typing import Dict, List
from .base import BaseDeviceProfiler
from .cpu_profiler import LocalCpuProfiler
from .nvidia_gpu_profiler import NvidiaGpuProfiler
from .mac_profiler import MacProfiler
from .jetson_profiler import JetsonProfiler
from .pi_profiler import PiProfiler

# --- Helper functions for platform detection ---

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

# --- End of Helper Functions ---


class ProfilerManager:
    """
    Coordinates multiple profilers based on device configuration.
    Manages lifecycle and ensures synchronized metrics collection.
    
    Now platform-aware: selects the correct profiler(s) for the OS and hardware.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.profilers: List[BaseDeviceProfiler] = []
        self.all_metrics: Dict[str, Dict] = {}
        self._initialize_profilers()
    
    def _initialize_profilers(self):
        """
        Detect the host platform and initialize the correct profilers.
        """
        device_type = self.config.get("device", {}).get("type", "cpu").lower()
        os_type = platform.system()
        
        # --- 1. Select the CPU/System Profiler ---
        if os_type == "Darwin":
            self.profilers.append(MacProfiler(self.config))

        elif os_type == "Linux":
            if is_jetson():
                # Use the specific Jetson profiler
                self.profilers.append(JetsonProfiler(self.config))
            elif is_raspberry_pi():
                # Use the specific Pi profiler
                self.profilers.append(PiProfiler(self.config))
            else:
                # Standard Linux (e.g., Desktop, Server)
                self.profilers.append(LocalCpuProfiler(self.config))

        else: # Windows or other operating systems, certain metrics will not be collected.
            self.profilers.append(LocalCpuProfiler(self.config))

        # --- 2. Add NVIDIA GPU Profiler (if applicable) ---
        if device_type == "cuda" and not is_jetson() and torch.cuda.is_available():
            try:
                self.profilers.append(NvidiaGpuProfiler(self.config))
                print("NVIDIA GPU Profiler initialized")
            except Exception as e:
                print(f"NVIDIA GPU Profiler unavailable: {e}")
 
    def start_all(self):
        """Start all profilers simultaneously."""
        for profiler in self.profilers:
            profiler.start_monitoring()
    
    def stop_all(self):
        """
        Stop all profilers simultaneously and store metrics.
        Returns the collected metrics.
        """
        print("Stopping profilers...")
        self.all_metrics = {} 
        for profiler in self.profilers:
            metrics = profiler.stop_monitoring() 
            
            # Determine a robust key for the metrics dict
            if isinstance(profiler, LocalCpuProfiler):
                profiler_name = "cpu_profiler"
            elif isinstance(profiler, NvidiaGpuProfiler):
                profiler_name = "nvidia_gpu_profiler"
            elif isinstance(profiler, MacProfiler):
                profiler_name = "mac_profiler"
            elif isinstance(profiler, JetsonProfiler):
                profiler_name = "jetson_profiler"
            elif isinstance(profiler, PiProfiler):
                profiler_name = "pi_profiler"
            else:
                profiler_name = profiler.__class__.__name__.replace("Profiler", "").lower()
                
            self.all_metrics[profiler_name] = metrics
        
        print("Profiling complete.")
        return self.all_metrics
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """
        Returns the dictionary of all collected metrics.
        """
        return self.all_metrics

    # --- Context Manager Support ---
    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit. Stops and stores metrics."""
        self.stop_all()