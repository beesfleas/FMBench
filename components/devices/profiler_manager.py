# components/profilers/profiler_manager.py
import torch
from typing import Dict, List
from .base import BaseDeviceProfiler
from .cpu_profiler import LocalCpuProfiler
from .nvidia_gpu_profiler import NvidiaGpuProfiler


class ProfilerManager:
    """
    Coordinates multiple profilers based on device configuration.
    Manages lifecycle and ensures synchronized metrics collection.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.profilers: List[BaseDeviceProfiler] = []
        self._initialize_profilers()
    
    def _initialize_profilers(self):
        """Detect available devices and initialize appropriate profilers."""
        device_type = self.config.get("device", {}).get("type", "cpu").lower()
        
        # Always profile CPU (orchestration overhead matters)
        self.profilers.append(LocalCpuProfiler(self.config))
        print("CPU Profiler initialized")
        
        # Add GPU profiler if using CUDA
        if device_type == "cuda" and torch.cuda.is_available():
            try:
                self.profilers.append(NvidiaGpuProfiler(self.config))
                print("GPU Profiler initialized")
            except Exception as e:
                print(f"GPU Profiler unavailable: {e}")
 
    def start_all(self):
        """Start all profilers simultaneously."""
        for profiler in self.profilers:
            profiler.start_monitoring()
    
    def stop_all(self):
        """Stop all profilers simultaneously."""
        for profiler in self.profilers:
            profiler.stop_monitoring()
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """
        Collect metrics from all profilers.
        Returns dict keyed by profiler type.
        """
        metrics = {}
        for profiler in self.profilers:
            profiler_name = profiler.__class__.__name__.replace("Profiler", "").lower()
            metrics[profiler_name] = profiler.get_metrics()
        return metrics
