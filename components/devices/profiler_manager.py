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
    
    (Minimally updated to support context manager and metric storage)
    """
    def __init__(self, config: Dict):
        self.config = config
        self.profilers: List[BaseDeviceProfiler] = []
        self.all_metrics: Dict[str, Dict] = {} # <-- ADDED: To store results
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
        """
        Stop all profilers simultaneously and store metrics.
        Returns the collected metrics.
        """
        self.all_metrics = {} # <-- ADDED: Clear previous metrics
        for profiler in self.profilers:
            # stop_monitoring() now returns the metrics
            metrics = profiler.stop_monitoring() 
            
            # ADDED: Determine a robust key for the metrics dict
            if isinstance(profiler, LocalCpuProfiler):
                profiler_name = "cpu_profiler"
            elif isinstance(profiler, NvidiaGpuProfiler):
                profiler_name = "nvidia_gpu_profiler"
            else:
                profiler_name = profiler.__class__.__name__.replace("Profiler", "").lower()
                
            self.all_metrics[profiler_name] = metrics
        
        return self.all_metrics # <-- ADDED: Return all collected metrics
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """
        Returns the dictionary of all collected metrics.
        """
        return self.all_metrics

    # --- ADDED: Context Manager Support ---
    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit. Stops and stores metrics."""
        self.stop_all()
    # --- END OF ADDED BLOCK ---