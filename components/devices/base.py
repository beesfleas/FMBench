from abc import ABC, abstractmethod
import threading

class BaseDeviceProfiler(ABC):
    """
    Abstract base class for all device profilers.
    Profilers are responsible for monitoring hardware metrics
    during a benchmark run.
    """
    def __init__(self, config):
        self.config = config
        self.metrics = {}
        self._monitoring_thread = None
        self._is_monitoring = False
        self.device_name = "[Unknown Device]"
        print(f"Initialized Profiler: {self.__class__.__name__}")

    @abstractmethod
    def get_device_info(self) -> str:
        """
        Return a string describing the hardware being profiled.
        This should be available after __init__.
        """
        raise NotImplementedError

    @abstractmethod
    def _monitor_process(self):
        """
        The core monitoring loop. This should run in a separate thread.
        It should sample metrics and update self.metrics.
        """
        raise NotImplementedError

    def _start_monitoring_thread(self):
        """Starts the monitoring thread."""
        thread = threading.Thread(target=self._monitor_process, daemon=True)
        thread.start()
        return thread

    def start_monitoring(self):
        """Starts the monitoring logic."""
        if self._monitoring_thread is None:
            self._is_monitoring = True
            self._monitoring_thread = self._start_monitoring_thread()
            print(f"{self.__class__.__name__} monitoring started...")
        else:
            print("Monitoring is already active.")

    def stop_monitoring(self):
        """Stops the monitoring thread and collects final metrics."""
        if self._monitoring_thread:
            self._is_monitoring = False
            self._monitoring_thread.join()
            self._monitoring_thread = None
            print(f"{self.__class__.__name__} monitoring stopped.")
        else:
            print("No monitoring to stop.")
        
        return self.get_metrics()

    @abstractmethod
    def get_metrics(self):
        """
        Returns the collected metrics in a structured dictionary.
        This is called after monitoring stops.
        """
        raise NotImplementedError

    def __enter__(self):
        """Start monitoring when entering a 'with' block."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop monitoring when exiting a 'with' block."""
        self.stop_monitoring()