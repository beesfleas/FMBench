from abc import ABC, abstractmethod
import threading
import logging

log = logging.getLogger(__name__)
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
        self._stop_event = threading.Event()
        self.device_name = "[Unknown Device]"

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
        thread = threading.Thread(target=self._monitor_process)
        thread.start()
        return thread

    def start_monitoring(self):
        """Starts the monitoring logic."""
        if self._monitoring_thread is None:
            self._stop_event.clear()
            self._monitoring_thread = self._start_monitoring_thread()
            log.debug("%s monitoring started", self.__class__.__name__)
        else:
            log.warning("%s monitoring already active", self.__class__.__name__)

    def stop_monitoring(self):
        """Stops the monitoring thread and collects final metrics."""
        if self._monitoring_thread:
            self._stop_event.set()
            try:
                self._monitoring_thread.join(timeout=5.0)
                if self._monitoring_thread.is_alive():
                    log.warning("%s monitoring thread did not stop within timeout", self.__class__.__name__)
            except Exception as e:
                log.error(f"Error joining monitoring thread: {e}")
            finally:
                self._monitoring_thread = None
            log.debug("%s monitoring stopped", self.__class__.__name__)
        else:
            log.warning("No active monitoring to stop")
        
        return self.get_metrics()

    def get_metrics(self):
        """
        Returns the collected metrics in a structured dictionary.
        This is called after monitoring stops.
        """
        return self.metrics

    def __enter__(self):
        """Start monitoring when entering a 'with' block."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop monitoring when exiting a 'with' block."""
        self.stop_monitoring()
