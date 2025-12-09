"""
Utilities for profiler CSV handling and metric calculations.
"""
import csv
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# Default results directory name (relative to project root)
DEFAULT_RESULTS_DIR = "results"
# Number of samples to buffer before flushing to disk
CSV_FLUSH_INTERVAL = 10


def get_project_root() -> Path:
    """
    Get the project root directory.
    Walks up from this file to find the project root (where run.py exists).
    """
    current = Path(__file__).resolve()
    # Walk up to find the project root (contains run.py or conf/)
    for parent in [current] + list(current.parents):
        if (parent / "run.py").exists() or (parent / "conf").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def get_results_directory(run_name: Optional[str] = None) -> Path:
    """
    Get or create the results directory for the current benchmark run.
    
    Args:
        run_name: Optional name for this run. If None, uses timestamp.
    
    Returns:
        Path to the results directory for this run.
    """
    project_root = get_project_root()
    results_base = project_root / DEFAULT_RESULTS_DIR
    
    if run_name is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    
    results_dir = results_base / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log.debug("Results directory: %s", results_dir)
    return results_dir


class MetricAccumulator:
    """
    Accumulates metric values and calculates statistics (min, max, avg).
    Supports tracking both all values and non-zero values separately.
    """
    
    def __init__(self, track_nonzero: bool = False):
        """
        Args:
            track_nonzero: If True, also track statistics for non-zero values only.
        """
        self.track_nonzero = track_nonzero
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        self.count = 0
        self.sum = 0.0
        self.max = float('-inf')
        self.min = float('inf')
        
        if self.track_nonzero:
            self.nonzero_count = 0
            self.nonzero_sum = 0.0
            self.nonzero_max = float('-inf')
            self.nonzero_min = float('inf')
    
    def add(self, value: float):
        """Add a value to the accumulator."""
        if value is None:
            return
            
        self.count += 1
        self.sum += value
        self.max = max(self.max, value)
        self.min = min(self.min, value)
        
        if self.track_nonzero and value != 0:
            self.nonzero_count += 1
            self.nonzero_sum += value
            self.nonzero_max = max(self.nonzero_max, value)
            self.nonzero_min = min(self.nonzero_min, value)
    
    @property
    def average(self) -> float:
        """Get the average of all values."""
        return self.sum / self.count if self.count > 0 else 0.0
    
    @property
    def nonzero_average(self) -> float:
        """Get the average of non-zero values (if tracking)."""
        if not self.track_nonzero:
            return self.average
        return self.nonzero_sum / self.nonzero_count if self.nonzero_count > 0 else 0.0
    
    def get_stats(self, use_nonzero: bool = False) -> Dict[str, float]:
        """
        Get statistics dictionary.
        
        Args:
            use_nonzero: If True and track_nonzero is enabled, return nonzero stats.
        
        Returns:
            Dict with 'count', 'average', 'peak', 'min' keys.
        """
        if use_nonzero and self.track_nonzero and self.nonzero_count > 0:
            return {
                "count": self.nonzero_count,
                "average": self.nonzero_average,
                "peak": self.nonzero_max,
                "min": self.nonzero_min,
            }
        elif self.count > 0:
            return {
                "count": self.count,
                "average": self.average,
                "peak": self.max,
                "min": self.min,
            }
        else:
            return {
                "count": 0,
                "average": 0.0,
                "peak": 0.0,
                "min": 0.0,
            }


class CSVWriter:
    """
    Buffered CSV writer for profiler samples.
    
    Handles file creation, header writing, buffering, and cleanup.
    """
    
    def __init__(self, filepath: str, flush_interval: int = CSV_FLUSH_INTERVAL):
        """
        Args:
            filepath: Path to the CSV file.
            flush_interval: Number of samples to buffer before flushing.
        """
        self.filepath = filepath
        self.flush_interval = flush_interval
        self._file = None
        self._writer = None
        self._fieldnames = None
        self._buffer_count = 0
        self._initialized = False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def _initialize(self, sample: Dict[str, Any]):
        """Initialize the CSV file with headers based on first sample."""
        if self._initialized:
            return
            
        try:
            self._fieldnames = list(sample.keys())
            self._file = open(self.filepath, 'w', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._initialized = True
            log.debug("CSV file initialized: %s", self.filepath)
        except Exception as e:
            log.error("Failed to create CSV file %s: %s", self.filepath, e)
            self._file = None
            self._writer = None
    
    def write_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Write a sample to the CSV file.
        
        Args:
            sample: Dictionary of metric values to write.
        
        Returns:
            True if write succeeded, False otherwise.
        """
        if not self._initialized:
            self._initialize(sample)
        
        if self._writer is None:
            return False
        
        try:
            self._writer.writerow(sample)
            self._buffer_count += 1
            
            # Flush periodically
            if self._buffer_count >= self.flush_interval:
                self._file.flush()
                self._buffer_count = 0
            
            return True
        except Exception as e:
            log.warning("Failed to write CSV sample: %s", e)
            return False
    
    def flush(self):
        """Force flush any buffered data to disk."""
        if self._file and not self._file.closed:
            try:
                self._file.flush()
                self._buffer_count = 0
            except Exception as e:
                log.warning("Failed to flush CSV file: %s", e)
    
    def close(self):
        """Close the CSV file."""
        if self._file and not self._file.closed:
            try:
                self._file.flush()
                self._file.close()
                log.debug("CSV file closed: %s", self.filepath)
            except Exception as e:
                log.warning("Failed to close CSV file: %s", e)
        self._file = None
        self._writer = None
        self._initialized = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def generate_csv_filepath(results_dir: Path, profiler_name: str, suffix: str = "") -> str:
    """
    Generate a CSV filepath for a profiler.
    
    Args:
        results_dir: The results directory path.
        profiler_name: Name of the profiler (e.g., 'cpu_profiler', 'nvidia_gpu_profiler').
        suffix: Optional suffix (e.g., '_gpu0' for multi-GPU).
    
    Returns:
        Full path to the CSV file.
    """
    filename = f"{profiler_name}{suffix}.csv"
    return str(results_dir / filename)


# Keep existing utility functions for backward compatibility

def read_samples_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Read all samples from a CSV file.
    Returns list of dictionaries with numeric values converted.
    """
    if not os.path.exists(filepath):
        return []
    
    samples = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted_row = {}
                for key, value in row.items():
                    if value is None or value == '':
                        converted_row[key] = None
                    else:
                        try:
                            converted_row[key] = float(value)
                        except ValueError:
                            converted_row[key] = value
                samples.append(converted_row)
    except Exception as e:
        log.error("Error reading CSV %s: %s", filepath, e)
    
    return samples


def calculate_metrics_from_samples(samples: List[Dict], 
                                    metric_keys: Dict[str, str]) -> Dict[str, Any]:
    """
    Calculate aggregate metrics from samples.
    
    Args:
        samples: List of sample dictionaries
        metric_keys: Dict mapping {metric_name: sample_key}
                     e.g., {'cpu_utilization': 'cpu_utilization_percent'}
    
    Returns:
        Dict with calculated metrics (peak, average, min)
    """
    metrics = {}
    
    for metric_name, sample_key in metric_keys.items():
        values = [s.get(sample_key) for s in samples if s.get(sample_key) is not None]
        
        if values:
            metrics[f'peak_{metric_name}'] = max(values)
            metrics[f'average_{metric_name}'] = sum(values) / len(values)
            metrics[f'min_{metric_name}'] = min(values)
    
    return metrics


def calculate_energy_from_samples(samples: List[Dict],
                                  energy_key: str,
                                  timestamp_key: str = 'timestamp') -> Dict[str, float]:
    """
    Calculate total energy and average power from energy samples.
    Assumes energy_key contains cumulative energy in Joules or microJoules.
    
    Returns dict with 'total_energy_joules' and 'average_power_watts'
    """
    metrics = {}
    
    energy_values = [s.get(energy_key) for s in samples if s.get(energy_key) is not None]
    
    if not energy_values or len(energy_values) < 2:
        return metrics
    
    first_energy = energy_values[0]
    last_energy = energy_values[-1]
    
    # Threshold for detecting microJoules (Intel RAPL typically reports in µJ)
    MICROJOULE_THRESHOLD = 1e6
    
    if last_energy > MICROJOULE_THRESHOLD:
        total_energy_joules = (last_energy - first_energy) / 1_000_000.0
    else:
        total_energy_joules = last_energy - first_energy
    
    metrics['total_energy_joules'] = total_energy_joules
    
    timestamps = [s.get(timestamp_key) for s in samples if s.get(timestamp_key) is not None]
    if timestamps and len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            metrics['average_power_watts'] = total_energy_joules / duration
    
    return metrics
