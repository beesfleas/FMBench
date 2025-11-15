"""
Utilities for profiler CSV handling and metric calculations.
"""
import csv
import os
from typing import Dict, List, Any
import logging

log = logging.getLogger(__name__)


def write_sample_to_csv(filepath: str, sample: Dict[str, Any], is_header: bool = False):
    """
    Write a single sample to CSV file (append mode).
    If is_header=True, write as header row using sample keys.
    """
    if not sample:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Determine if file exists (to know if we should write header)
    file_exists = os.path.exists(filepath)
    
    try:
        with open(filepath, 'a', newline='') as f:
            fieldnames = list(sample.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(sample)
    except Exception as e:
        log.error(f"Error writing to CSV {filepath}: {e}")


def read_samples_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """
    Read all samples from a CSV file.
    Returns list of dictionaries with string values.
    """
    if not os.path.exists(filepath):
        return []
    
    samples = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings to float/int where applicable
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
        log.error(f"Error reading CSV {filepath}: {e}")
    
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
    
    # Calculate total energy (delta between last and first)
    first_energy = energy_values[0]
    last_energy = energy_values[-1]
    
    # Check if energy is in microJoules (Intel RAPL) and convert to Joules
    if last_energy > 1e6:  # Likely in microJoules
        total_energy_joules = (last_energy - first_energy) / 1_000_000.0
    else:  # Already in Joules
        total_energy_joules = last_energy - first_energy
    
    metrics['total_energy_joules'] = total_energy_joules
    
    # Calculate monitoring duration
    timestamps = [s.get(timestamp_key) for s in samples if s.get(timestamp_key) is not None]
    if timestamps and len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            metrics['average_power_watts'] = total_energy_joules / duration
    
    return metrics
