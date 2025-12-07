#!/usr/bin/env python
"""Quick test to verify all profilers work with the new CSV + metrics pattern."""

from omegaconf import DictConfig
from components.devices.cpu_profiler import LocalCpuProfiler
from components.devices.pi_profiler import PiProfiler
from components.devices.jetson_profiler import JetsonProfiler
from components.devices.mac_profiler import MacProfiler
from components.devices.profiler_utils import get_results_directory

# Create minimal config
config = DictConfig({"sampling_interval": 0.5})

# Get a test results directory
results_dir = get_results_directory("test_profilers")
print(f"Test results directory: {results_dir}")

# Test instantiation (without profiler_manager for simple test)
cpu_prof = LocalCpuProfiler(config, results_dir=results_dir)
pi_prof = PiProfiler(config, results_dir=results_dir)
jetson_prof = JetsonProfiler(config, results_dir=results_dir)
mac_prof = MacProfiler(config, results_dir=results_dir)

# Verify metrics dict structure
print("CPU Profiler metrics structure:", list(cpu_prof.metrics.keys()))
print("Pi Profiler metrics structure:", list(pi_prof.metrics.keys()))
print("Jetson Profiler metrics structure:", list(jetson_prof.metrics.keys()))
print("Mac Profiler metrics structure:", list(mac_prof.metrics.keys()))

# Verify expected fields exist
expected_keys = ["device_name", "num_samples", "csv_filepath"]
for profiler, name in [
    (cpu_prof, "CPU"),
    (pi_prof, "Pi"),
    (jetson_prof, "Jetson"),
    (mac_prof, "Mac"),
]:
    for key in expected_keys:
        if key not in profiler.metrics:
            print(f"ERROR: {name} profiler missing key: {key}")
        else:
            print(f"✓ {name} profiler has {key}")

print(f"\nAll profilers instantiated successfully!")
print(f"CSV files will be saved to: {results_dir}")
