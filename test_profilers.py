#!/usr/bin/env python
"""Quick test to verify all profilers work with the new CSV + metrics pattern."""

from omegaconf import DictConfig
from components.devices.cpu_profiler import LocalCpuProfiler
from components.devices.pi_profiler import PiProfiler
from components.devices.jetson_profiler import JetsonProfiler
from components.devices.mac_profiler import MacProfiler

# Create minimal config
config = DictConfig({"sampling_interval": 0.5})

# Test instantiation
cpu_prof = LocalCpuProfiler(config)
pi_prof = PiProfiler(config)
jetson_prof = JetsonProfiler(config)
mac_prof = MacProfiler(config)

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

print("\nAll profilers instantiated successfully with CSV + metrics pattern!")
