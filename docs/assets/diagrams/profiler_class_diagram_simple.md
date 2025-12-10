classDiagram
    %% Relationships
    base <|-- cpu_profiler
    base <|-- nvidia_gpu_profiler
    base <|-- mac_profiler
    base <|-- pi_profiler
    base <|-- jetson_profiler

    profiler_manager --> base : uses
    profiler_manager --> cpu_profiler : imports
    profiler_manager --> nvidia_gpu_profiler : imports
    profiler_manager --> mac_profiler : imports
    profiler_manager --> pi_profiler : imports
    profiler_manager --> jetson_profiler : imports

    base ..> profiler_utils : uses

    %% Simplified Classes (Key fields & methods only)

    class profiler_manager {
        config: Dict
        profilers: List
        results_dir: Path
        initialize_profilers()
        start_all()
        stop_all()
        get_all_metrics()
        __enter__()
        __exit__()
    }

    class base {
        config: Dict
        metrics: Dict
        device_name: str
        results_dir: Path
        start_monitoring()
        stop_monitoring()
        get_metrics()
        _monitor_process()*
    }

    class cpu_profiler {
        cpu_type: str
        power_monitoring_available: bool
        temp_monitoring_available: bool
        _detect_cpu_type()
        _monitor_process()
    }

    class mac_profiler {
        device_name: str
        total_energy_joules: float
        _monitor_process()
    }

    class nvidia_gpu_profiler {
        device_index: int
        device_name: str
        power_available: bool
        temp_available: bool
        util_available: bool
        _monitor_process()
        stop_monitoring()
    }

    class pi_profiler {
        device_name: str
        temp_available: bool
        power_monitoring_available: bool
        _monitor_process()
    }

    class jetson_profiler {
        device_name: str
        has_jtop: bool
        total_energy_joules: float
        _monitor_process()
    }

    class profiler_utils {
        MetricAccumulator
        CSVWriter
        get_results_directory(run_name)
        generate_csv_filepath(results_dir, profiler_name)
        calculate_metrics_from_samples(samples)
        calculate_energy_from_samples(samples)
    }