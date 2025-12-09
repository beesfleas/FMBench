classDiagram
    %% Relationships: File to File (Dependency/Inheritance)
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
    cpu_profiler ..> profiler_utils : uses
    nvidia_gpu_profiler ..> profiler_utils : uses
    mac_profiler ..> profiler_utils : uses
    pi_profiler ..> profiler_utils : uses
    jetson_profiler ..> profiler_utils : uses
    profiler_manager ..> profiler_utils : uses

    %% Files as Classes

    class profiler_manager {
        DEFAULT_RESULTS_DIR: str
        config: Dict
        profilers: List
        results_dir: Path
        _system_info: Dict
        _pynvml_initialized: bool
        all_metrics: Dict
        get_system_info() : Dict
        is_jetson() : bool
        is_raspberry_pi() : bool
        is_soc_device() : bool
        get_platform_profiler_classes(device_override) : List
        __init__(config, run_name)
        _initialize_profilers()
        get_system_info()
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
        results_dir: Optional[Path]
        csv_filepath: Optional[str]
        _monitoring_thread: Thread
        _stop_event: Event
        __init__(config, results_dir)
        get_device_info()*
        start_monitoring()
        stop_monitoring()
        get_metrics()
        __enter__()
        __exit__()
        _monitor_process()*
        _start_monitoring_thread()
    }

    class cpu_profiler {
        profiler_manager: ProfilerManager
        sampling_interval: float
        device_name: str
        cpu_type: str
        energy_counter_path: Optional[str]
        power_monitoring_available: bool
        temp_monitoring_available: bool
        metrics: Dict
        __init__(config, profiler_manager, results_dir)
        get_device_info()
        _detect_cpu_type()
        _find_intel_rapl_path()
        _find_amd_energy_path()
        _check_metric_availability()
        _check_power_availability()
        _read_energy_uj()
        _monitor_process()
    }

    class mac_profiler {
        METRIC_PATTERNS: List
        sampling_interval: float
        sampling_interval_ms: int
        device_name: str
        metrics: Dict
        powermetrics_process: Popen
        _can_read_powermetrics: bool
        last_known_metrics: Dict
        total_energy_joules: float
        start_time: float
        last_sample_time: float
        last_psutil_time: float
        last_cpu_util: float
        csv_writer: CSVWriter
        accumulators: Dict
        __init__(config, profiler_manager, results_dir)
        get_device_info()
        _start_powermetrics_process()
        _parse_powermetrics_block(block)
        _update_stats()
        _record_sample(power_metrics)
        _monitor_process()
    }

    class nvidia_gpu_profiler {
        profiler_manager: ProfilerManager
        device_index: int
        handle: nvmlDevice
        device_name: str
        sampling_interval: float
        metrics: Dict
        power_available: bool
        temp_available: bool
        memory_available: bool
        util_available: bool
        __init__(config, device_index, profiler_manager, results_dir)
        get_device_info()
        _check_metric_availability()
        _monitor_process()
        stop_monitoring()
    }

    class pi_profiler {
        profiler_manager: ProfilerManager
        sampling_interval: float
        device_name: str
        metrics: Dict
        thermal_zone_path: str
        temp_available: bool
        power_monitoring_available: bool
        __init__(config, profiler_manager, results_dir)
        get_device_info()
        _check_power_availability()
        _read_temp()
        _read_power_watts()
        _monitor_process()
    }

    class jetson_profiler {
        profiler_manager: ProfilerManager
        sampling_interval: float
        device_name: str
        jtop_wrapper: Any
        has_jtop: bool
        metrics: Dict
        total_energy_joules: float
        __init__(config, profiler_manager, results_dir)
        get_device_info()
        _monitor_process()
        _collection_loop(start_time, jetson, accumulators)
        _update_metrics(accumulators, duration)
    }

    class profiler_utils {
        DEFAULT_RESULTS_DIR: str
        CSV_FLUSH_INTERVAL: int
        track_nonzero: bool
        count: int
        sum: float
        max: float
        min: float
        nonzero_count: int
        nonzero_sum: float
        nonzero_max: float
        nonzero_min: float
        filepath: str
        flush_interval: int
        _file: File
        _writer: DictWriter
        _fieldnames: List
        _buffer_count: int
        _initialized: bool
        get_project_root() : Path
        get_results_directory(run_name) : Path
        generate_csv_filepath(results_dir, profiler_name, suffix) : str
        read_samples_from_csv(filepath) : List
        calculate_metrics_from_samples(samples, metric_keys) : Dict
        calculate_energy_from_samples(samples, energy_key, timestamp_key) : Dict
        __init__(track_nonzero)
        reset()
        add(value)
        get_stats(use_nonzero)
        __init__(filepath, flush_interval)
        _initialize(sample)
        write_sample(sample)
        flush()
        close()
        __enter__()
        __exit__()
    }

