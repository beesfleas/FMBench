classDiagram
    %% Core Orchestration
    benchmark_suite --> run : executes (subprocess)
    run --> runner : calls
    runner --> profiler_manager : initializes
    runner --> model_factory : uses
    runner --> scenarios : instantiates

    class benchmark_suite {
        build_configs()
        run_benchmarks()
        print_summary()
    }

    class run {
        main()
        _setup_jetson_compatibility()
        _check_transformers_version()
    }

    class runner {
        run_benchmark()
        _setup_profilers()
        _setup_benchmark()
        _run_execution()
    }

    %% Components - Devices
    profiler_manager --> base_profiler : manages
    base_profiler <|-- cpu_profiler
    base_profiler <|-- gpu_profiler
    base_profiler <|-- jetson_profiler
    
    class profiler_manager {
        results_dir: Path
        profilers: List
        start_all()
        stop_all()
        get_all_metrics()
    }

    class base_profiler {
        start_monitoring()
        stop_monitoring()
        get_metrics()
        _monitor_process()*
    }

    %% Components - Models
    model_factory --> base_model : creates
    base_model <|-- huggingface_llm
    base_model <|-- huggingface_vlm
    base_model <|-- huggingface_timeseries

    class model_factory {
        get_model_loader(config)
    }

    class base_model {
        load_model()
        predict(prompt, ...)
        unload_model()
    }

    %% Components - Scenarios
    scenarios <|-- common_nlp
    scenarios <|-- common_vlm
    scenarios <|-- common_timeseries

    class scenarios {
        name: str
        tasks: List
        load_tasks()
        evaluate(task, output)
    }

    %% Utilities
    benchmark_suite ..> suite_utils : uses
    runner ..> suite_utils : uses

    class suite_utils {
        get_model_category()
        is_model_allowed_for_device()
        format_time()
    }
