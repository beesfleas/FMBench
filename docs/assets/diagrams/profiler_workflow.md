sequenceDiagram
    autonumber
    
    box "Application Layer" #f5f5f5
        participant Runner as runner.py
    end

    box "Profiler Management" #e1f5fe
        participant Manager as ProfilerManager
        participant Factory as Hardware<br/>Detection
    end

    box "Device Implementation" #e8f5e9
        participant Base as BaseDeviceProfiler
        participant Thread as Monitoring<br/>Thread
    end

    box "Data & Utils" #fff3e0
        participant Utils as ProfilerUtils<br/>& CSV
    end

    %% 1. Initialization Phase
    Runner->>Manager: __init__(config)
    activate Manager
    Manager->>Factory: get_platform_profiler_classes()
    Factory-->>Manager: [LocalCpuProfiler,<br/>NvidiaGpuProfiler, etc.]
    
    loop For each detected class
        Manager->>Base: Instantiate<br/>(e.g. LocalCpuProfiler)
        Base->>Base: _check_metric_availability()
    end
    deactivate Manager

    %% 2. Context Entry & Start
    Runner->>Manager: __enter__()
    activate Manager
    Manager->>Manager: start_all()
    
    loop For each profiler
        Manager->>Base: start_monitoring()
        activate Base
        Base->>Thread: threading.Thread(target=_monitor_process).start()
        activate Thread
        Thread-->>Base: Thread Started
        deactivate Base
    end
    deactivate Manager

    %% 3. The Monitoring Loop (Happens in background)
    Note right of Thread: Concurrently with<br/>Runner logic
    loop Until _stop_event is set
        Thread->>Thread: Sample Hardware<br/>(psutil, nvml, etc.)
        Thread->>Utils: CSVWriter.write_sample()
        Thread->>Utils: MetricAccumulator.add()
        Thread->>Thread: Update self.metrics<br/>(Real-time)
        Thread->>Thread: sleep(interval)
    end

    %% 4. Teardown & Aggregation
    Runner->>Manager: __exit__()
    activate Manager
    Manager->>Manager: stop_all()
    
    loop For each profiler
        Manager->>Base: stop_monitoring()
        activate Base
        Base->>Thread: _stop_event.set()
        Base->>Thread: join()
        Thread-->>Base: Thread Finished
        deactivate Thread
        Base->>Base: get_metrics()<br/>(Final Aggregation)
        Base-->>Manager: metrics dict
        deactivate Base
    end

    Manager->>Manager: Collect all_metrics
    deactivate Manager
