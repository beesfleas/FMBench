classDiagram
    %% Factory
    model_factory --> BaseModelLoader : creates

    class model_factory {
        _MODEL_LOADERS: Dict
        get_model_loader(config)
    }

    %% Base Interface
    BaseModelLoader <|-- HuggingFaceLLMLoader
    BaseModelLoader <|-- HuggingFaceVLMLoader
    BaseModelLoader <|-- HuggingFaceTimeSeriesLoader

    class BaseModelLoader {
        <<abstract>>
        load_model(config)
        predict(prompt, image, time_series_data)
        unload_model()
        compute_perplexity(text)
    }

    %% Implementations
    class HuggingFaceLLMLoader {
        model
        tokenizer
        config
        load_model(config)
        predict(prompt)
        unload_model()
        compute_perplexity(text)
    }

    class HuggingFaceVLMLoader {
        model
        processor
        config
        load_model(config)
        predict(prompt, image)
        unload_model()
        _process_image(image)
        _get_model_inputs(prompt, image)
    }

    class HuggingFaceTimeSeriesLoader {
        model
        config
        device
        pipeline
        load_model(config)
        predict(prompt, time_series_data)
        unload_model()
    }

    %% Relationships / Helpers
    HuggingFaceLLMLoader ..> device_utils : uses
    HuggingFaceVLMLoader ..> device_utils : uses
    HuggingFaceTimeSeriesLoader ..> device_utils : uses
    
    class device_utils {
        get_mps_safe_load_kwargs()
        move_to_device()
        clear_device_cache()
    }
