def get_model_configs(model_name):
    model_name = model_name.lower()
    if model_name == "mistral-7b":
        model_config = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "gpu_memory_utilization": 0.7,
        }
    elif model_name == "film-7b":
        model_config = {
            "model": "In2Training/FILM-7B",
            "gpu_memory_utilization": 0.7,
        }
    elif model_name == "llama2-7b":
        model_config = {
            "model": "meta-llama/Llama-2-7b-hf",
        }
    elif model_name == "llama2-7b-32k":
        model_config = {
            "model": "togethercomputer/LLaMA-2-7B-32K",
            "max_model_len": 32768,
        }
    elif model_name == "llama2-7b-80k":
        model_config = {
            "model": "yaofu/llama-2-7b-80k",
            "max_model_len": 36000,
        }
    elif model_name == "yi-6b-200k":
        model_config = {
            "model": "01-ai/Yi-6B-200K",
            "max_model_len": 32768,
        }
    elif model_name == "yi-9b-200k":
        model_config = {
            "model": "01-ai/Yi-9B-200K",
            "max_model_len": 32768,
        }
    elif model_name == "yi-34b-200k":
        model_config = {
            "model": "01-ai/Yi-34B-200K",
            "max_model_len": 32768,
            "tensor_parallel_size": 2
        }
    elif model_name == "llama3-8b-1048k":
        model_config = {
            "model": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
            "max_model_len": 32768,
        }
    elif model_name == "opt-125m": # for fast prototyping
        model_config = {
            "model": "facebook/opt-125m"
        }
    elif model_name.lower() == "phi3-small-128k-instruct":
        model_config = {
            "model": "microsoft/Phi-3-mini-128k-instruct",
            "trust_remote_code": True,
            "max_model_len": 33000,
        }
    elif model_name.lower() == "command-r":
        model_config = {
            "model": "CohereForAI/c4ai-command-r-v01",
            "max_model_len": 33000,
            "tensor_parallel_size": 4,
            "distributed_executor_backend": "ray",
        }
    elif model_name.lower() == "llama3-70b-1048k":
        model_config = {
            "model": "gradientai/Llama-3-70B-Instruct-Gradient-1048k",
            "max_model_len": 33000,
            "tensor_parallel_size": 4,
            "distributed_executor_backend": "ray",
        }
    else:
        # TODO : set your model here
        raise ValueError(f"Model name not recognized: {model_name}")
    
    model_config["enable_prefix_caching"] = True
    model_config["max_logprobs"] = 100
    return model_config
