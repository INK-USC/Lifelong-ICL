# from .model import Model
from .vllm import vLLM
from .cohere import Cohere
from .openai import OpenAI

def get_model(args, logger):
    if args.backend == "vllm":
        return vLLM(args, logger)
    elif args.backend == "cohere":
        return Cohere(args, logger)
    elif args.backend == "openai":
        return OpenAI(args, logger)
    else:
        raise NotImplementedError("Supported backends: vllm, cohere, openai")
