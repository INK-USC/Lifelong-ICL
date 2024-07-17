import os
import sys
import argparse
import logging

import random
import numpy as np

from .run_niah import run_niah
from .run_evaluate import run_evaluate
from model.vllm import vLLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_niah_all(args, output_dir):
    for _prompt_name in ["SFquestion"]:
        args.prompt_name = _prompt_name
        args.prompt_file = os.path.join(args.prompt_dir, f"{args.prompt_name}.json")

        if "passkey" in _prompt_name:
            _passkeys = ["32123432", "quasar", "ska71h2j8"]
        else:
            _passkeys = ["default"]
        
        for _pass_key in _passkeys:
            args.pass_key = _pass_key
            args.scorer = "hard_check"

            logger = run_niah_wrapper(args, output_dir)

            if "passkey" not in _prompt_name:
                for _scorer in ["rouge"]:
                    run_evaluate(args, _scorer, logger)


def run_niah_wrapper(args, output_dir):
    
    args.output_dir = os.path.join(output_dir, f'{args.prompt_name}_{args.pass_key}', f'{args.min_length}_{args.max_length}_{args.test_length_n}_{args.depth_pcnt_n}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    args.model_to_evaluate.logger = logger    
    run_niah(args, logger)

    return logger


def main():
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("--haystack_dir", default="data")
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    # Exp
    parser.add_argument("--rouge", default="rouge_1", type=str, choices=["rouge_1", "rouge_l"])
    parser.add_argument("--context_check", action="store_true", help="Store context if set.")
    parser.add_argument("--all", action="store_true", help="Run different setting loading model once.")

    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--prompt_name", type=str, default="SFquestion")
    parser.add_argument("--pass_key", default="default", type=str)

    # scorer
    parser.add_argument("--scorer", default="hard_check", type=str, choices=["hard_check","rouge"],
                        help="hard_check: contained & exact match \n rouge: rouge-1 and rouge-L")
    
    # LLM
    parser.add_argument("--backend", default="vllm", choices=["vllm"])
    parser.add_argument("--model", default="llama2-13b", required=True)

    # NITH
    parser.add_argument("--max_length", default=32000, type=int)
    parser.add_argument("--min_length", default=100, type=int)
    parser.add_argument("--test_length_n", default=8, type=int)
    parser.add_argument("--depth_pcnt_n", default=8, type=int)

    args = parser.parse_args()
    args.prompt_file = os.path.join(args.prompt_dir, f"{args.prompt_name}.json")

    # logger for setting up model
    model_output_dir = os.path.join(args.output_dir, args.model)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(model_output_dir, "log.txt")),
                                logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    model = vLLM(args, logger)
    model.set_up_model()
    args.model_to_evaluate = model

    if args.all:
        run_niah_all(args, model_output_dir)
    else:
        run_niah_wrapper(args, model_output_dir)

if __name__=='__main__':
    main()