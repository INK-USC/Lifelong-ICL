import os
import sys
import argparse
import logging

import random
import numpy as np

from run_baseline import run_baseline
from run import run
from run_repeat import run_repeat

def main():
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    # Data
    parser.add_argument("--split", default="pilot", type=str,
                        help="A list of tasks to run")
    parser.add_argument("--exp_mode", default="baseline", type=str, choices=["baseline", "recall", "replay", "repeat", "remove"],
                        help="recall: evaluate on seen tasks without new examples; \
                          replay: evaluate on seen/unseen tasks with few-shot examples")

    # Task Haystack Settings    
    parser.add_argument("--permutation_num", type=int, default=5, help="Number of permutations to use")
    parser.add_argument("--n_shot", default=2, type=int, 
                        help="Number of examples per class (!), for the meta-train tasks")
    parser.add_argument("--n_task", default=-1, type=int, 
                        help="Number of tasks, for the meta-train tasks")

    # Continual Settings
    parser.add_argument("--check_context", action="store_true", help="Enable context check if specified")
    parser.add_argument("--incre_min_idx", type=int, default=None, help="Mininum task index for incremental test")
    parser.add_argument("--incre_max_idx", type=int, default=None, help="Maximum task index for incremental test")
    parser.add_argument("--incre_indexes", type=lambda s: [int(item) for item in s.split(',')], default=None, help="Specific indices for incremental test, seperated by ','")
    parser.add_argument("--test_all", action="store_true", help="Disable skipping tested cases")
    
    ## Controlled Experiments Settings
    parser.add_argument("--use_paraphrase", action="store_true", help="Disable skipping tested cases")
    parser.add_argument("--use_random_labels", action="store_true", help="Use random labels in ICL")
    parser.add_argument("--use_wrong_labels", action="store_true", help="Use wrong labels in ICL")
    parser.add_argument("--global_perturbation", action="store_true", help="Use random/wrong labels in ICL for all tasks (by default it's just the test task).")

    parser.add_argument("--n_repeat", default=None, type=int, help="Number of times repeating the few-shot examples")
    parser.add_argument("--repeat_shuffle", action="store_true", help="In repeats, shuffle the order of the examples")
    parser.add_argument("--use_irrlevant_prefix", action="store_true", help="Use irrelevant text as context for control")
    parser.add_argument("--irrelevant_dir", default="needle-in-the-haystack-test/data/PaulGrahamEssays", help="Directory of irrelevant text")

    # Formatting
    parser.add_argument("--task_sep", default="\n\n", help="Separator between tasks.")
    parser.add_argument("--demo_sep", default="\n", help="Separator between the in-context examples")

    # LLM
    parser.add_argument("--backend", default="vllm", choices=["vllm", "cohere", "openai"])
    parser.add_argument("--model", default="llama2-13b", required=True)
    parser.add_argument("--inference_method", default="rank", choices=["rank", "greedy"])

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    # avoid warnings of "too many requests"
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.ERROR)

    if args.exp_mode == "baseline":
        run_baseline(args, logger)
    elif args.exp_mode in ["recall", "replay", "remove"]:
        run(args, logger)
    elif args.exp_mode == "repeat":
        run_repeat(args, logger)

if __name__=='__main__':
    main()