import argparse
import logging
import os
import pandas as pd
import json
import sys
import math

from dataloader.single_task import SingleTaskDataloader
from dataloader.multi_task import MultiTaskDataloader
from dataloader.utils import read_json, generate_random_permutations
from model.openai_batch import OpenAIBatch

def create_jsonl(args, logger):
    model = OpenAIBatch(args, logger)
    model.set_up_model()

    # do as run_baseline.py
    split_file = os.path.join("configs", "{}.json".format(args.split))
    task_list = read_json(split_file)["train"]
    
    # in cases where we don't need all tasks, we select a subset of it
    if args.n_task != len(task_list):
        permutations = generate_random_permutations(task_list, n_item=len(task_list), n_permutations=1, seed=args.seed)
        task_list = permutations[0][:args.n_task]

    train_set_range = 5 if args.n_shot != 0 else 1
    for train_set_id in range(train_set_range):
        
        tasks_per_file = math.ceil(len(task_list) / args.n_split_perm)
        for j in range(args.n_split_perm):
            testcase_list = []
            start_index = j * tasks_per_file
            end_index = min(start_index + tasks_per_file, len(task_list))

            for test_task_name in task_list[start_index: end_index]:
                test_dataloader = SingleTaskDataloader(args, logger)
                dataset_config, train_sets, test_set = test_dataloader.load_data(test_task_name)
                n_class = len(dataset_config["options"])

                prefix = test_dataloader.prepare_prefix(dataset_config, train_sets[train_set_id][:args.n_shot * n_class])
                prompts = test_dataloader.prepare_prompts(dataset_config, test_set)

                whole_prompts = [prefix + args.demo_sep + prompt for prompt in prompts]
                        
                # {n_task}#{n_shot}#{perm_id : i}#{few-shot split id}#{test-task-name}#{case-id}
                for case_id, _prompt in enumerate(whole_prompts):
                    request_json = {
                        "custom_id": f"{args.n_task}#{args.n_shot}#{-1}#{train_set_id}#{test_task_name}#{case_id}#{0}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": model.prepare_request_body(_prompt, args.inference_method)
                    }
                    testcase_list.append(request_json)
            
            # save this request to jsonl
            file_name = os.path.join(args.output_dir, "permutations", f"ffs_{train_set_id}_part_{j}.jsonl")
            if len(testcase_list) > 0:
                with open(file_name, 'w') as file:
                    for obj in testcase_list:
                        file.write(json.dumps(obj) + '\n')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="output/pilot/openai")
    parser.add_argument("--permutation_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--split", default="pilot", type=str, 
                        help="A list of tasks to run")
    parser.add_argument("--n_shot", default=2, type=int, 
                        help="Number of examples per class (!), for the meta-train tasks")
    parser.add_argument("--n_task", default=-1, type=int, 
                         help="Number of tasks, for the meta-train tasks")
    parser.add_argument("--inference_method", default="rank", choices=["rank", "greedy"])
    parser.add_argument("--model", default="gpt-4o", choices=["gpt-3.5", "gpt-4o"])
    parser.add_argument("--n_split_perm", type=int, default=1)
    parser.add_argument("--task_sep", default="\n\n", help="Separator between tasks.")
    parser.add_argument("--demo_sep", default="\n", help="Separator between the in-context examples")

    args = parser.parse_args()

    # output_dir should be a subdir of data_dir
    args.output_dir = os.path.join(args.output_dir, "inputs")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "permutations"), exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    create_jsonl(args, logger)


if __name__ == "__main__":
    main()