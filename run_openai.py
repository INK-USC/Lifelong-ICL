import argparse
import logging
import os
import pandas as pd
import json
import sys
import time
from openai import OpenAI
from types import SimpleNamespace

from dataloader.single_task import SingleTaskDataloader
from dataloader.utils import read_json
from model.openai_batch import OpenAIBatch

from utils import evaluate


# evaluate results of all tasks in one permuation
def evaluate_result(model, result_list, df, results_file, args, logger):
    split_file = os.path.join("configs", "{}.json".format(args.split))
    all_test_task_list = read_json(split_file)["test"]
    
    # collect all results for each task in this split
    test_task_list = set()
    task_results = {task_name : [] for task_name in all_test_task_list}
    for result in result_list:
        # {n_task}#{n_shot}#{perm_id}#{few-shot split id}#{test-task-name}#{case-id}#{global_prefix_n_tokens}
        unique_id = result.custom_id.split("#")
        n_task, n_shot, perm_id, few_shot_sample_id, task_name, case_id, global_prefix_n_tokens = unique_id
        output = result.response.body
        test_task_list.add(task_name)
        task_results[task_name].append({"case_id": case_id, "result": output})
    
    for task_name in test_task_list:
        cases = task_results[task_name]
        sorted_cases = sorted(cases, key=lambda x: int(x["case_id"]))
        results = [item["result"] for item in sorted_cases]

        # collect all preds for each task
        test_dataloader = SingleTaskDataloader(args, logger)
        dataset_config, _, test_set = test_dataloader.load_data(task_name)
            
        if args.inference_method == "rank":
            raw_predictions = model.get_predictions_rank(results, dataset_config["options"])
        elif args.inference_method == "greedy":
            raw_predictions = model.get_predictions_greedy(results, results)

        true_labels = [item["label"] for item in test_set]
        predictions = [item["prediction"] for item in raw_predictions]
        acc, macro_f1, ood_rate = evaluate(args, true_labels, predictions, dataset_config["options"])
        logger.info("n_task: {}, n_shot: {}, perm_id: {}, fewshot_sample_id: {}, task: {}, acc: {:.2f}, macro_f1: {:.4f}, ood_rate: {:.2f}".format(n_task, n_shot, perm_id, few_shot_sample_id, task_name, acc, macro_f1, ood_rate))
        df.loc[len(df.index)] = [n_task, n_shot, perm_id, few_shot_sample_id, global_prefix_n_tokens, task_name, acc, macro_f1, ood_rate]
        df.to_csv(results_file)

def load_job_status(logger, args):
    file_name = os.path.join(args.result_dir, "status.json")
    if not os.path.exists(file_name) or args.restart:
        return []
    
    batch_list = []
    with open(file_name, 'r') as file:
        for line in file.readlines():
            batch_list.append(json.loads(line))
    return batch_list

def save_job_status(batch_job, logger, args):
    file_name = os.path.join(args.result_dir, "status.json")
    with open(file_name, 'w') as file:
        for obj in batch_job:
            file.write(json.dumps(obj) + '\n')

def run_and_evaluate(logger, args):
    model = OpenAIBatch(args, logger)
    model.set_up_model()

    # results_file
    results_file = os.path.join(args.result_dir, "results.csv")
    if os.path.exists(results_file) and not args.restart:
        df = pd.read_csv(results_file, index_col=0)
    else:
        # df = pd.DataFrame(columns=["permutation_id", "task_name", "n_shot_per_class", "accuracy", "macro_f1", "ood_rate"])
        df = pd.DataFrame(columns=["n_task", "n_shot", "permutation_id", "fewshot_sample_id", "global_prefix_n_tokens", "task_name", "accuracy", "macro_f1", "ood_rate"])

    # send requests
    batch_job_list = load_job_status(logger, args)

    # delete failed jobs from the queue
    batch_job_list = [batch_job for batch_job in batch_job_list if batch_job["status"] != "failed"]
    save_job_status(batch_job_list, logger, args)

    for file_name in os.listdir(args.input_dir):
        if file_name not in [x["file_name"] for x in batch_job_list]:
            logger.info(f"Sending batch request: {file_name}")
            full_name = os.path.join(args.input_dir, file_name)
            batch_job_id = model.send_batch_request(full_name)
            batch_job_list.append({"file_name": file_name, 
                                    "batch_job_id": batch_job_id, 
                                    "status": "pending", 
                                    "request_counts": {"total": 0, "completed": 0, "failed": 0}
                                    })
            save_job_status(batch_job_list, logger, args)
    
    # retrieve requests in a loop
    while any(batch_job["status"] != "saved" for batch_job in batch_job_list):
        for i, batch_job in enumerate(batch_job_list):
            batch_job_list[i], content = model.retrieve_batch_request(batch_job, args.output_dir)
            
            if content is not None:
                evaluate_result(model, content, df, results_file, args, logger)
                batch_job_list[i]["status"] = "saved"

        # log all batch status
        for batch_job in batch_job_list:
            file_name, status, total, completed, failed = batch_job["file_name"], batch_job["status"], batch_job["request_counts"]["total"], batch_job["request_counts"]["completed"], batch_job["request_counts"]["failed"]
            logger.info(f"batch: {file_name}, status: {status}, {completed} / {total} ({failed} failed)")
        
        save_job_status(batch_job_list, logger, args)

        # the batch work may take up to hours if there's a long queue
        # we might not want to keep this program running
        if not args.keep_waiting: 
            break

        if any(batch_job["status"] != "saved" for batch_job in batch_job_list):
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--result_dir", default="output/pilot/openai")
    parser.add_argument("--split", default="pilot", type=str,
                        help="A list of tasks to run")
    parser.add_argument("--model", default="gpt-4o", choices=["gpt-3.5", "gpt-4o"])
    parser.add_argument("--inference_method", default="rank", choices=["rank", "greedy"])
    parser.add_argument("--restart", action='store_true')
    parser.add_argument("--keep_waiting", action="store_true")
    
    args = parser.parse_args()
    args.input_dir = os.path.join(args.result_dir, "inputs", "permutations")
    args.output_dir = os.path.join(args.result_dir, "outputs")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.result_dir, "log.txt")),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    run_and_evaluate(logger, args)

if __name__ == "__main__":
    main()