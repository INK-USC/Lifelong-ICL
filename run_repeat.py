import os
import pandas as pd
import random

from model import get_model
from dataloader.single_task import SingleTaskDataloader
from dataloader.utils import read_json, generate_random_permutations

from utils import evaluate, seed_everything, get_irrelevant_context

MAX_SEQ_LEN = 32768

def run_repeat(args, logger):
    seed_everything(args.seed)
    
    # Data
    split_file = os.path.join("configs", "{}.json".format(args.split))
    task_list = read_json(split_file)["train"]
    if args.n_repeat is None: # by default set n_repeat to the numbert of train tasks
        args.n_repeat = args.n_task

    # Model
    model = get_model(args, logger)
    model.set_up_model()

    # Save Results
    results_file = os.path.join(args.output_dir, "results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file, index_col=0)
    else:
        df = pd.DataFrame(columns=["n_task", "n_shot", "fewshot_sample_id", "n_repeat", "global_prefix_n_tokens", "task_name", "accuracy", "macro_f1", "ood_rate"])
    
    # in cases where we don't need all tasks, we select a subset of it
    if args.n_task != len(task_list):
        permutations = generate_random_permutations(task_list, n_item=len(task_list), n_permutations=1, seed=args.seed)
        task_list = permutations[0][:args.n_task]

    for task_name in task_list:

        dataloader = SingleTaskDataloader(args, logger)
        dataset_config, train_sets, test_set = dataloader.load_data(task_name)
        n_class = len(dataset_config["options"])
    
        for j in range(5):
            if not df[(df['fewshot_sample_id'] == j) & (df['task_name'] == task_name)].empty:
                logger.info(f"task: {task_name}, fewshot_sample_id: {j}, tested, skip.")
                continue
    
            prefix = dataloader.prepare_prefix(dataset_config, train_sets[j][:args.n_shot * n_class])
            prefix_n_tokens = model.get_n_tokens(prefix)
            n_repeat = min(args.n_repeat, MAX_SEQ_LEN // prefix_n_tokens)

            if args.repeat_shuffle:
                examples = train_sets[j][:args.n_shot * n_class]
                repeated_examples = examples.copy()
                for j in range(n_repeat-1):
                    random.shuffle(examples)
                    repeated_examples.extend(examples)
                repeated_prefix = dataloader.prepare_prefix(dataset_config, repeated_examples)
            elif args.use_irrlevant_prefix:
                examples = train_sets[j][:args.n_shot * n_class]
                one_time_prefix = dataloader.prepare_prefix(dataset_config, examples)

                prefix_len = model.get_n_tokens(one_time_prefix)
                target_irrelevant_context_len = prefix_len * (n_repeat-1)

                irrelevant_context = get_irrelevant_context(args.irrelevant_dir, target_irrelevant_context_len, model)
                repeated_prefix = irrelevant_context + args.task_sep + one_time_prefix
            else:
                repeated_examples = train_sets[j][:args.n_shot * n_class] * n_repeat
                repeated_prefix = dataloader.prepare_prefix(dataset_config, repeated_examples)

            repeated_prefix_n_tokens = model.get_n_tokens(repeated_prefix)

            prompts = dataloader.prepare_prompts(dataset_config, test_set)

            if args.inference_method == "greedy":
                raw_predictions = model.run_greedy(repeated_prefix, prompts)
            elif args.inference_method == "rank":
                raw_predictions = model.run_rank(repeated_prefix, prompts, options=dataset_config["options"])

            true_labels = [item["label"] for item in test_set]
            predictions = [item["prediction"] for item in raw_predictions]
            acc, macro_f1, ood_rate = evaluate(args, true_labels, predictions, dataset_config["options"])
            
            logger.info("task: {}, train_set_id: {}, n_shot_per_class: {}, n_repeat: {}, prefix_tokens: {}, acc: {:.2f}, macro_f1: {:.4f}, ood_rate: {:.2f}".format(task_name, j, args.n_shot, n_repeat, repeated_prefix_n_tokens, acc, macro_f1, ood_rate))
            df.loc[len(df.index)] = [len(task_list), args.n_shot, j, n_repeat, repeated_prefix_n_tokens, task_name, acc, macro_f1, ood_rate]
            df.to_csv(results_file)