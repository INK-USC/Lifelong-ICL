import os
import pandas as pd

from model import get_model
from dataloader.single_task import SingleTaskDataloader
from dataloader.multi_task import MultiTaskDataloader
from dataloader.utils import read_json, generate_random_permutations

from utils import evaluate

def run(args, logger):
    # Data
    split_file = os.path.join("configs", "{}.json".format(args.split))
    task_list = read_json(split_file)["train"]

    # Model
    model = get_model(args, logger)
    model.set_up_model()

    # Save Results
    results_file = os.path.join(args.output_dir, "results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file, index_col=0)
    else:
        df = pd.DataFrame(columns=["n_task", "n_shot", "permutation_id", "fewshot_sample_id", "global_prefix_n_tokens", "task_name", "accuracy", "macro_f1", "ood_rate"])
    
    dataloader = MultiTaskDataloader(args, logger)

    # in cases where we don't need all tasks, we select a subset of it
    if args.n_task != len(task_list):
        permutations = generate_random_permutations(task_list, n_item=len(task_list), n_permutations=1, seed=args.seed)
        task_list = permutations[0][:args.n_task]

    dataloader.load_data(task_list)

    # create permutations and save to disk
    permutations = dataloader.create_permutation(task_list, n=args.permutation_num, seed=args.seed)
    perm_file = os.path.join(args.output_dir, "permutations.csv")
    df_perm = pd.DataFrame(permutations)
    df_perm.to_csv(perm_file, index=True)

    for i, perm in enumerate(permutations):
        logger.info("Permutation {}: {}".format(i, perm))
        
        for j in range(5):  # 5 random few-shot splits

            if (args.use_random_labels or args.use_wrong_labels) and (args.global_perturbation):
                global_prefix = dataloader.prepare_prefix(perm, j, perturb=perm) # perturb labels for all tasks
            else:
                global_prefix = dataloader.prepare_prefix(perm, j) # normal: use correct label for all tasks
                
            global_prefix_n_tokens = model.get_n_tokens(global_prefix)

            for test_task_name in task_list:
                if not df[(df['permutation_id'] == i) & (df['fewshot_sample_id'] == j) & (df['task_name'] == test_task_name)].empty:
                    logger.info(f"perm_id: {i}, fewshot_sample_id: {j}, task: {test_task_name} tested, skip.")
                    continue 

                if args.exp_mode == "remove":
                    global_prefix = dataloader.prepare_prefix(perm, j, exclude=[test_task_name])
                    global_prefix_n_tokens = model.get_n_tokens(global_prefix)
                elif (args.use_random_labels or args.use_wrong_labels) and (not args.global_perturbation): # perturb labels for the current task
                    global_prefix = dataloader.prepare_prefix(perm, j, perturb=[test_task_name])
                    global_prefix_n_tokens = model.get_n_tokens(global_prefix)

                test_dataloader = SingleTaskDataloader(args, logger)
                dataset_config, train_sets, test_set = test_dataloader.load_data(test_task_name)

                # recall: directly do the task; replay: replay the (same set of) few-shot examples
                n_shot_per_class = 0 if args.exp_mode in ["recall", "remove"] else args.n_shot
                n_class = len(dataset_config["options"])
                
                prefix = test_dataloader.prepare_prefix(dataset_config, train_sets[j][:n_shot_per_class * n_class], use_paraphrase=args.use_paraphrase)
                prompts = test_dataloader.prepare_prompts(dataset_config, test_set)

                if args.inference_method == "greedy":
                    raw_predictions = model.run_greedy(global_prefix + args.task_sep + prefix, prompts)
                elif args.inference_method == "rank":
                    raw_predictions = model.run_rank(global_prefix + args.task_sep + prefix, prompts, options=dataset_config["options"])
                
                true_labels = [item["label"] for item in test_set]
                predictions = [item["prediction"] for item in raw_predictions]
                acc, macro_f1, ood_rate = evaluate(args, true_labels, predictions, dataset_config["options"])

                logger.info(f"n_task: {len(task_list)}, n_shot: {args.n_shot}, perm_id: {i}, fsl_id: {j}, prefix_tokens: {global_prefix_n_tokens}, task: {test_task_name}, acc: {acc:.2f}, macro_f1: {macro_f1:.4f}, ood_rate: {ood_rate:.2f}")
                df.loc[len(df.index)] = [len(task_list), args.n_shot, i, j, global_prefix_n_tokens, test_task_name, acc, macro_f1, ood_rate]
                df.to_csv(results_file)
