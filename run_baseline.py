import os
import pandas as pd

from model import get_model
from dataloader.single_task import SingleTaskDataloader
from dataloader.utils import read_json

from utils import evaluate

def run_baseline(args, logger):

    # Data
    split_file = os.path.join("configs", "{}.json".format(args.split))
    task_list = read_json(split_file)["test"]

    # Model
    model = get_model(args, logger)
    model.set_up_model()

    # Save Results
    df = pd.DataFrame(columns=["task_name", "train_set_id", "n_shot_per_class", "accuracy", "macro_f1", "ood_rate"])
    results_file = os.path.join(args.output_dir, "results.csv")

    for task_name in task_list:

        dataloader = SingleTaskDataloader(args, logger)
        dataset_config, train_sets, test_set = dataloader.load_data(task_name)
        n_class = len(dataset_config["options"])

        for n_shot_per_class in [0,1,2,3,4,5,6,7,8,12,16]:
            for train_set_id in range(5): # 5 random samples
                prefix = dataloader.prepare_prefix(dataset_config, train_sets[train_set_id][:n_shot_per_class * n_class])
                prompts = dataloader.prepare_prompts(dataset_config, test_set)

                if args.inference_method == "greedy":
                    raw_predictions = model.run_greedy(prefix, prompts)
                elif args.inference_method == "rank":
                    raw_predictions = model.run_rank(prefix, prompts, options=dataset_config["options"])

                true_labels = [item["label"] for item in test_set]
                predictions = [item["prediction"] for item in raw_predictions]
                acc, macro_f1, ood_rate = evaluate(args, true_labels, predictions, dataset_config["options"])
                
                logger.info("task: {}, train_set_id: {}, n_shot_per_class: {}, acc: {:.2f}, macro_f1: {:.4f}, ood_rate: {:.2f}".format(task_name, train_set_id, n_shot_per_class, acc, macro_f1, ood_rate))
                df.loc[len(df.index)] = [task_name, train_set_id, n_shot_per_class, acc, macro_f1, ood_rate]
                df.to_csv(results_file)

                # don't need to repeat 5 times for 0-shot experiments
                if n_shot_per_class == 0: break