import os

from .dataloader import Dataloader
from .single_task import SingleTaskDataloader
from .utils import read_jsonl, read_json, generate_random_permutations

class MultiTaskDataloader(Dataloader):
    def load_data(self, task_list):
        self.logger.info("loading data for {}".format(task_list))
        self.task_list = task_list
        self.data = {}
        for task_name in task_list:
            single_task_dataloader = SingleTaskDataloader(self.args, self.logger)
            single_task_dataloader.load_data(task_name)
            self.data[task_name] = single_task_dataloader

    def create_permutation(self, task_list, n=100, seed=42):
        permutations = generate_random_permutations(task_list, len(task_list), n, seed)
        return permutations
    
    def prepare_prefix(self, permutation, train_set_id, exclude=[], perturb=[]):
        prefixes = []
        for task_name in permutation:
            if task_name in exclude:
                continue
            perturb_options = {"use_random_labels" :self.args.use_random_labels, "use_wrong_labels": self.args.use_wrong_labels} if task_name in perturb else {}
            config, train_set = self.data[task_name].config, self.data[task_name].train_sets[train_set_id]
            n_class = len(config["options"])
            training_examples = train_set[:self.args.n_shot * n_class] # n_shot_per_class
            prefix = self.data[task_name].prepare_prefix(config, training_examples, **perturb_options)
            prefixes.append(prefix)
        prefix = self.args.task_sep.join(prefixes)
        return prefix
    
