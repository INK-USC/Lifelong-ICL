import os

from .dataloader import Dataloader
from .single_task import SingleTaskDataloader
from .utils import read_jsonl, read_json, generate_random_permutations

class IncrementalDataloader(Dataloader):
    def load_data(self, task_list):
        self.logger.info("loading data for {}".format(task_list))
        self.task_list = task_list
        self.data = {}
        for task_name in task_list:
            single_task_dataloader = SingleTaskDataloader(self.args, self.logger)
            single_task_dataloader.load_data(task_name)
            self.data[task_name] = single_task_dataloader

    def create_permutation(self, task_list, n, seed):
        permutations = generate_random_permutations(task_list, len(task_list), n, seed)
        return permutations

    def prepare_prefix(self, permutation, train_set_id, test_idx_list):
        prefixes = []
        for task_name in permutation:
            config, train_set = self.data[task_name].config, self.data[task_name].train_sets[train_set_id]
            n_class = len(config["options"])
            training_examples = train_set[:self.args.n_shot * n_class] # n_shot_per_class
            prefix = self.data[task_name].prepare_prefix(config, training_examples)
            prefixes.append(prefix)

        prefix_list = []
        for i in test_idx_list:
            prefix = self.args.task_sep.join(prefixes[:i+1]) # include the first i-th prefix
            prefix_list.append(prefix)
        return prefix_list