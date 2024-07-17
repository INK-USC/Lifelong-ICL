import os
import random

from .dataloader import Dataloader
from .utils import read_jsonl, read_json

class SingleTaskDataloader(Dataloader):
    def load_data(self, task_name):
        self.logger.info("loading data for {}".format(task_name))
        data_dir = os.path.join(self.args.data_dir, task_name)
        self.config = read_json(os.path.join(data_dir, "config.json"))
        self.train_sets = [read_jsonl(os.path.join(data_dir, "train_{}.jsonl").format(i)) for i in range(5)]
        self.test_set = read_jsonl(os.path.join(data_dir, "test.jsonl"))
        return self.config, self.train_sets, self.test_set

    def prepare_prefix(self, dataset_config, training_examples, use_paraphrase=False, use_random_labels=False, use_wrong_labels=False):
        template = dataset_config["demonstration_prompt"]
        instruction_key = "instruction" if not use_paraphrase else "instruction_2" # in controlled experiments we use paraphrasd instructions
        prefixes = [dataset_config[instruction_key]]
        
        if use_random_labels:
            training_examples = self.perturb_with_random_labels(training_examples)
        elif use_wrong_labels:
            training_examples = self.perturb_with_wrong_labels(training_examples)

        for example in training_examples:
            prefixes.append(template.format(**example))
        prefix = self.args.demo_sep.join(prefixes) # usually self.args.demo_sep is "\n"
        return prefix
    
    def prepare_prompts(self, dataset_config, inference_examples):
        template = dataset_config["inference_prompt"]
        prompts = []
        for example in inference_examples:
            prompts.append(template.format(**example))
        return prompts

    def perturb_with_random_labels(self, training_examples):
        new_training_examples = []
        for item in training_examples:
            new_item = item.copy()
            new_item["label"] = random.choice(self.config["options"])
            new_training_examples.append(new_item)
        return new_training_examples

    def perturb_with_wrong_labels(self, training_examples):
        new_training_examples = []
        for item in training_examples:
            new_item = item.copy()
            wrong_options = self.config["options"].copy()
            wrong_options.remove(new_item["label"])
            new_item["label"] = random.choice(wrong_options)
            new_training_examples.append(new_item)
        return new_training_examples 