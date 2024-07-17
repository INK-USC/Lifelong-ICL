import os
import json
import random
import numpy as np


N_SAMPLES = 5
DATA_DIR = "../../data"

class Task():
    def __init__(self):
        self.identifier = None
        self.task_type = None
        self.label_field = "label"
        
    def map_hf_dp_to_dict(self, dp):
        raise NotImplementedError
    
    def write_to_jsonl(self, datapoints, out_file):
        with open(out_file, "w") as fout:
            for dp in datapoints:
                d = self.map_hf_dp_to_dict(dp)
                fout.write(json.dumps(d)+"\n")

    def create_unique_ids(self, train_sets, test_set):
        for train_set_id in range(len(train_sets)):
            train_set = train_sets[train_set_id]
            for example_id in range(len(train_set)):
                train_set[example_id]["id"] = "train_{}_{}".format(train_set_id, example_id)
        for example_id in range(len(test_set)):
            test_set[example_id]["id"] = "test_{}".format(example_id)
        return train_sets, test_set

    def sample_train_test_set(self, all_set, test_percent=0.2, seed=42):
        split_data = all_set.train_test_split(test_size=test_percent, seed=seed)
    
        training_set = split_data['train']
        test_set = split_data['test']

        return training_set, test_set


class ClassificationTask(Task):
    def __init__(self):
        super().__init__()
        self.label_field = "label"

    def load_hf_dataset(self):
        raise NotImplementedError

    def load_and_sample_data(self, k=20, n_test=100, seed=42):
        hf_train_set, hf_test_set = self.load_hf_dataset()

        hf_train_set = hf_train_set.shuffle(seed=seed)
        classes = list(self.label.keys())

        d = {}
        n = k * N_SAMPLES
        for c in classes:
            subset = hf_train_set.filter(lambda x: x[self.label_field] == c)
            d[c] = subset.select(range(min(n, len(subset))))

        random.seed(seed)
        train_sets = []
        for i in range(N_SAMPLES):
            train_set = []
            for j in range(k):
                examples = [d[c][(i*k+j) % len(d[c])] for c in classes]
                random.shuffle(examples)
                train_set += examples
            train_sets.append(train_set)

        hf_test_set = hf_test_set.filter(lambda x: x[self.label_field] in classes)
        test_set = hf_test_set.shuffle(seed=seed).select(range(min(n_test, len(hf_test_set))))
        test_set = [item for item in test_set]

        return train_sets, test_set

    def run(self):
        print("Processing {}".format(self.identifier))
        train_sets, test_set = self.load_and_sample_data()
        train_sets, test_set = self.create_unique_ids(train_sets, test_set)

        task_outdir = os.path.join(DATA_DIR, self.identifier)
        if not os.path.exists(task_outdir): os.makedirs(task_outdir)

        for i in range(N_SAMPLES):
            filename = os.path.join(task_outdir, "train_{}.jsonl".format(i))
            self.write_to_jsonl(train_sets[i], filename)
        
        filename = os.path.join(task_outdir, "test.jsonl")
        self.write_to_jsonl(test_set, filename)

