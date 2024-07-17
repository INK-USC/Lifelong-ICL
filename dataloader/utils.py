import json
import random

def read_jsonl(filename):
    data = []
    with open(filename) as fin:
        for l in fin:
            data.append(json.loads(l))
    return data

def read_json(filename):
    with open(filename) as fin:
        config = json.load(fin)
    return config

def generate_random_permutations(lst, n_item, n_permutations, seed):
    random.seed(seed)
    permutations = []
    for _ in range(n_permutations):
        permutation = random.sample(lst, n_item)
        permutations.append(permutation)
    return permutations