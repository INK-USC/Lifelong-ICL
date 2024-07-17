import numpy as np
import os
import random
import torch 
import glob

from sklearn.metrics import f1_score, accuracy_score

def evaluate(args, labels, predictions, options):
    acc = accuracy_score(y_true=labels, y_pred=predictions)
    macro_f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")

    if args.inference_method == "greedy":
        ood = [0.0 if pred in options else 1.0 for pred in predictions]
        ood_rate = sum(ood) / len(ood)
    elif args.inference_method == "rank":
        ood = [0.0 if pred != "NO_PREDICTION" else 1.0 for pred in predictions]
        ood_rate = sum(ood) / len(ood)

    return acc, macro_f1, ood_rate

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
     
def get_irrelevant_context(data_dir, max_context_length, model):
    context = ""
    while model.get_n_tokens(context) < max_context_length:
        for file in glob.glob(os.path.join(data_dir, "*.txt")):
            with open(file, 'r') as f:
                context += f.read()

    tokenized_context = model.encode_text_to_tokens(context)
    truncaed = tokenized_context[:max_context_length]
    truncaed_context = model.decode_tokens(truncaed)

    return truncaed_context