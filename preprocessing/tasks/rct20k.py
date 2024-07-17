from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class RCT20k(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "rct20k"
        self.task_type = "classification"
        # conclusions is considered as two tokens (con-clusions) but conclusion is one token.
        self.label = {
            "objective": "objective",
            "conclusions": "conclusion",
            "results": "result",
            "methods": "method",
            "background": "background"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("armanc/pubmed-rct20k", split="train")
        hf_test = load_dataset("armanc/pubmed-rct20k", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = RCT20k()
    dataset.run()

if __name__ == "__main__":
    main()