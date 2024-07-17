from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class CB(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "cb"
        self.task_type = "classification"
        self.label = {
            0: "entailment",
            1: "contradiction",
            2: "neutral"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["premise"] = dp["premise"]
        d["hypothesis"] = dp["hypothesis"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("super_glue", "cb", split="train")
        hf_test = load_dataset("super_glue", "cb", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = CB()
    dataset.run()

if __name__ == "__main__":
    main()