from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class WNLI(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "wnli"
        self.task_type = "classification"
        self.label = {
            1: "entailment",
            0: "not entailment",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["sentence1"] = dp["sentence1"]
        d["sentence2"] = dp["sentence2"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["idx"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "wnli", split="train")
        hf_test = load_dataset("nyu-mll/glue", "wnli", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = WNLI()
    dataset.run()

if __name__ == "__main__":
    main()