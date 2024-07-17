from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class QNLI(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "qnli"
        self.task_type = "classification"
        self.label = {
            0: "entailment",
            1: "not entailment",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["question"] = dp["question"]
        d["sentence"] = dp["sentence"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["idx"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "qnli", split="train")
        hf_test = load_dataset("nyu-mll/glue", "qnli", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = QNLI()
    dataset.run()

if __name__ == "__main__":
    main()