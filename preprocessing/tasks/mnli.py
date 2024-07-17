from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class MNLI(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "mnli"
        self.task_type = "classification"
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["premise"] = dp["premise"]
        d["hypothesis"] = dp["hypothesis"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "mnli", split="train")
        hf_test = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
        return hf_train, hf_test
    
def main():
    dataset = MNLI()
    dataset.run()

if __name__ == "__main__":
    main()