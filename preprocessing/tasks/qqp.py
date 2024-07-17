from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class QQP(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "qqp"
        self.task_type = "classification"
        self.label = {
            0: "not duplicate",
            1: "duplicate",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["question1"] = dp["question1"]
        d["question2"] = dp["question2"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "qqp", split="train[:1%]")
        hf_test = load_dataset("nyu-mll/glue", "qqp", split="validation[:10%]")
        return hf_train, hf_test
    
def main():
    dataset = QQP()
    dataset.run()

if __name__ == "__main__":
    main()