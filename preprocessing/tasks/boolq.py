from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class BoolQ(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "boolq"
        self.task_type = "classification"
        self.label = {
            0: "no",
            1: "yes",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["question"] = dp["question"]
        d["passage"] = dp["passage"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("super_glue", "boolq", split="train")
        hf_test = load_dataset("super_glue", "boolq", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = BoolQ()
    dataset.run()

if __name__ == "__main__":
    main()