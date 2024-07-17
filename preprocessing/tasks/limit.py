from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class Limit(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = "motion"
        self.identifier = "limit"
        self.task_type = "classification"
        self.label = {
            "yes": "yes",
            "no": "no"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["sentence"]
        d["label"] = self.label[dp["motion"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("IBM/limit", split="train")
        hf_test = load_dataset("IBM/limit", split="test")
        return hf_train, hf_test
    
def main():
    dataset = Limit()
    dataset.run()

if __name__ == "__main__":
    main()