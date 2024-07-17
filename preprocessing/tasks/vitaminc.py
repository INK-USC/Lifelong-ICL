from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class VitaminC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "vitaminc"
        self.task_type = "classification"
        self.label = {
            "SUPPORTS": "supports",
            "REFUTES": "refutes",
            "NOT ENOUGH INFO": "not enough info"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["claim"] = dp["claim"]
        d["evidence"] = dp["evidence"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["unique_id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("tals/vitaminc", split="train")
        hf_test = load_dataset("tals/vitaminc", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = VitaminC()
    dataset.run()

if __name__ == "__main__":
    main()