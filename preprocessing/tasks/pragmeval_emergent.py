from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class PragmevalEmergent(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "pragmeval_emergent"
        self.task_type = "classification"
        self.label = {
            0: "observing",
            1: "for",
            2: "against"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["sentence1"] = dp["sentence1"]
        d["sentence2"] = dp["sentence2"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("sileod/pragmeval", "emergent", split="train")
        hf_test = load_dataset("sileod/pragmeval", "emergent", split="test")
        return hf_train, hf_test
    
def main():
    dataset = PragmevalEmergent()
    dataset.run()

if __name__ == "__main__":
    main()