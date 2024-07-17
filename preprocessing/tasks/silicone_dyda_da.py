from datasets import load_dataset
import json
from task import ClassificationTask, N_SAMPLES

class SiliconeDydaDA(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = "Label"
        self.identifier = "silicone_dyda_da"
        self.task_type = "classification"
        self.label = {
            1: "directive",
            0: "commissive",
            2: "inform",
            3: "question"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["Utterance"]
        d["label"] = self.label[dp["Label"]]
        d["id"] = dp["id"]
        return d

    def load_hf_dataset(self):
        hf_train = load_dataset("silicone", "dyda_da", split="train")
        hf_test = load_dataset("silicone", "dyda_da", split="test")
        return hf_train, hf_test
    
def main():
    dataset = SiliconeDydaDA()
    dataset.run()

if __name__ == "__main__":
    main()