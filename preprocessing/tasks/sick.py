from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SICK(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "sick"
        self.task_type = "classification"
        self.label = {
            0: "entailment",
            1: "neutral",
            2: "contradiction"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["sentence_A"] = dp["sentence_A"]
        d["sentence_B"] = dp["sentence_B"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("sick", split="train")
        hf_test = load_dataset("sick", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = SICK()
    dataset.run()

if __name__ == "__main__":
    main()