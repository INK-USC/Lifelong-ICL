from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class WiC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "wic"
        self.task_type = "classification"
        self.label = {
            0: "no",
            1: "yes",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["word"] = dp["word"]
        d["sentence1"] = dp["sentence1"]
        d["sentence2"] = dp["sentence2"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("super_glue", "wic", split="train")
        hf_test = load_dataset("super_glue", "wic", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = WiC()
    dataset.run()

if __name__ == "__main__":
    main()