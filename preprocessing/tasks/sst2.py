from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SST2(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "sst2"
        self.task_type = "classification"
        self.label = {
            0: "negative",
            1: "positive",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["sentence"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "sst2", split="train[:10%]")
        hf_test = load_dataset("nyu-mll/glue", "sst2", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = SST2()
    dataset.run()

if __name__ == "__main__":
    main()