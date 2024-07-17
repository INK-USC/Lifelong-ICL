from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SST5(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "sst5"
        self.task_type = "classification"
        self.label = {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "extremely positive" # very positive
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("SetFit/sst5", split="train")
        hf_test = load_dataset("SetFit/sst5", split="test")
        return hf_train, hf_test
    
def main():
    dataset = SST5()
    dataset.run()

if __name__ == "__main__":
    main()