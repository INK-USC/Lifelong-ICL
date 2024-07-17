from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class TREC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "trec"
        self.task_type = "classification"
        self.label = {
            0: "Abbreviation",
            1: "Entity",
            2: "Description",
            3: "Person",
            4: "Location",
            5: "Number"
        }
        self.label_field = "coarse_label"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["coarse_label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("trec", split="train")
        hf_test = load_dataset("trec", split="test")
        return hf_train, hf_test
    
def main():
    dataset = TREC()
    dataset.run()

if __name__ == "__main__":
    main()