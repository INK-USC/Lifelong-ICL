from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class BabiNLI(ClassificationTask):

    def __init__(self, subset_name):
        super().__init__()
        self.identifier = "babi_nli_{}".format(subset_name.replace("-", "_"))
        self.subset_name = subset_name
        self.task_type = "classification"
        self.label = {
            0: "False",
            1: "True"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["premise"] = dp["premise"]
        d["hypothesis"] = dp["hypothesis"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("tasksource/babi_nli", self.subset_name, split="train")
        hf_test = load_dataset("tasksource/babi_nli", self.subset_name, split="validation")
        return hf_train, hf_test
    
def main():
    subset_name = "single-supporting-fact"
    dataset = BabiNLI(subset_name=subset_name)
    dataset.run()

if __name__ == "__main__":
    main()