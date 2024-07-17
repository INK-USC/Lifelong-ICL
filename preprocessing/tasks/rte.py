from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class RTE(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "rte"
        self.task_type = "classification"
        self.label = {
            0: "True",
            1: "False",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["premise"] = dp["premise"]
        d["hypothesis"] = dp["hypothesis"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("super_glue", "rte", split="train")
        hf_test = load_dataset("super_glue", "rte", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = RTE()
    dataset.run()

if __name__ == "__main__":
    main()