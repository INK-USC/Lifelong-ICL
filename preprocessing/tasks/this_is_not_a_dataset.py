from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class NotDataset(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "this_is_not_a_dataset"
        self.task_type = "classification"
        self.label = {
            True: "true",
            False: "false"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["sentence"] = dp["sentence"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("HiTZ/This-is-not-a-dataset", split="train")
        hf_test = load_dataset("HiTZ/This-is-not-a-dataset", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = NotDataset()
    dataset.run()

if __name__ == "__main__":
    main()