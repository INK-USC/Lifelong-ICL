from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class BeaverTails(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "beaver_tails"
        self.task_type = "classification"
        self.label = {
            True: "yes",
            False: "no"
        }
        self.label_field = "is_safe"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["prompt"] = dp["prompt"]
        d["response"] = dp["response"]
        d["label"] = self.label[dp["is_safe"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        hf_test = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        return hf_train, hf_test
    
def main():
    dataset = BeaverTails()
    dataset.run()

if __name__ == "__main__":
    main()