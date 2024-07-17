from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class IsHumor(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "is_humor"
        self.task_type = "classification"
        self.label = {
            0: "not humorous",
            1: "humorous",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("Blablablab/SOCKET", "hahackathon#is_humor", split="train")
        hf_test = load_dataset("Blablablab/SOCKET", "hahackathon#is_humor", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = IsHumor()
    dataset.run()

if __name__ == "__main__":
    main()