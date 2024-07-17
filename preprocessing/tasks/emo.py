from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class EMO(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "emo"
        self.task_type = "classification"
        self.label = {
            0: "others",
            1: "happy",
            2: "sad",
            3: "angry",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("emo", split="train")
        hf_test = load_dataset("emo", split="test")
        return hf_train, hf_test
    
def main():
    dataset = EMO()
    dataset.run()

if __name__ == "__main__":
    main()