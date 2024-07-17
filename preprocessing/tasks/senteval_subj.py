from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SentevalSubj(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "senteval_subj"
        self.task_type = "classification"
        self.label = {
            0: "objective",
            1: "subjective"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("SetFit/subj", split="train")
        hf_test = load_dataset("SetFit/subj", split="test")
        return hf_train, hf_test
    
def main():
    dataset = SentevalSubj()
    dataset.run()

if __name__ == "__main__":
    main()