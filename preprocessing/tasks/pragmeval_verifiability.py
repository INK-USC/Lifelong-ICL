from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class PragmEvalVerifiability(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "pragmeval_verifiability"
        self.task_type = "classification"
        self.label = {
            0: "experiential",
            1: "unverifiable",
            2: "non-experiential"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["sentence"] = dp["sentence"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("sileod/pragmeval", "verifiability", split="train")
        hf_test = load_dataset("sileod/pragmeval", "verifiability", split="test")
        return hf_train, hf_test
    
def main():
    dataset = PragmEvalVerifiability()
    dataset.run()

if __name__ == "__main__":
    main()