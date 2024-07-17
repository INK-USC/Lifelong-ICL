from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class EnvironmentalClaims(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "environmental_claims"
        self.task_type = "classification"
        self.label = {
            0: "no",
            1: "yes"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("climatebert/environmental_claims", split="train")
        hf_test = load_dataset("climatebert/environmental_claims", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = EnvironmentalClaims()
    dataset.run()

if __name__ == "__main__":
    main()