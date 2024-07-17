from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class TCFDRecommendations(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "tcfd_recommendations"
        self.task_type = "classification"
        self.label = {
            0: "none",
            1: "metrics",
            2: "strategy",
            3: "risk",
            4: "governance",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("climatebert/tcfd_recommendations", split="train")
        hf_test = load_dataset("climatebert/tcfd_recommendations", split="test")
        return hf_train, hf_test
    
def main():
    dataset = TCFDRecommendations()
    dataset.run()

if __name__ == "__main__":
    main()