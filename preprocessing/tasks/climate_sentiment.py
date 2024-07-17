from datasets import load_dataset
import json
from task import ClassificationTask, N_SAMPLES

class ClimateSentiment(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "climate_sentiment"
        self.task_type = "classification"
        self.label = {
            0: "risk",
            1: "neutral",
            2: "opportunity"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d

    def load_hf_dataset(self):
        hf_train = load_dataset("climatebert/climate_sentiment", split="train")
        hf_test = load_dataset("climatebert/climate_sentiment", split="test")
        return hf_train, hf_test
    
def main():
    dataset = ClimateSentiment()
    dataset.run()

if __name__ == "__main__":
    main()