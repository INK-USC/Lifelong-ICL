from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class StanceFeminist(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "stance_feminist"
        self.task_type = "classification"
        self.label = {
            0: "no stance",
            1: "against",
            2: "favor",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("tweet_eval", "stance_feminist", split="train")
        hf_test = load_dataset("tweet_eval", "stance_feminist", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = StanceFeminist()
    dataset.run()

if __name__ == "__main__":
    main()