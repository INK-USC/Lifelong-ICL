from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class PoemSentiment(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "poem_sentiment"
        self.task_type = "classification"
        self.label = {
            0: "negative",
            1: "positive",
            2: "no impact"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["verse_text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("poem_sentiment", split="train")
        hf_test = load_dataset("poem_sentiment", split="test")
        return hf_train, hf_test
    
def main():
    dataset = PoemSentiment()
    dataset.run()

if __name__ == "__main__":
    main()