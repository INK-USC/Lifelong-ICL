from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class AGNews(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "ag_news"
        self.task_type = "classification"
        self.label = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Technology",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("ag_news", split="train[:10%]")
        hf_test = load_dataset("ag_news", split="test")
        return hf_train, hf_test

def main():
    dataset = AGNews()
    dataset.run()

if __name__ == "__main__":
    main()