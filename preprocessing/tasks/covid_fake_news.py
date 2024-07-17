from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class COVIDFakeNews(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "covid_fake_news"
        self.task_type = "classification"
        self.label = {
            "real": "real",
            "fake": "fake"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["tweet"] = dp["tweet"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nanyy1025/covid_fake_news", split="train")
        hf_test = load_dataset("nanyy1025/covid_fake_news", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = COVIDFakeNews()
    dataset.run()

if __name__ == "__main__":
    main()