from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class NewsData(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = "Category"
        self.identifier = "news_data"
        self.task_type = "classification"
        self.label = {
            'sports': "sports",
            'business': "business",
            'politics': "politics",
            'health': "health",
            'tech': "tech",
            'entertainment': "entertainment"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["title"] = dp["Title"]
        d["excerpt"] = dp["Excerpt"]
        d["label"] = self.label[dp["Category"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("okite97/news-data", split="train")
        hf_test = load_dataset("okite97/news-data", split="test")
        return hf_train, hf_test
    
def main():
    dataset = NewsData()
    dataset.run()

if __name__ == "__main__":
    main()