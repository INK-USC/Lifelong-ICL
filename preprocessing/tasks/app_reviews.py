from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class AppReviews(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = 'star'
        self.identifier = "app_reviews"
        self.task_type = "classification"
        self.label = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["review"]
        d["label"] = self.label[dp["star"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_set = load_dataset("app_reviews", split="train")
        hf_train, hf_test = self.sample_train_test_set(hf_set)
        return hf_train, hf_test

def main():
    dataset = AppReviews()
    dataset.run()

if __name__ == "__main__":
    main()