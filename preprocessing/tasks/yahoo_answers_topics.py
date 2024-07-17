from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class YahooAnswersTopics(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = 'topic'
        self.identifier = "yahoo_answers_topics"
        self.task_type = "classification"
        self.label = {
            0: "society & culture",
            1: "science & mathematics",
            2: "health",
            3: "education & reference",
            4: "computers & internet",
            5: "sports",
            6: "business & finance",
            7: "entertainment & music",
            8: "family & relationships",
            9: "politics & government"
        }


    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["title"] = dp["question_title"]
        d["content"] = dp["question_content"]
        d["label"] = self.label[dp["topic"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("yahoo_answers_topics", split="train")
        hf_test = load_dataset("yahoo_answers_topics", split="test")
        return hf_train, hf_test
    
def main():
    dataset = YahooAnswersTopics()
    dataset.run()

if __name__ == "__main__":
    main()