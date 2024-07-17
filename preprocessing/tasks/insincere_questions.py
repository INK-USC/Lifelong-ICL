from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class InsincereQuestions(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "insincere_questions"
        self.task_type = "classification"
        self.label = {
            1: "insincere question",
            0: "valid question"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("SetFit/insincere-questions", split="train")
        hf_test = load_dataset("SetFit/insincere-questions", split="test")
        return hf_train, hf_test
    
def main():
    dataset = InsincereQuestions()
    dataset.run()

if __name__ == "__main__":
    main()