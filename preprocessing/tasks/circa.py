from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class CIRCA(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = 'goldstandard1'
        self.identifier = "circa"
        self.task_type = "classification"
        self.label = {
            0: 'yes',
            1: 'no',
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["answer"] = dp["answer-Y"]
        d["question"] = dp["question-X"]
        d["context"] = dp["context"]
        d["label"] = self.label[dp["goldstandard1"]]
        return d

    def load_hf_dataset(self):
        hf_set = load_dataset("circa", split="train")
        hf_train, hf_test = self.sample_train_test_set(hf_set)
        return hf_train, hf_test
    
def main():
    dataset = CIRCA()
    dataset.run()

if __name__ == "__main__":
    main()