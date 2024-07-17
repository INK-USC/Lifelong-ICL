from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class Ethos(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "ethos"
        self.task_type = "classification"
        self.label = {
            0: 'no hate speech',
            1: 'hate speech'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_set = load_dataset("SetFit/ethos", 'binary', split="train")
        hf_train, hf_test = self.sample_train_test_set(hf_set)
        return hf_train, hf_test
    
def main():
    dataset = Ethos()
    dataset.run()

if __name__ == "__main__":
    main()