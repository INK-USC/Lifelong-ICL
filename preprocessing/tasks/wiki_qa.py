from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class WikiQA(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "wiki_qa"
        self.task_type = "classification"
        self.label = {
            0: "no",
            1: "yes"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["question"] = dp["question"]
        d["answer"] = dp["answer"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("wiki_qa", split="train")
        hf_test = load_dataset("wiki_qa", split="test")
        return hf_train, hf_test
    
def main():
    dataset = WikiQA()
    dataset.run()

if __name__ == "__main__":
    main()