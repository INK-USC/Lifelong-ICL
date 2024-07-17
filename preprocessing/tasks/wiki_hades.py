from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class WikiHaDes(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "wiki_hades"
        self.task_type = "classification"
        self.label = {
            1: "hallucinated",
            0: "not hallucinated",
        }
        self.label_field = "hallucination"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["replaced"]
        d["label"] = self.label[dp["hallucination"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("tasksource/wiki-hades", split="train")
        hf_test = load_dataset("tasksource/wiki-hades", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = WikiHaDes()
    dataset.run()

if __name__ == "__main__":
    main()