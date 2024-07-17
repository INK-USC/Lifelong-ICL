from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class LexicalRCRoot09(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "lexical_rc_root09"
        self.task_type = "classification"
        self.label = {
            "RANDOM": "unrelated",
            "HYPER": "hypernym",
            "COORD": "co-hyponym"
        }
        self.label_field = "relation"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["head"] = dp["head"]
        d["tail"] = dp["tail"]
        d["label"] = self.label[dp["relation"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("relbert/lexical_relation_classification", "ROOT09", split="train")
        hf_test = load_dataset("relbert/lexical_relation_classification", "ROOT09", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = LexicalRCRoot09()
    dataset.run()

if __name__ == "__main__":
    main()