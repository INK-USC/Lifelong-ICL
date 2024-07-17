from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class ACL_ARC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "acl_arc"
        self.task_type = "classification"
        self.label = {
            0: "Background",
            1: "Uses",
            2: "Comparison",
            3: "Continuation", 
            4: "Motivation", 
            5: "Future"
        }
        self.label_field = "intent"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["input"] = dp["cleaned_cite_text"]
        d["label"] = self.label[dp["intent"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("hrithikpiyush/acl-arc", split="train")
        hf_test = load_dataset("hrithikpiyush/acl-arc", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = ACL_ARC()
    dataset.run()

if __name__ == "__main__":
    main()