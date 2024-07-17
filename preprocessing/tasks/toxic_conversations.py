from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class ToxicConversations(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "toxic_conversations"
        self.task_type = "classification"
        self.label = {
            1: "toxic",
            0: "not toxic"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("SetFit/toxic_conversations", split="train")
        hf_test = load_dataset("SetFit/toxic_conversations", split="test")
        return hf_train, hf_test
    
def main():
    dataset = ToxicConversations()
    dataset.run()

if __name__ == "__main__":
    main()