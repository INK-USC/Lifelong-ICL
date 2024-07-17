from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class BragAction(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "brag_action"
        self.task_type = "classification"
        self.label = {
            0: "not bragging",
            1: "bragging",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("Blablablab/SOCKET", "bragging#brag_action", split="train")
        hf_test = load_dataset("Blablablab/SOCKET", "bragging#brag_action", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = BragAction()
    dataset.run()

if __name__ == "__main__":
    main()