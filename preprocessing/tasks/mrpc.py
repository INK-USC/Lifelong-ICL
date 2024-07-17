from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class MRPC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "mrpc"
        self.task_type = "classification"
        self.label = {
            0: 'not equivalent',
            1: 'equivalent'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["sentence1"] = dp["sentence1"]
        d["sentence2"] = dp["sentence2"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "mrpc", split="train")
        hf_test = load_dataset("nyu-mll/glue", "mrpc", split="test")
        return hf_train, hf_test
    
def main():
    dataset = MRPC()
    dataset.run()

if __name__ == "__main__":
    main()