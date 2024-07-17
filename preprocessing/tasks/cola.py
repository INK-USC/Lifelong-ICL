from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class COLA(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "cola"
        self.task_type = "classification"
        self.label = {
            0: 'unacceptable',
            1: 'acceptable'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["text"] = dp["sentence"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nyu-mll/glue", "cola", split="train")
        hf_test = load_dataset("nyu-mll/glue", "cola", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = COLA()
    dataset.run()

if __name__ == "__main__":
    main()