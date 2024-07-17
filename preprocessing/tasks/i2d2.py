from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class I2D2(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "i2d2"
        self.task_type = "classification"
        self.label = {
            0: "false",
            1: "true"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["sentence1"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("tasksource/I2D2", split="train")
        hf_test = load_dataset("tasksource/I2D2", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = I2D2()
    dataset.run()

if __name__ == "__main__":
    main()