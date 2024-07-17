from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class Liar(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "liar"
        self.task_type = "classification"
        self.label = {
            0: "false",
            1: "half true",
            2: "mostly true",
            3: "true",
            4: "barely true",
            5: "pants fire"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["statement"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("ucsbnlp/liar", split='train')
        hf_test = load_dataset("ucsbnlp/liar", split='test')
        return hf_train, hf_test
    
def main():
    dataset = Liar()
    dataset.run()

if __name__ == "__main__":
    main()