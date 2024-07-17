from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class PunDetection(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "pun_detection"
        self.task_type = "classification"
        self.label = {
            0: "no",
            1: "yes",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("frostymelonade/SemEval2017-task7-pun-detection", split="test[:80%]")
        hf_test = load_dataset("frostymelonade/SemEval2017-task7-pun-detection", split="test[80%:]")
        return hf_train, hf_test
    
def main():
    dataset = PunDetection()
    dataset.run()

if __name__ == "__main__":
    main()