from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class HealthFact(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "health_fact"
        self.task_type = "classification"
        self.label = {
            0: "false",
            1: "mixture",
            2: "true",
            3: "unproven"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["claim"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("health_fact", split='train')
        hf_test = load_dataset("health_fact", split='test')
        return hf_train, hf_test
    
def main():
    dataset = HealthFact()
    dataset.run()

if __name__ == "__main__":
    main()