from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class ClimateFever(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = 'claim_label'
        self.identifier = "climate_fever"
        self.task_type = "classification"
        self.label = {
            0: 'supported',
            2: 'not enough info',
            1: 'refuted',
            3: 'disputed'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["text"] = dp["claim"]
        d["label"] = self.label[dp["claim_label"]]
        return d
    
    def load_hf_dataset(self):
        hf_set = load_dataset("climate_fever", split="test")
        hf_train, hf_test = self.sample_train_test_set(hf_set)
        return hf_train, hf_test
    
def main():
    dataset = ClimateFever()
    dataset.run()

if __name__ == "__main__":
    main()