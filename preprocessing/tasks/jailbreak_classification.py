from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class JailbreakClassification(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = 'type'
        self.identifier = "jailbreak_classification"
        self.task_type = "classification"
        self.label = {
            'benign': 'benign',
            'jailbreak': "jailbreak",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["prompt"]
        d["label"] = dp["type"]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("jackhhao/jailbreak-classification", split='train')
        hf_test = load_dataset("jackhhao/jailbreak-classification", split='test')
        return hf_train, hf_test
    
def main():
    dataset = JailbreakClassification()
    dataset.run()

if __name__ == "__main__":
    main()