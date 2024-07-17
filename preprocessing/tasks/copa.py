from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class COPA(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = 'question'
        self.identifier = "copa"
        self.task_type = "classification"
        self.label = {
            'cause': 'cause',
            'effect': 'effect'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        choices = [dp["choice1"], dp["choice2"]]
        d["answer"] = choices[dp["label"]]

        d["premise"] = dp["premise"]
        d["label"] = dp["question"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("super_glue", "copa", split="train")
        hf_test = load_dataset("super_glue", "copa", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = COPA()
    dataset.run()

if __name__ == "__main__":
    main()