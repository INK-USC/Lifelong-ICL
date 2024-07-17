from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SaraEntailment(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "sara_entailment"
        self.task_type = "classification"
        self.label = {
            "Entailment": "entailment",
            "Contradiction": "contradiction",
        }
        self.label_field = "answer"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["question"] = dp["question"].strip()
        d["statute"] = dp["statute"].strip()
        d["description"] = dp["description"].strip()
        d["label"] = self.label[dp["answer"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        # legalbench predefines the few-shot examples; 
        # but we need many samples of them and we need to change the number of shots sometimes.
        # We use these datasets for "inference-only" evaluation, we don't use it for training;
        # Hence, we use the "test" split in the legalbench and create our train-test splits.
        full_dataset = load_dataset("nguha/legalbench", "sara_entailment", split="test").shuffle(seed=42)
        splits = full_dataset.train_test_split(test_size=0.4)
        hf_train = splits["train"]
        hf_test = splits["test"]
        return hf_train, hf_test
    
def main():
    dataset = SaraEntailment()
    dataset.run()

if __name__ == "__main__":
    main()