from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from task import ClassificationTask, N_SAMPLES

class FEVER(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "fever"
        self.task_type = "classification"
        self.label = {
            "SUPPORTS": 'supported',
            "NOT ENOUGH INFO": 'not enough info',
            "REFUTES": 'refuted'
        }

    def _clean_dataset(self, dataset):
        df = pd.DataFrame(dataset)
        cleaned_df = df.drop_duplicates(subset=['claim', 'label'])
        return Dataset.from_pandas(cleaned_df)

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["text"] = dp["claim"]
        d["label"] = self.label[dp["label"]]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("fever", 'v1.0', split="train", trust_remote_code=True)
        hf_test = load_dataset("fever", 'v1.0', split="labelled_dev", trust_remote_code=True)
        # remove duplicated examples
        hf_train = self._clean_dataset(hf_train)
        hf_test = self._clean_dataset(hf_test)
        return hf_train, hf_test
    
def main():
    dataset = FEVER()
    dataset.run()

if __name__ == "__main__":
    main()