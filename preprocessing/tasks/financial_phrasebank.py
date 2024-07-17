from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class FinancialPhrasebank(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "financial_phrasebank"
        self.task_type = "classification"
        self.label = {
            1: "neutral",
            2: "positive",
            0: "negative"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["sentence"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_all = load_dataset("financial_phrasebank", "sentences_allagree", split='train')
        hf_train, hf_test = self.sample_train_test_set(hf_all)
        return hf_train, hf_test
    
def main():
    dataset = FinancialPhrasebank()
    dataset.run()

if __name__ == "__main__":
    main()