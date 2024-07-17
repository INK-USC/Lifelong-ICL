from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class HateSpeech18(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "hate_speech18"
        self.task_type = "classification"
        self.label = {
            1: "hate",
            0: "no hate"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_all = load_dataset("hate_speech18", split='train')
        hf_train, hf_test = self.sample_train_test_set(hf_all)
        return hf_train, hf_test
    
def main():
    dataset = HateSpeech18()
    dataset.run()

if __name__ == "__main__":
    main()