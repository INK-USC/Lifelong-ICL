from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SMSSpam(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "sms_spam"
        self.task_type = "classification"
        self.label = {
            0: 'ham',
            1: "spam",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["sms"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_set = load_dataset("sms_spam", split='train')
        hf_train, hf_test = self.sample_train_test_set(hf_set)
        return hf_train, hf_test
    
def main():
    dataset = SMSSpam()
    dataset.run()

if __name__ == "__main__":
    main()