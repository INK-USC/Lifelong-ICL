from datasets import load_dataset
import json
from task import ClassificationTask, N_SAMPLES

class DisasterResponseMessage(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = "related"
        self.identifier = "disaster_response_message"
        self.task_type = "classification"
        self.label = {
            1: "true",
            0: "false"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["message"]
        d["label"] = self.label[dp["related"]]
        d["id"] = dp["id"]
        return d

    def _hf_process(self, dataset):
        return dataset.filter(lambda x: x['related'] in [0,1])
    
    def load_hf_dataset(self):
        hf_train = load_dataset("disaster_response_messages", split="train")
        hf_test = load_dataset("disaster_response_messages", split="test")
        hf_train, hf_test = self._hf_process(hf_train), self._hf_process(hf_test)
        return hf_train, hf_test
    
def main():
    dataset = DisasterResponseMessage()
    dataset.run()

if __name__ == "__main__":
    main()