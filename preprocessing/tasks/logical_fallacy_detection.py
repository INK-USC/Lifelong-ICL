from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class LogicalFallacyDetection(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "logical_fallacy_detection"
        self.task_type = "classification"
        self.label = {
            'Valid': 'valid',
            'Invalid': 'invalid'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        return d

    def _hf_extract_question(self, example):
        text = example["inputs"]
        head_1 = "Q: Do you think the following argument is 'Valid' or 'Invalid'? "
        head_2 = "Q: This AI is identifying whether statements contain fallacies. The AI responds with 'Valid' or 'Invalid' as appropriate. "
        if text.startswith(head_1):
            text = text[len(head_1):]
        elif text.startswith(head_2):
            text = text[len(head_2):]

        tail = "\n A:"
        text = text.rstrip(tail)
        return {"text": text,"label": example["targets"][0]}
    
    def _hf_process(self, dataset):
        return dataset.map(self._hf_extract_question)

    def load_hf_dataset(self):
        hf_train = load_dataset("tasksource/bigbench", "logical_fallacy_detection", split="train")
        hf_test = load_dataset("tasksource/bigbench", "logical_fallacy_detection", split="validation")
        hf_train, hf_test = self._hf_process(hf_train), self._hf_process(hf_test)
        return hf_train, hf_test
    
def main():
    dataset = LogicalFallacyDetection()
    dataset.run()

if __name__ == "__main__":
    main()