from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class MetaphorBoolean(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "metaphor_boolean"
        self.task_type = "classification"
        self.label = {
            'False': 'false',
            'True': 'true'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["id"] = dp["id"]
        d["text"] = dp["text"]
        d["explanation"] = dp["explanation"]
        d["label"] = self.label[dp["label"]]
        return d

    def _hf_extract_question(self, example):
        text = example["inputs"]
        head = "The essence of the task: Given a metaphoric sentence, identify if the second sentence is the correct paraphrase of the first.\nQ: "
        tail = "\n  choice: True\n  choice: False\n A:"
        text = text[len(head): -len(tail)]
        texts = text.split(" <--> ")
        sentence = texts[0] if texts[0][-1] == '.' else texts[0] + '.'
        explanation = texts[-1] if texts[1][-1] == '.' else texts[1] + '.'
        return {"text": sentence, "explanation": explanation, "label": example["targets"][0]}
    
    def _hf_process(self, dataset):
        return dataset.map(self._hf_extract_question)

    def load_hf_dataset(self):
        hf_train = load_dataset("tasksource/bigbench", "metaphor_boolean", split="train")
        hf_test = load_dataset("tasksource/bigbench", "metaphor_boolean", split="validation")
        hf_train, hf_test = self._hf_process(hf_train), self._hf_process(hf_test)
        return hf_train, hf_test
    
def main():
    dataset = MetaphorBoolean()
    dataset.run()

if __name__ == "__main__":
    main()