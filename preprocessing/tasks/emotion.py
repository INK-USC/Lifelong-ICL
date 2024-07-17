from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class Emotion(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "emotion"
        self.task_type = "classification"
        self.label = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("dair-ai/emotion", "split", split="train")
        hf_test = load_dataset("dair-ai/emotion", "split", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = Emotion()
    dataset.run()

if __name__ == "__main__":
    main()