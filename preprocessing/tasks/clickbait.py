from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class Clickbait(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "clickbait"
        self.task_type = "classification"
        self.label = {
            0: "no",
            1: "yes"
        }
        self.label_field = "clickbait"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["title"]
        d["label"] = self.label[dp["clickbait"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("marksverdhei/clickbait_title_classification", split="train[:80%]")
        hf_test = load_dataset("marksverdhei/clickbait_title_classification", split="train[80%:]")
        return hf_train, hf_test
    
def main():
    dataset = Clickbait()
    dataset.run()

if __name__ == "__main__":
    main()