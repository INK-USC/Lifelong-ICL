from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class ProsocialDialog(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.label_field = "safety_label"
        self.identifier = "prosocial_dialog"
        self.task_type = "classification"
        self.label = {
            "__needs_caution__": "caution needed",
            "__casual__" : "casual",
            "__needs_intervention__": "intervention needed",
            "__probably_needs_caution__": "probably caution needed",
            "__possibly_needs_caution__": "possibly caution needed",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["context"]
        d["label"] = self.label[dp["safety_label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("allenai/prosocial-dialog", split="train")
        hf_test = load_dataset("allenai/prosocial-dialog", split="test")
        return hf_train, hf_test
    
def main():
    dataset = ProsocialDialog()
    dataset.run()

if __name__ == "__main__":
    main()