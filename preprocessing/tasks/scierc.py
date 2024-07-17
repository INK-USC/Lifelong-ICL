from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class SciERC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "scierc"
        self.task_type = "classification"
        self.label = {
            'USED-FOR': 'Used For',
            'FEATURE-OF': 'Feature Of',
            'HYPONYM-OF': 'Hyponym Of',
            'COMPARE': 'Compare',
            'EVALUATE-FOR': 'Evaluate For',
            'CONJUNCTION': 'Conjunction',
            'PART-OF': 'Part Of',
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("nsusemiehl/SciERC", split="train")
        hf_test = load_dataset("nsusemiehl/SciERC", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = SciERC()
    dataset.run()

if __name__ == "__main__":
    main()