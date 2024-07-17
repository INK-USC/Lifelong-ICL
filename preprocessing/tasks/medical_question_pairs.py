from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class MedicalQuestionPairs(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "medical_question_pairs"
        self.task_type = "classification"
        self.label = {
            0: "not similar",
            1: "similar",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["question_1"] = dp["question_1"]
        d["question_2"] = dp["question_2"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("medical_questions_pairs", split="train[:80%]")
        hf_test = load_dataset("medical_questions_pairs", split="train[80%:]")
        return hf_train, hf_test
    
def main():
    dataset = MedicalQuestionPairs()
    dataset.run()

if __name__ == "__main__":
    main()