from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class StudentQuestionCategories(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "student_question_categories"
        self.task_type = "classification"
        self.label = {
            0: "biology",
            1: "chemistry",
            2: 'maths',
            3: 'physics'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("SetFit/student-question-categories", split="train")
        hf_test = load_dataset("SetFit/student-question-categories", split="test")
        return hf_train, hf_test
    
def main():
    dataset = StudentQuestionCategories()
    dataset.run()

if __name__ == "__main__":
    main()