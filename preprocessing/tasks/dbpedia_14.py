from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class DBPedia14(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "dbpedia_14"
        self.task_type = "classification"
        self.label = {
            0: "company",
            1: "educational institution",
            2: 'artist',
            3: 'athlete',
            4: 'office holder',
            5: 'mean of transportation',
            6: 'building',
            7: 'natural place',
            8: 'village',
            9: 'animal',
            10: 'plant',
            11: 'album',
            12: 'film',
            13: 'written work'
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["content"] = dp["content"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        d["title"] = dp["title"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("fancyzhx/dbpedia_14", split="train")
        hf_test = load_dataset("fancyzhx/dbpedia_14", split="test")
        return hf_train, hf_test
    
def main():
    dataset = DBPedia14()
    dataset.run()

if __name__ == "__main__":
    main()