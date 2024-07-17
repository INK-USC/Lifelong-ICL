from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class AmazonMassiveScenario(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "amazon_massive_scenario"
        self.task_type = "classification"
        label_names = ['cooking', 'audio', 'news', 'lists', 'datetime', 'takeaway', 'weather', 'play', 'recommendation', 'qa', 'alarm', 'calendar', 'iot', 'social', 'music', 'email', 'general', 'transport']
        self.label = {item: item for item in label_names}
        self.label_field = "label_text"

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["text"]
        d["label"] = dp["label_text"]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("SetFit/amazon_massive_scenario_en-US", split="train")
        hf_test = load_dataset("SetFit/amazon_massive_scenario_en-US", split="test")
        return hf_train, hf_test
    
def main():
    dataset = AmazonMassiveScenario()
    dataset.run()

if __name__ == "__main__":
    main()