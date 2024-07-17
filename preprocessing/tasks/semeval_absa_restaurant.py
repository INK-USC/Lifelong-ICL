from datasets import load_dataset
import json
from task import ClassificationTask, N_SAMPLES

class SemEvalAbsaRestaurant(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "semeval_absa_restaurant"
        self.task_type = "classification"
        self.label = {
            'negative': "negative",
            'positive': "positive",
            'neutral': "neutral",
            'conflict': "conflict",
        }

    def _hf_get_aspects_polarity(self, example):
        # transfer json string to string. Only taking the first term & polarity
        text = example['text']
        example = example['aspects']
        term, polarity = example["term"][0], example["polarity"][0]
        
        # term span
        span_1, span_2 = example["from"][0], example["to"][0]
        characters = list(text)
        characters.insert(span_1, "<term> ")
        characters.insert(span_2 + 1, " </term>")
        span_text = ''.join(characters)
        return {'span_text': span_text, 'term': term, 'label': polarity}


    def _hf_get_category_polarity(self, example):
        # transfer json string to string. Only taking the first category & polarity
        example = example['category']
        category, polarity = example["category"][0], example["polarity"][0]
        return {'class': category, 'label': polarity}

    def _hf_process(self, dataset):
        dataset = dataset.map(self._hf_get_aspects_polarity)
        dataset = dataset.filter(lambda x: x["term"]!="")
        return dataset

    def map_hf_dp_to_dict(self, dp):
        d = {}
        d["text"] = dp["span_text"]
        d["term"] = dp["term"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("jakartaresearch/semeval-absa", 'restaurant', split="train")
        hf_test = load_dataset("jakartaresearch/semeval-absa", 'restaurant', split="validation")
        hf_train, hf_test = self._hf_process(hf_train), self._hf_process(hf_test)
        return hf_train, hf_test
    
def main():
    dataset = SemEvalAbsaRestaurant()
    dataset.run()

if __name__ == "__main__":
    main()