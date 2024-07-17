from datasets import load_dataset

from task import ClassificationTask, N_SAMPLES

class WSC(ClassificationTask):

    def __init__(self):
        super().__init__()
        self.identifier = "wsc"
        self.task_type = "classification"
        self.label = {
            0: "false",
            1: "true",
        }

    def map_hf_dp_to_dict(self, dp):
        d = {}   
        # process span
        raw_text = dp["text"]
        span_dicts = [{"index":  dp["span1_index"], "text": dp["span1_text"]}, {"index": dp["span2_index"], "text": dp["span2_text"]}]
        span_dicts = sorted(span_dicts, key=lambda x: x["index"])
        words = raw_text.split()
        
        offset = 0
        for span_index in span_dicts:
            words.insert(span_index["index"] + offset, "<span>")
            words.insert(span_index["index"] + len(span_index["text"].split()) + offset + 1, "</span>")
            offset += 2

        text = " ".join(words)
        d["text"] = text
        d["span1_text"] = dp["span1_text"]
        d["span2_text"] = dp["span2_text"]
        d["label"] = self.label[dp["label"]]
        d["id"] = dp["id"]
        return d
    
    def load_hf_dataset(self):
        hf_train = load_dataset("super_glue", "wsc.fixed", split="train")
        hf_test = load_dataset("super_glue", "wsc.fixed", split="validation")
        return hf_train, hf_test
    
def main():
    dataset = WSC()
    dataset.run()

if __name__ == "__main__":
    main()