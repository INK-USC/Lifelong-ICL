import time

from .model import Model
from .vllm_configs import get_model_configs

from vllm import LLM, SamplingParams

class vLLM(Model):
    def set_up_model(self):
        self.logger.info("Setting up model {} with vLLM ...".format(self.args.model))
        start = time.time()

        configs = get_model_configs(self.args.model)
        self.model = LLM(**configs)
        self.tokenizer = self.model.llm_engine.tokenizer

        model_loading_time = time.time() - start
        self.logger.info("Model loading time: {} secs".format(model_loading_time))

    def run_greedy(self, prefix, prompts):
        params = SamplingParams(temperature=0.0, stop="\n")
        actual_prompts = [prefix + self.args.demo_sep + prompt for prompt in prompts]
        outputs = self.model.generate(actual_prompts, params)
        predictions = [{"prediction": output.outputs[0].text.strip()} for output in outputs]
        return predictions

    def run_rank(self, prefix, prompts, options):
        params = SamplingParams(temperature=0.0, logprobs=100, max_tokens=1)
        actual_prompts = [prefix + self.args.demo_sep + prompt for prompt in prompts]

        outputs = self.model.generate(actual_prompts, params)
        option_tokens = self.get_label_tokens(options)
        predictions = self.rank_postprocess(outputs, option_tokens)
        return predictions
    
    def rank_postprocess(self, outputs, option_tokens):
        predictions = []

        for output in outputs:
            # In the (rare) case of exceeding context length limit or other vLLM issues
            if len(output.outputs[0].logprobs) == 0:
                pred_obj = {"prediction": "NO_PREDICTION"}
                predictions.append(pred_obj)
                continue

            # In the (rare) case where none of the label words show up in the top100 tokens,
            # we should predict this special word `NO_PREDICTION` to avoid random guessing (done by max()).
            # -1e-9 will rank higher than -1e10 in the max() operation.
            option_logprobs = {"NO_PREDICTION": -1e9}
            for option_name, option_token in option_tokens.items():
                # if the option is missing, we assign -1e10 to its logprobs
                output_logprobs = output.outputs[0].logprobs[0]
                if option_token[0] in output_logprobs:
                    option_logprobs[option_name] = output_logprobs[option_token[0]].logprob
                else:
                    option_logprobs[option_name] = -1e10

            # select the key with the maximum value in the dictionary
            predicted_label = max(option_logprobs, key=lambda k: option_logprobs[k])
            option_logprobs.pop("NO_PREDICTION")

            pred_obj = {
                "generation": output.outputs[0].text.strip(),
                "prediction": predicted_label,
                "logprobs": option_logprobs,
            }
            predictions.append(pred_obj)
            
        return predictions

    def get_label_tokens(self, options):
        # TODO: 
        # [1:] is ok for llama2 and mistral; 
        # need special care for llama3 and gemma?
        # need to double check for other models
        if self.args.model == "llama2-7b-32k": # this tokenizer does not has <bos>
            option_tokens = {option: self.tokenizer.encode(option)[0:] for option in options}
        elif "yi" in self.args.model.lower() or "phi" in self.args.model.lower(): # yi tokenizer does not have a " " in the beginning
            option_tokens = {option: self.tokenizer.encode(" "+option)[0:] for option in options}
        elif "llama3" in self.args.model.lower() or "c4ai" in self.args.model.lower(): # llama3 tokenizer does not have a " " in the beginning anymore
            option_tokens = {option: self.tokenizer.encode(" "+option)[1:] for option in options}
        else: # by default, we assume the tokenizer encodes with a <bos> and an extra " " (applicable to llama2 models and mistral models)
            option_tokens = {option: self.tokenizer.encode(option)[1:] for option in options}
        return option_tokens
    
    def get_n_tokens(self, sequence):
        # Mainly for computing the prefix length.
        token_ids = self.tokenizer.encode(sequence)
        return len(token_ids)

    def encode_text_to_tokens(self, text):
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, text):
        return self.tokenizer.tokenizer.decode(text)

    def get_max_length(self):
        configs = get_model_configs(self.args.model)
        if 'max_model_len' in configs:
            return configs['max_model_len']
        return None

    def run_nith(self, context):
        params = SamplingParams(temperature=0.0, max_tokens=50)
        outputs = self.model.generate(context, params)
        return outputs[0].outputs[0].text
