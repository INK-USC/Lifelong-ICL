import openai
import retrying
import os
import tiktoken

from tqdm import tqdm

from .model import Model
from .utils import load_api_key

class OpenAI(Model):
    def set_up_model(self):
        home_directory = os.path.expanduser("~")
        api_key = load_api_key(os.path.join(home_directory, "openai_api_key.txt"))
        self.model = openai.OpenAI(api_key=api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.args.model)

    @retrying.retry(
        stop_max_attempt_number=100, # Maximum number of attempts
        wait_exponential_multiplier=1000, wait_exponential_max=10000, 
        # Wait 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    def call_openai_api(self, prompt, mode):
        if mode == "greedy": 
            additional_args = {"temperature": 0.0, "stop": ["\n"], "max_tokens": 5}
        elif mode == "rank":
            additional_args = {"logprobs": True, "top_logprobs": 20, "max_tokens": 1}

        output = self.model.chat.completions.create(
            model=self.args.model, # gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            **additional_args,
        )
        return output

    def run_greedy(self, prefix, prompts):
        actual_prompts = [prefix + "\n" + prompt for prompt in prompts]
        predictions = []
        for prompt in tqdm(actual_prompts):
            output = self.call_openai_api(prompt, mode="greedy")
            predictions.append({"prediction": output.choices[0].message.strip()})
        return predictions

    def run_rank(self, prefix, prompts, options):
        actual_prompts = [prefix + "\n" + prompt for prompt in prompts]
        outputs = []
        for prompt in tqdm(actual_prompts):
            raw_output = self.call_openai_api(prompt, mode="rank")
            outputs.append(raw_output)

        option_tokens = self.get_label_tokens(options)
        predictions = self.rank_postprocess(outputs, option_tokens)

        return predictions
    
    def rank_postprocess(self, outputs, option_tokens):
        predictions = []

        for output in outputs:

            logprobs = output.choices[0].logprobs.content[0].top_logprobs

            if len(logprobs) == 0:
                pred_obj = {"prediction": "NO_PREDICTION"}
                predictions.append(pred_obj)
                continue

            output_mapped = {item.token: item.logprob for item in logprobs}

            option_logprobs = {"NO_PREDICTION": -1e9}
            for option_name, option_token in option_tokens.items():
                option_logprobs[option_name] = output_mapped.get(option_token[0], -1e10)

            # select the key with the maximum value in the dictionary
            predicted_label = max(option_logprobs, key=lambda k: option_logprobs[k])
            option_logprobs.pop("NO_PREDICTION")

            pred_obj = {
                "generation": output.choices[0].message.content,
                "prediction": predicted_label.strip(),
                "logprobs": option_logprobs,
            }
            predictions.append(pred_obj)
            
        return predictions
    
    def get_label_tokens(self, options):
        option_tokens = {}
        for option in options:
            # string -> token ids
            token_ids = self.tokenizer.encode(option)
            # token ids -> subwords # python uses utf-8 encoding by default
            token_subwords = [self.tokenizer.decode_single_token_bytes(token).decode("utf-8") for token in token_ids] 
            option_tokens[option] = token_subwords

            # string -> token ids
            token_ids = self.tokenizer.encode(" " + option)
            # token ids -> subwords # python uses utf-8 encoding by default
            token_subwords = [self.tokenizer.decode_single_token_bytes(token).decode("utf-8") for token in token_ids] 
            option_tokens[" " + option] = token_subwords

        return option_tokens
    
    def get_n_tokens(self, sequence):
        token_ids = self.tokenizer.encode(sequence)
        return len(token_ids)