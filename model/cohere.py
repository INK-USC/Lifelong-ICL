import cohere
import retrying
import os

from tqdm import tqdm

from .model import Model
from .utils import load_api_key

class Cohere(Model):
    def set_up_model(self):
        home_directory = os.path.expanduser("~")
        api_key = load_api_key(os.path.join(home_directory, "cohere_api_key.txt"))
        self.model = cohere.Client(api_key)

    @retrying.retry(
        stop_max_attempt_number=100, # Maximum number of attempts
        wait_exponential_multiplier=1000, wait_exponential_max=10000, 
        # Wait 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    def call_cohere_api(self, prompt):
        output = self.model.chat(
            message=prompt,
            model=self.args.model, # command, command-r, command-r-plus
            temperature=0.0,
            stop_sequences=["\n"],
        )
        return output

    def run_greedy(self, prefix, prompts):
        actual_prompts = [prefix + "\n" + prompt for prompt in prompts]
        predictions = []
        for prompt in tqdm(actual_prompts):
            output = self.call_cohere_api(prompt)
            predictions.append({"prediction": output.text.strip()})
        return predictions

    def run_rank(self, prefix, prompts, options):
        raise NotImplementedError("Cohere API does not return logprobs, hence ranking-style inference is not supported.")