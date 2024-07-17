import openai
import retrying
import os
import tiktoken
from types import SimpleNamespace
import json

from tqdm import tqdm

from .model import Model
from .utils import load_api_key

class OpenAIBatch(Model):
    def set_up_model(self):
        home_directory = os.path.expanduser("~")
        api_key = load_api_key(os.path.join(home_directory, "openai_api_key.txt"))
        self.model = openai.OpenAI(api_key=api_key)

        if self.args.model == "gpt-4o":
            self.model_name = "gpt-4o"
        elif self.args.model == "gpt-3.5":
            self.model_name = "gpt-3.5-turbo"

        # for easy of use, put api key in .env
        self.client = openai.OpenAI(api_key=api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    def prepare_request_body(self, prompt, mode):
        if mode == "greedy": 
            additional_args = {"temperature": 0.0, "stop": ["\n"], "max_tokens": 5}
        elif mode == "rank":
            additional_args = {"logprobs": True, "top_logprobs": 20, "max_tokens": 1}

        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **additional_args,
        }
        return body

    def send_batch_request(self, file_name):
        batch_file = self.client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_job.id

    def _save_output_file(self, type, results, file_name, output_dir):
        file_path = os.path.join(output_dir, f"{type}_{file_name}")
        with open(file_path, 'w') as f:
            f.write(results)
        self.logger.info(f"Saved to {file_path}")

    def _convert_to_obj(self, data):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._convert_to_obj(data[i])
        elif isinstance(data, dict):
            for key, value in data.items():
                data[key] = self._convert_to_obj(value)
            return SimpleNamespace(**data)
        return data

    def _load_output_file(self, type, file_name, output_dir):
        content = []
        file_path = os.path.join(output_dir, f"{type}_{file_name}")
        with open(file_path, 'r') as f:
            for line in f.readlines():
                content.append(json.loads(line))
        return self._convert_to_obj(content)


    def cancel_job(self, batch_job):
        self.client.batches.cancel(batch_job["batch_job_id"])

    def retrieve_batch_request(self, batch_job_info, output_dir):
        batch_job_id = batch_job_info["batch_job_id"]
        content = None
        if batch_job_info["status"] != "saved":
            result = self.client.batches.retrieve(batch_job_id)
            
            # update status
            batch_job_info["status"] = result.status
            batch_job_info["request_counts"]["total"] = result.request_counts.total
            batch_job_info["request_counts"]["completed"] = result.request_counts.completed
            batch_job_info["request_counts"]["failed"] = result.request_counts.failed

            if result.status == "completed":
                content = self.client.files.content(result.output_file_id).content.decode('utf-8')
                self._save_output_file("raw", content, batch_job_info["file_name"], output_dir)
                content = self._load_output_file("raw", batch_job_info["file_name"], output_dir)

        return batch_job_info, content
    
    def get_predictions_greedy(self, outputs):
        predictions = [output.choices[0].message.strip() for output in outputs]
        return predictions

    def get_predictions_rank(self, outputs, options):
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