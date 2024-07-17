import pandas as pd
import os
import glob
import json

"""
Most functions are same with the original NITH test.
Some are simplified and modified.
"""

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                model_to_test,
                needle,
                haystack_dir,
                context_lengths,
                document_depth_percents,
                retrieve_question,
                results_file,
                evaluator,
                final_context_length_buffer=150,
                prompt="",
                context_check_dir="/temp",
                context_check=False):
        # simplified
        self.model_to_test = model_to_test
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.context_lengths = context_lengths
        self.document_depth_percents = document_depth_percents
        self.retrieve_question = retrieve_question
        self.results_file = results_file
        self.evaluation_model = evaluator
        self.create_df()
        self.prompt = prompt

        self.final_context_length_buffer = final_context_length_buffer
        
        self.context_check_dir = context_check_dir
        self.context_check = context_check

    def run_test(self, logger):
        cnt = 0
        tot_num = len(self.context_lengths) * len(self.document_depth_percents)
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                # modify
                result = self.evaluate_and_log(context_length, depth_percent, logger)

                logger.info(f"{cnt+1}/{tot_num}, cxt_len: {context_length}, depth_pcnt: {depth_percent}, score: {result['score']}")
                self.df.loc[(len(self.df.index))] = [context_length, depth_percent, result['score'], result['real_context_length'], result['real_depth_percent'], result['model_response'].replace("\n", "\\n")]
                self.df.to_csv(self.results_file)
                cnt += 1

    def create_df(self):
        self.df = pd.DataFrame(columns=['context_length', 'depth_percent', 'score', 'real_context_length', 'real_depth_percent', 'model_response'])
        
    def evaluate_and_log(self, context_length, depth_percent, logger):

        context_dict = self.generate_context(context_length, depth_percent)

        # modify
        response = self.model_to_test.run_nith(context_dict['context'])

        score = self.evaluation_model.evaluate_response(response)
       
        results = {
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'real_context_length' : int(context_dict['real_context_length']),
            'real_depth_percent' : float(context_dict['real_depth_percent']),
            'model_response' : response,
            'score' : score,
        }

        # modify
        real_context_len = int(self.get_context_length_in_tokens(context_dict['context'])),
        logger.info(f"-- Test Summary -- ")
        logger.info(f"real context: {real_context_len}")
        logger.info(f"Context: {context_length} tokens")
        logger.info(f"Depth: {depth_percent}")
        logger.info(f"Score: {score}")
        logger.info(f"Response: {response}\n")

        # moved the saving part outside this function
        return results
    
    def result_exists(self, context_length, depth_percent):
        if not self.df[(self.df['context_length']==context_length) & (self.df['depth_percent']==depth_percent)].empty:
            return True
        return False
        
    def generate_context(self, context_length, depth_percent):
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.model_to_test.encode_text_to_tokens(self.needle)
        assert len(tokens_needle) > 0
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        # we may use prompts when evaluating model, save space for that
        context_length -= self.final_context_length_buffer

        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]


        real_depth_percent = depth_percent

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else: 
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]

            # TODO : period_tokens may be different for some models
            #        need to configure them
            name = self.model_to_test.args.model.lower()
            if name in ["llama2-7b", "llama2-7b-32k", "llama2-7b-80k"]:
                period_tokens = [869, 29889]
            elif name in ["film-7b", "llama3-8b-instruct-1048k"]:
                period_tokens = [13]
            elif name in ["mistral-7b-instruct-v0.2"]:
                period_tokens = [28723]
            else:
                period_tokens = self.model_to_test.encode_text_to_tokens(".")

            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]
            
            real_depth_percent = float(insertion_point) / float(len(tokens_new_context))

        # Convert back to a string and return it
        new_context = self.model_to_test.decode_tokens(tokens_new_context)
        new_context = self.prompt + new_context
        new_context = new_context + self.retrieve_question

        result = {
            'context': new_context,
            'real_context_length': len(self.model_to_test.encode_text_to_tokens(new_context)),
            'real_depth_percent': real_depth_percent,
        }

        assert self.needle in new_context

        if self.context_check:
            with open(os.path.join(self.context_check_dir, f'{context_length}_{depth_percent}.txt'), 'w') as f:
                f.write(new_context)

        return result

    def get_context_length_in_tokens(self, context):
        return len(self.model_to_test.encode_text_to_tokens(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context, context_length):
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if (len(tokens) > context_length):
            context = self.model_to_test.decode_tokens(tokens[:context_length])
        return context

