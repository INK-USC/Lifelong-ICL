class Model:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.model = None

    def set_up_model(self):
        raise NotImplementedError
    
    def run_rank(self, prefix, prompts, options):
        raise NotImplementedError
    
    def run_greedy(self, prefix, prompts):
        raise NotImplementedError

    def run_nith(self, context, retrieve_question):
        raise NotImplementedError

    def encode_text_to_tokens(self, text):
        raise NotImplementedError
    
    def decode_tokens(self, text):
        raise NotImplementedError

    def get_max_length(self):
        raise NotImplementedError

    def run_nith(self, context, retrieve_question):
        raise NotImplementedError