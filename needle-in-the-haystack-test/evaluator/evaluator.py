class Evaluator:
    def __init__(self, answer, args, logger):
        self.answer = answer
        self.args = args
        self.logger = logger
        self.model = None

    def evaluate_response(self, response: str):
        raise NotImplementedError