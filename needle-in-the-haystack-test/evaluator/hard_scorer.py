from .evaluator import Evaluator

class HardcheckEvaluator(Evaluator):
    def __init__(self, answer, args, logger):
        self.answer = answer
        self.args = args
        self.logger = logger
        
    def evaluate_response(self, response: str):
        # post-process
        response = str(response).replace("\\n","\n")
        response = response.lower().replace('\n',' ').replace('.',' ').strip()
        self.answer = self.answer.lower().replace('\n',' ').replace('.',' ').strip()

        self.logger.info(f"scored-answer:{self.answer}")
        self.logger.info(f"scored-response:{response}")

        score_contain = self.evaluate_contain(response)
        score_match = self.evaluate_exact_match(response)
        score = (score_match + score_contain) / 2
        return score

    def evaluate_contain(self, response: str):
        return 1 if self.answer in response else 0
    
    def evaluate_exact_match(self, response: str):
        accept = [self.answer]
        return 1 if response in accept else 0