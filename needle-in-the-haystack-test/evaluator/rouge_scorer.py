from .evaluator import Evaluator
from rouge_score import rouge_scorer

class RougeEvaluator(Evaluator):
    def __init__(self, answer, args, logger):
        self.answer = answer
        self.args = args
        self.logger = logger
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
    def evaluate_response(self, response: str):
        # post-process, remove '\n' which may affect score
        response = str(response).replace("\\n","\n")
        response = response.lower().replace('\n',' ').replace('.',' ').strip()
        self.answer = self.answer.lower().replace('\n',' ').replace('.',' ').strip()

        result = self.scorer.score(self.answer, response)
        if self.args.rouge == "rouge_1":
            return result["rouge1"]
        else:
            return result["rougeL"]
