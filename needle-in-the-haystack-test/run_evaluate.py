import pandas as pd
import json
import os
from .evaluator.hard_scorer import HardcheckEvaluator
from .evaluator.rouge_scorer import RougeEvaluator


def run_evaluate(args, scorer, logger):
    result_file = os.path.join(args.output_dir, f'{args.scorer}_results.csv')

    df = pd.read_csv(result_file, index_col=0)

    with open(args.prompt_file, 'r') as f:
        question_dict = json.load(f)
    
    answer = question_dict["answer"]
    if "passkey" in args.prompt_name:
        answer = answer.replace("<passkey>", args.pass_key)

    if scorer == "hard_check":
        evaluator = HardcheckEvaluator(answer, args, logger)
    elif scorer == "rouge":
        evaluator = RougeEvaluator(answer, args, logger)
    else:
        raise NotImplementedError

    df['new_score'] = df['model_response'].apply(evaluator.evaluate_response)
    new_result_file = os.path.join(args.output_dir, f'{scorer}_results.csv')
    df.to_csv(new_result_file)


    

    
    
