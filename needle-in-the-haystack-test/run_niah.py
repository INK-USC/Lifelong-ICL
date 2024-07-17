from .LLMNeedleHaystack import LLMNeedleHaystackTester
from .evaluator.hard_scorer import HardcheckEvaluator
from .evaluator.rouge_scorer import RougeEvaluator
import os
import numpy as np
import json
import sys

def run_niah(args, logger):

    # set model
    model = args.model_to_evaluate

    # set context
    with open(args.prompt_file, 'r') as f:
        question_set = json.load(f)

    head_prompt = question_set["head_prompt"]
    needle = question_set["needle"]
    question = question_set["question"]
    answer = question_set["answer"]

    if "passkey" in args.prompt_file:
        answer = answer.replace("<passkey>", args.pass_key)
        needle = needle.replace("<passkey>", args.pass_key)

    # set scorer
    if args.scorer == "hard_check":
        scorer = HardcheckEvaluator(answer, args, logger)
    elif args.scorer == "rouge":
        scorer = RougeEvaluator(answer, args, logger)
    else:
        raise NotImplementedError

    # set context_len
    context_lengths = np.round(np.linspace(args.min_length, args.max_length, num=args.test_length_n, endpoint=True)).astype(int)
    print(context_lengths)

    document_depth_percents = np.round(np.linspace(0, 100, num=args.depth_pcnt_n, endpoint=True)).astype(int)
    print(document_depth_percents)
    
    results_file = os.path.join(args.output_dir, f'{args.scorer}_results.csv')
    
    context_check_dir = os.path.join(args.output_dir, "context")
    if not os.path.exists(context_check_dir):
        os.makedirs(context_check_dir, exist_ok=True)

    final_context_length_buffer = 200

    test_runner = LLMNeedleHaystackTester(
        model_to_test=model,
        needle=needle,
        haystack_dir=args.haystack_dir,
        context_lengths=context_lengths,
        document_depth_percents=document_depth_percents,
        retrieve_question=question,
        results_file=results_file,
        evaluator=scorer,
        final_context_length_buffer=final_context_length_buffer,
        prompt=head_prompt,
        context_check=args.context_check,
        context_check_dir=context_check_dir,
    )
 
    logger.info(f"prompt_file:{args.prompt_name}")
    logger.info(f"head_prompt: {head_prompt}")
    logger.info(f"needle: {needle}")
    logger.info(f"question: {question}")
    logger.info(f"answer: {answer}")
    logger.info(f"context_length: {','.join([str(cl) for cl in context_lengths])}")
    logger.info(f"depth_percents: {','.join([str(dp) for dp in document_depth_percents])}")

    # run
    test_runner.run_test(logger)
        


