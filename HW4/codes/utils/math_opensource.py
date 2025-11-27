import json
import logging
import math
import os
import re
from pathlib import Path
from statistics import mean

import timeout_decorator
from collections import Counter
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def init_fn():
    from .math_opensource_utils.parser import parse_ground_truth, STRIP_EXCEPTIONS, extract_answer, strip_string
    from .math_opensource_utils.parser import choice_answer_clean
    from .math_opensource_utils.grader import math_equal_process

def work(args):
    i, job = args
    from .math_opensource_utils.parser import parse_ground_truth, STRIP_EXCEPTIONS, extract_answer, strip_string
    from .math_opensource_utils.parser import choice_answer_clean
    from .math_opensource_utils.grader import math_equal_process
    # total_num = len(job["gen"])

    data_name = job['task']
    job['gt_cot'], job['gt'] = parse_ground_truth(job, data_name)

    prediction = extract_answer(job['response'], data_name)
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)

    # cleaning choice results
    if job["gt"] in ["A", "B", "C", "D", "E"] and prediction not in ["A", "B", "C", "D", "E"]:
            prediction = choice_answer_clean(prediction)
    elif is_multi_choice(job["gt"]) and not is_multi_choice(prediction):
            # remove any non-choice char
            prediction = "".join(
                [c for c in prediction if c in ["A", "B", "C", "D", "E"]]
            )

    params = [(prediction, job['gt'])]
    result = math_equal_process(params[0])
        
    return i, float(result), prediction

def compute_scores(jobs, cache_path):
    with tqdm(total=len(jobs)) as pbar:
        with ProcessPool(max_workers=20, initializer=init_fn) as pool:
            future = pool.map(work, list(enumerate(jobs)), timeout=10)
            iterator = future.result()
            while True:
                try:
                    i, result, prediction = next(iterator)
                    jobs[i]['accuracy'] = result
                    jobs[i]['extracted_answer'] = prediction
                    jobs[i]['timeout_cnt'] = 0
                    pbar.update(1)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                except Exception as error:
                    print(error.traceback)
                    exit()
        for job in jobs:
            if "accuracy" not in job:
                job['accuracy'] = 0
                job['extracted_answer'] = "Timeout"
                job['timeout_cnt'] = 1
    save_cache(jobs, cache_path)
    return mean(x['accuracy'] for x in jobs)

def save_cache(jobs, cache_path):
    with open(cache_path, "w") as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + "\n")
            g.flush()
            
if __name__=="__main__":
    data = [
        {
            "id": 0,
            "answer": "0.5",
            "response": "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction. Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{422}}$.\nThe answer is \\frac{211}{422}}",
            "task": "math500box"
        },
    ]
    print(compute_scores(data, "./tmp/eval.json"))