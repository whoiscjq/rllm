"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset

PN_shots= [{
        "Reason": {
            "Step 1": "Calculate P(B=0|do(A=0))\nP(B=0|do(A=0))=P(B=0|A=0)=0.2544",
            "Step 2": "Upper bound of PN: min{1, [P(B=0)|do(A=0)-P(A=0,B=0)]/P(A=1,B=1)}\n=min{1, (0.2544-0.0814)/0.6525}\n=min{1, 0.2652}\n=0.2652"
        },
    "process":"Calculate P(B=0|do(A=0))\nP(B=0|do(A=0))=P(B=0|A=0)=0.2544. Upper bound of PN: min{1, [P(B=0)|do(A=0)-P(A=0,B=0)]/P(A=1,B=1)}\n=min{1, (0.2544-0.0814)/0.6525}\n=min{1, 0.2652}\n=0.2652 ",
        "Type": "PN",
        "gt_answer": "{\"PROB\": \"0.2652\"}",
        "idx": 22649,
        "question": "Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:Amount of sunlight exposure has a direct effect on vitamin d levels.Amount of sunlight exposure has a direct effect on bone density.Amount of sunlight exposure has a direct effect on risk of osteoporosis.Vitamin d levels has a direct effect on bone density.Bone density has a direct effect on risk of osteoporosis.\nFor those with amount of sunlight exposure being limited (A=0), the probability of vitamin d levels being low (B=0) is 0.2544. The probability of amount of sunlight exposure being limited (A=0) and vitamin d levels being low (B=0) is 0.0814. The probability of amount of sunlight exposure being abundant (A=1) and vitamin d levels being optimal (B=1) is 0.6525. \nInstruction: Consider the probability of necessity (PN) of amount of sunlight exposure on vitamin d levels.\nQuestion: Given that amount of sunlight exposure was abundant and vitamin d levels was optimal, what is the upper bound of the probability of the vitamin d levels would have been low if the amount of sunlight exposure had been limited?\nProvide the calculation result to four decimal places in JSON format, like {\"PROB\": \"0.1234\"}:"
    },
    {
        "Reason": {
            "Step 1": "Calculate P(C=1|do(A=0))\nP(C=1|do(A=0))=P(C=1|A=0)=0.5063",
            "Step 2": "Lower bound of PN: max{0, [P(C=1)-P(C=1|do(A=0))]/P(A=1,C=1)}\n=max{0, (0.5886-0.5063)/0.4267}\n=max{0, 0.1931}\n=0.1931"
        },
        "Type": "PN",
        "gt_answer": "{\"PROB\": \"0.1931\"}",
        "process": "Calculate P(C=1|do(A=0))\nP(C=1|do(A=0))=P(C=1|A=0)=0.5063.Lower bound of PN: max{0, [P(C=1)-P(C=1|do(A=0))]/P(A=1,C=1)}\n=max{0, (0.5886-0.5063)/0.4267}\n=max{0, 0.1931}\n=0.1931",
        "idx": 22650,
        "question": "Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:Amount of sunlight exposure has a direct effect on vitamin d levels.Amount of sunlight exposure has a direct effect on bone density.Amount of sunlight exposure has a direct effect on risk of osteoporosis.Vitamin d levels has a direct effect on bone density.Bone density has a direct effect on risk of osteoporosis.\nFor those with amount of sunlight exposure being limited (A=0), the probability of bone density being high (C=1) is 0.5063. The probability of bone density being high (C=1) is 0.5886. The probability of amount of sunlight exposure being abundant (A=1) and bone density being high (C=1) is 0.4267. \nInstruction: Consider the probability of necessity (PN) of amount of sunlight exposure on bone density.\nQuestion: Given that amount of sunlight exposure was abundant and bone density was high, what is the lower bound of the probability of the bone density would have been low if the amount of sunlight exposure had been limited?\nProvide the calculation result to four decimal places in JSON format, like {\"PROB\": \"0.1234\"}:"
    }]
PN_shot=f"Here are several examples about how to solve PN problems:\n question: {PN_shots[0]['question']}\n answer: {PN_shots[0]['process']}\n question: {PN_shots[1]['question']}\n answer: {PN_shots[1]['process']}\n"

PS_shots=[
        {
        "Reason": {
            "Step 1": "Calculate P(B=0|do(A=1))\nP(B=0|do(A=1))=P(B=0|A=1)=0.8894",
            "Step 2": "Lower bound of PS: max{0, [P(B=0)-P(B=0|do(A=1))]/P(A=0,B=0)}\n=max{0, (0.7504-0.8894)/0.4507}\n=max{0, -0.3085}\n=0.0000"
        },
        "Type": "PS",
        "gt_answer": "{\"PROB\": \"0.0000\"}",
        "idx": 140,
        "process": "<think>Calculate P(B=0|do(A=1))\nP(B=0|do(A=1))=P(B=0|A=1)=0.8894Lower bound of PS: max{0, [P(B=0)-P(B=0|do(A=1))]/P(A=0,B=0)}\n=max{0, (0.7504-0.8894)/0.4507}\n=max{0, -0.3085}\n=0.0000</think>{\"PROB\": \"0.0000\"}",
        "question": "Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:Ppfy has a direct effect on setv.Ppfy has a direct effect on grae.Setv has a direct effect on vdng.\nFor those with ppfy being high (A=1), the probability of setv being low (B=0) is 0.8894. The probability of setv being low (B=0) is 0.7504. The probability of ppfy being low (A=0) and setv being low (B=0) is 0.4507. \nInstruction: Consider the probability of sufficiency (PS) of ppfy on setv.\nQuestion: Given that ppfy was low and setv was low, what is the lower bound of the probability that setv would have been high if the ppfy had been high?\nProvide the calculation result to four decimal places in JSON format, like {\"PROB\": \"0.1234\"}:"
    },
    {
        "Reason": {
            "Step 1": "Calculate P(B=1|do(A=1))\nP(B=1|do(A=1))=P(B=1|A=1)=0.1106",
            "Step 2": "Upper bound of PS: min{1, [P(B=1)|do(A=1)-P(A=1,B=1)]/P(A=0,B=0)}\n=min{1, (0.1106-0.0373)/0.4507}\n=min{1, 0.1627}\n=0.1627"
        },
        "Type": "PS",
        "gt_answer": "{\"PROB\": \"0.1627\"}",
        "idx": 141,
        "process": "<think>Calculate P(B=1|do(A=1))\nP(B=1|do(A=1))=P(B=1|A=1)=0.1106Upper bound of PS: min{1, [P(B=1)|do(A=1)-P(A=1,B=1)]/P(A=0,B=0)}\n=min{1, (0.1106-0.0373)/0.4507}\n=min{1, 0.1627}\n=0.1627</think>{\"PROB\": \"0.1627\"}",
        "question": "Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:Ppfy has a direct effect on setv.Ppfy has a direct effect on grae.Setv has a direct effect on vdng.\nFor those with ppfy being high (A=1), the probability of setv being high (B=1) is 0.1106. The probability of ppfy being low (A=0) and setv being low (B=0) is 0.4507. The probability of ppfy being high (A=1) and setv being high (B=1) is 0.0373. \nInstruction: Consider the probability of sufficiency (PS) of ppfy on setv.\nQuestion: Given that ppfy was low and setv was low, what is the upper bound of the probability that setv would have been high if the ppfy had been high?\nProvide the calculation result to four decimal places in JSON format, like {\"PROB\": \"0.1234\"}:"
    }
]
PS_shot=f"Here are several examples about how to solve PS problems:\n question: {PS_shots[0]['question']}\n answer: {PS_shots[0]['process']}\n question: {PS_shots[1]['question']}\n answer: {PS_shots[1]['process']}\n"


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, instruction: str = None) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        mytype=example.pop('Type')
        if mytype=='PN':
            shot=PN_shot
        elif mytype=='PS':
            shot=PS_shot
        else:
            shot=''
        #oneshot='question: Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:Amount of sunlight exposure has a direct effect on vitamin d levels.Amount of sunlight exposure has a direct effect on bone density.Amount of sunlight exposure has a direct effect on risk of osteoporosis.Vitamin d levels has a direct effect on bone density.Bone density has a direct effect on risk of osteoporosis.\nFor those with amount of sunlight exposure being limited (A=0), the probability of vitamin d levels being low (B=0) is 0.2544. The probability of amount of sunlight exposure being limited (A=0) and vitamin d levels being low (B=0) is 0.0814. The probability of amount of sunlight exposure being abundant (A=1) and vitamin d levels being optimal (B=1) is 0.6525. \nInstruction: Consider the probability of necessity (PN) of amount of sunlight exposure on vitamin d levels.\nQuestion: Given that amount of sunlight exposure was abundant and vitamin d levels was optimal, what is the upper bound of the probability of the vitamin d levels would have been low if the amount of sunlight exposure had been limited?\nProvide the calculation result to four decimal places in JSON format, like {"PROB": "0.1234"}:\nanswer: Calculate P(B=0|do(A=0))\nP(B=0|do(A=0))=P(B=0|A=0)=0.2544. Upper bound of PN: min{1, [P(B=0)|do(A=0)-P(A=0,B=0)]/P(A=1,B=1)}\n=min{1, (0.2544-0.0814)/0.6525}\n=min{1, 0.2652}\n=0.2652 \n\n'
        
        if instruction is None:
            instruction="Let's think step by step."
            #instruction = "Let's think step by step and output the final answer within \\boxed{}."
        
        question = f"{shot}\n question: {question} {instruction} <think>\n"
        question =f"{question}{instruction}"
        
        answer = example.pop('gt_answer')

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default="/mnt/workspace/RL_for_Causal/rllm/data/MIX_PHRASE1",
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    # Initialize datasets
    train_datasets = [TrainDataset.Math.MIX_PHRASE1] #modified
    train_dataset = load_dataset(train_datasets[0])
    #test_datasets = [TestDataset.Math.AIME, TestDataset.Math.AMC, TestDataset.Math.MATH, TestDataset.Math.MINERVA, TestDataset.Math.OLYMPIAD_BENCH]
    test_datasets=[TestDataset.Math.MIX_PHRASE1]
    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'deepscaler_train.parquet'))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)