"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from rllm.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from rllm.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

import json 
from contextlib import redirect_stdout
from io import StringIO

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

##### extract prob from json ######

def is_numeric(value):
    try:
        float(value)
        return True
    except Exception:
        return False
    
import re
def extract_prob(model_response,cut=False):  #get last prob
    PATTERN_DUAL =  r'\{\"ANSWER\":.*?\}'
    PATTERN_SINGLE = r'\{\"PROB\":.*?\}'
    matches = re.findall(PATTERN_DUAL, model_response, re.DOTALL)
    if not matches:
        matches = re.findall(PATTERN_SINGLE, model_response, re.DOTALL)
    matched_str = None
    prob_str_value=None
    
    for match in matches:
        if match:
            matched_str = match.lower()
            if matched_str.startswith("{{") and matched_str.endswith("}}}"):
                matched_str = matched_str[1:-2]
            elif matched_str.startswith("{{") and matched_str.endswith("}}"):
                matched_str = matched_str[1:-1]
            elif matched_str.startswith("{{") and matched_str.endswith("}"):
                matched_str = matched_str[1:]
            elif matched_str.startswith("{") and matched_str.endswith("}}"):
                matched_str = matched_str[:-1]
        else:
            matched_str = None


        if matched_str:
            try:
                inner_json_obj = json.loads(matched_str)
            except json.JSONDecodeError:
                return None
            tmp_str_value = inner_json_obj.get("prob", None)
            if tmp_str_value is not None:
                prob_str_value=str(tmp_str_value)
    if matched_str is None:
        prob_str_value = None

    pred_value = prob_str_value if prob_str_value and is_numeric(prob_str_value) else None
    if pred_value and cut:
        pred_value=f"{float(pred_value):.4f}"
    return pred_value
##### extract prob from json ######

def run_response_python(text):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        #print("No code block found.\n")
        #print("input: \n"+text+"\n")
        return None
    output_buffer = StringIO()
    global_scope = {}
    local_scope = {}
    #print("code: \n"+code+"\n")
    try:
        with redirect_stdout(output_buffer):
            exec(code, global_scope, local_scope)
        
        output=output_buffer.getvalue()
    
    except Exception as e:
        #print(f"Exception {e}")
        return None
    
    #print("code output:"+output)
    return output

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response.split("<｜Assistant｜>")[1]


        
        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[-1]
        else:
            print("</think> not found \n")
            #print("model_response: "+model_response) 
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        #print("model_response"+model_response)

        model_solution= run_response_python(model_solution)
        if not model_solution:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        model_answer = extract_prob(model_solution) # modified
        #print(model_answer)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.metadata.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = extract_prob(str(truth),cut=True) #modified
            # if "\\boxed" in truth:
            #     processed_truth = extract_prob(truth)
            #     if processed_truth is not None:
            #         processed_ground_truths.append(processed_truth)
            # else:
            processed_ground_truths.append(truth)
            #print(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                # Apply tool call bonus if applicable and answer is correct
                reward = self.config.correct_reward
                if input.metadata.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus
                return RewardOutput(reward=reward, is_correct=True)


                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)



def rllm_reward_fn_calm(data_source: str, llm_solution: str, ground_truth: Union[str, List[str]], extra_info={}, **kwargs):
    """Evaluates mathematical solutions against ground truth answers.

    This function creates a reward function to evaluate mathematical solutions by comparing
    them against provided ground truth answers. It can optionally use a language model
    for more sophisticated answer validation.

    Args:
        data_source: The source/dataset the problem comes from
        llm_solution: The solution string provided by the language model to evaluate
        ground_truth: Either a single string or list of strings containing valid answers
        enable_llm: Whether to enable language model validation for complex cases (default: False)

    Returns:
        bool: True if the solution is deemed correct, False otherwise

    Example:
        >>> rllm_reward_fn_math("gsm8k", "x = 5", "5", False)
        True
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=None,
                                            problem_type=RewardType.MATH,
                                            model_response=llm_solution,
                                            metadata={"answer": ground_truth, **extra_info},
                                            data_source=data_source))
    return reward_response.reward


if __name__ == "__main__":
    print(RewardConfig)
    reward = RewardMathFn(RewardConfig)
    test_input = RewardInput(
        data_source="",
        problem=(
            "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots "
            "$r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots "
            "$r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient "
            "of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, "
            "leaving your answer in terms of $x$. (You may assume that $x$ is not equal to "
            "any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
        ),
        problem_type=RewardType.MATH,
        model_response=(
            "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."
        ),
        metadata={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"], "has_toolcall": True}
    )

    true_input="""
    <｜begin▁of▁sentence｜><｜User｜>Input Info: Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships:Ppfy has a direct effect on setv.Ppfy has a direct effect on grae.Setv has a direct effect on vdng. For those with setv being low (B=0), the probability of vdng being low (C=0) is 0.7585. For those with setv being high (B=1), the probability of vdng being low (C=0) is 0.6749. Instruction: Consider the average treatment effect (ATE) of setv on vdng. Question: If setv is changed to be low, will vdng be more likely to be low? Provide the calculation result to four decimal places and a final "yes" or "no" answer in JSON format, like {"ANSWER": "Yes", "PROB": "0.1234"}: 
  You must write a python code to solve it.Write a python program to solve this question after thinking. You should only output the program and do not explain..
**Requirements**
1. Compute the correct answer
2. Print the output to stdout following EXACTLY the given format
3. Do NOT add any extra text/messages (like "Result: ")
Let's think step by step.<｜Assistant｜><think>

Okay, so I have this problem where I need to determine if changing setv to low will make vdng more likely to be low. Let's break it down.

First, the given conditions are:
- Ppfy affects setv.
- Ppfy also affects grae.
- Setv affects vdng.

But I don't need to worry about the other variables like Ppfy or grae for this calculation. The question is specifically about the average treatment effect (ATE) of setv on vdng.

The data provided is about the probability of vdng being low (C=0) when setv is low (B=0) or high (B=1). When setv is low, P(C=0) is 0.7585, and when it's high, it's 0.6749.

Wait, the question is asking if setting setv to low will make vdng more likely to be low. So I need to compare the probabilities when setv is low versus high.

If setv is low, the probability of vdng being low is 0.7585. If setv is high, it's 0.6749. So 0.7585 is higher than 0.6749. That means setting setv to low increases the probability that vdng is low.

So the answer should be yes, because 0.7585 > 0.6749. The difference is about 0.0836, which is positive, indicating that setv being low leads to a higher probability of vdng being low.

Now, how to structure this in code. I'll read the two probabilities, compute the average treatment effect, which in this case is just the difference between the two. Then, compare if the low setv probability is higher than high. If yes, output yes with the respective probabilities.

Wait, but the question is about the average treatment effect, which is the average of the individual treatment effects across the population. But here, we're given the overall probabilities for setv being low and high. So the ATE is the difference between the two outcomes.

So the ATE is 0.7585 - 0.6749 = 0.0836. Since it's positive, changing setv to low increases the probability of vdng being low.

Therefore, the code should compute this difference and decide based on its sign.
</think>

```python
# Calculate the probability difference to determine ATE
prob_low = 0.7585
prob_high = 0.6749

ate = prob_low - prob_high

# Determine if setv being low increases the probability of vdng being low
if prob_low > prob_high:
    answer = "Yes"
else:
    answer = "No"

# Print the result in the specified format
print(f'{{"ANSWER": "{answer}", "PROB": "{ate:.4f}"}}')
```
    """
    test_input = RewardInput(
        data_source="",
        problem=(
            "Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots "
            "$r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots "
            "$r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient "
            "of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, "
            "leaving your answer in terms of $x$. (You may assume that $x$ is not equal to "
            "any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."
        ),
        problem_type=RewardType.MATH,
        model_response=(true_input),
        # model_response=(
        #     """
        #     <think>...</think>\nThe answer is \\boxed{{\"PROB\": \"0.5889\"}}
        #     ```python
        #     print('{\"PROB\": \"0.5889\"}')
        #     ```
            
        #     """#\\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.
        # ),
        metadata={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$",'{\"PROB\": \"0.5889\"}','{"ANSWER": "Yes", "PROB": "0.08362406810633793"}'], "has_toolcall": True}
    )
    output = reward(test_input) #{\"PROB\": \"0.5889\"}
    print(output)
