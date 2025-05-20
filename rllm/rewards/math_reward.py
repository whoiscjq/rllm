"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from rllm.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from rllm.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

from rllm.system_prompts import ORM_PROMPT
from rllm.utils import call_gemini_llm, call_oai_rm_llm
import json 

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
        pred_value=f"{float(pred_value):.4g}"
    return pred_value
##### extract prob from json ######

def check_python(text):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return True
    else:
        return False

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
        model_response = input.model_response
        calc_mode=True


        
        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        if calc_mode:
            if not check_python(model_solution):
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        model_answer = extract_prob(model_solution) # modified
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
            truth = extract_prob(str(truth)) #modified
            if "\\boxed" in truth:
                processed_truth = extract_prob(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
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

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            
            for ground_truth in processed_ground_truths:
                
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)



def rllm_reward_fn_math(data_source: str, llm_solution: str, ground_truth: Union[str, List[str]], extra_info={}, **kwargs):
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
            "<think>...</think>\nThe answer is \\boxed{{\"PROB\": \"0.5889\"}}"#\\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.
        ),
        metadata={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$","{\"PROB\": \"0.5889\"}"], "has_toolcall": True}
    )
    output = reward(test_input) #{\"PROB\": \"0.5889\"}
    print(output)
