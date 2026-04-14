# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

from verl.utils.reward_score.math_p_symbolic import compute_score as math_p_symbolic_compute_score
from verl.utils.reward_score.math_p import compute_score as math_p_compute_score
from llm.utils.math_utils import extract_answer
from llm.utils.basic_utils import relative_precision
import llm.utils.basic_utils as basic_utils
import re, ipdb, itertools
st = ipdb.set_trace

TEXT_RE    = re.compile(r'\\text\{([^}]+)\}')
NUMBER_RE = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

def compute_score_combined_original(solution_str, ground_truth, answer_type) -> dict:
    """
    Old compute score code for numerical and equation answers.
    """
    if answer_type == "numerical":
        return math_p_compute_score(solution_str, ground_truth)
    elif answer_type == "equation":
        return math_p_symbolic_compute_score(solution_str, ground_truth, {})
    else:
        return math_p_symbolic_compute_score(solution_str, ground_truth, {})

def math_verify_compute_score(model_output: str, ground_truth: str, answer_type: str, timeout_score: float = 0) -> tuple[float, str]:
    """
    This function computes the score for the solution string and the ground truth,
    using the math-verify library.
    It cases on the answer type to determine which function to use.
    answer_type is the type of the answer, which can be "numerical", "letter" for
    MC questions, "equation", "single part sentence answer", or "multi-part answer"
    if a single question has multiple parts in the answer (like answer is: this
    or this), or "multi-part question and answer" if the answer and question
    have multiple parts, or "uncategorizable" otherwise.
    """
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    if "\\boxed{" not in model_output:
        model_output = "\\boxed{" + model_output + "}"
    else:
        model_output = model_output
    
    # Check if ground truth needs latex wrapper for equation type
    if answer_type == "equation" or answer_type == "numerical" or answer_type == "letter" and \
       not (ground_truth.startswith('$') and ground_truth.endswith('$')) and \
       not (ground_truth.startswith('$$') and ground_truth.endswith('$$')) and \
       not (ground_truth.startswith('\\[') and ground_truth.endswith('\\]')) and \
       not (ground_truth.startswith('\\(') and ground_truth.endswith('\\)')):
        ground_truth = '$' + ground_truth + '$'
    
    actual_ground_truth = parse(ground_truth)    
    predicted_answer = parse(model_output)

    # For numerical answer types, extract numbers from \text{} patterns and add them to the predicted answer list
    if answer_type == "numerical" and isinstance(predicted_answer, list):
        additional_predicted_answers = []
        for item in predicted_answer:
            if isinstance(item, str):
                # Look for patterns like \text{200 N}, \text{5.5 m/s}, etc.
                text_matches = TEXT_RE.findall(item)
                for text_content in text_matches:
                    if isinstance(item, str):
                        # Extract the first number from the text content
                        m = NUMBER_RE.search(text_content)
                        if not m:
                            continue
                        number_str = m.group(0)
        
                        try:
                            # Add the number as a string
                            additional_predicted_answers.append(number_str)
                        except ValueError:
                            pass
                        try:
                            # Add the number as a float/int
                            if '.' in number_str or 'e' in number_str.lower():
                                additional_predicted_answers.append(float(number_str))
                            else:
                                additional_predicted_answers.append(int(number_str))
                        except ValueError:
                            pass
                     

        # Add the additional ground truths to the list
        predicted_answer.extend(additional_predicted_answers)
    
    # For letter answers types, if the answer is a letter, then add the lowercase and uppercase versions of the letter to the ground truth
    if answer_type == "letter":
        if isinstance(predicted_answer, str):
            predicted_answer = [predicted_answer, predicted_answer.lower(), predicted_answer.upper()] if predicted_answer.isalpha() else [predicted_answer]
        elif isinstance(actual_ground_truth, list):
            # Only apply case transformations to string items that are English letters
            lowercase_items = [item.lower() for item in predicted_answer if isinstance(item, str) and item.isalpha()]
            uppercase_items = [item.upper() for item in predicted_answer if isinstance(item, str) and item.isalpha()]
            predicted_answer.extend(lowercase_items)
            predicted_answer.extend(uppercase_items)

    try:
        ret_score = basic_utils.verify(actual_ground_truth, predicted_answer, timeout_seconds=5)  # Uses math-verify library
        if not ret_score and answer_type == "numerical":
            for pred, gt in itertools.product(predicted_answer, actual_ground_truth):
                if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
                    if relative_precision(pred, gt):
                        ret_score = 1.0
                        break
                    else:
                        if basic_utils.check_float_answer(pred, gt, mode='relative'):
                            ret_score = 1.0
                            break
                else:
                    ret_score = 0.0
    except TimeoutException:
        ret_score = 0.0 # timeout_score
    except Exception:
        ret_score = 0.0

    # print(f"Answer Type: {answer_type} | Ground Truth: {ground_truth} | Parsed Ground Truth: {actual_ground_truth} | Parsed Predicted Answer: {predicted_answer} | Equal Answers: {bool(ret_score == 1)}")
    try:
        parsed_answers = str(predicted_answer)
    except:
        parsed_answers = ""
    
    assert ret_score in (0.0, 1.0, True, False, 0, 1), f"ret_score escaped as {ret_score!r}"
    return ret_score, parsed_answers

def compute_score_combined_updated(solution_str, ground_truth, answer_type) -> dict:
    """
    This function computes the score for the solution string and the ground truth,
    using the math-verify library.
    It cases on the answer type to determine which function to use.
    answer_type is the type of the answer, which can be "numerical", "letter" for
    MC questions, "equation", "single part sentence answer", or "multi-part answer"
    if a single question has multiple parts in the answer (like answer is: this
    or this), or "multi-part question and answer" if the answer and question
    have multiple parts, or "uncategorizable" otherwise.
    """
    retval = 0.
    has_boxed = 0
    parse_error = 0
    string_in_last_boxed = ""
    
    verifiable_answer_types = ["numerical", "letter", "equation"]

    try:
        if answer_type in verifiable_answer_types:
            retval, string_in_last_boxed = math_verify_compute_score(solution_str, ground_truth, answer_type, 0)
        # elif answer_type == "single part sentence answer":
        #     retval, string_in_last_boxed = math_verify_compute_score(solution_str, ground_truth, answer_type, 0)
        # elif answer_type == "multi-part answer":
        #     retval, string_in_last_boxed = math_verify_compute_score(solution_str, ground_truth, answer_type, 0)
        # elif answer_type == "multi-part question and answer":
        #     retval, string_in_last_boxed = math_verify_compute_score(solution_str, ground_truth, answer_type, 0)
        # elif answer_type == "multi-part question and answer":
        #     retval, string_in_last_boxed = math_verify_compute_score(solution_str, ground_truth, answer_type, 0)
        else: # uncategorizable
            # Old code from compute_score above
            print(f"DEBUG: compute_score_combined_updated: running on unsupported answer_type {answer_type}, since only support {verifiable_answer_types} answer_types")

            # for debugging
            boxed_match = re.findall(r'\\boxed{(.+?)}', solution_str, re.DOTALL)
            if len(boxed_match) > 0:
                has_boxed = 1
            else:
                has_boxed = 0
            
            # for reward
            boxed_original = extract_answer(solution_str, 'physics')
            string_in_last_boxed = boxed_original if isinstance(boxed_original, str) else str(boxed_original)
            correct, verify_reason = basic_utils.check_answer_heuristic(string_in_last_boxed, str(ground_truth), 2, "exact")
            if correct:
                retval = 1.
    except Exception as e:
        parse_error = 1
        print(f"compute_score_combined_updated: answer_type {answer_type} | solution_str: {solution_str} | ground_truth: {ground_truth} | Exception {e}")
    
    try:
        retval = float(bool(retval))
    except:
        retval = 0.0
        try:
            print(f"compute_score_combined_updated: retval is {retval!r}")
        except:
            pass
    
    results = {
        "score": retval,
        "has_boxed": has_boxed,
        "acc": retval,
        "pred": str(string_in_last_boxed),
        "parse_error": parse_error,
        "parsed_answer": str(string_in_last_boxed),
    }
    return results

if __name__ == "__main__":
    pass