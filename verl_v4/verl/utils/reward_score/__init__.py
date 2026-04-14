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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated

from . import math_verify
import ipdb
st = ipdb.set_trace

def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    reward_options=None,
    config=None,
    reward_fn_name=None,
    **reward_kwargs
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    # We need another reward function for validation
    # res = math_verify_numerical_answer_simple_relative(solution_str, ground_truth, use_relative_precision=False, max_relative_error=0.01, mode='relative', relative_to_absolute_threshold=1e-2, extra_info=None)
    # st() 
    # set(dataset['scene_type'])
    
    if data_source == "math_dapo":
        # st()
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)        
    elif reward_fn_name in ["nikash_reward"] or "Types" in data_source:
        from . import math_combined
        res = math_combined.compute_score_combined_updated(solution_str, ground_truth, extra_info["answer_type"])
    elif reward_fn_name in ["fast_reward"]:
        score = math_verify.compute_score(solution_str, ground_truth)        
        res = {
            "score": score,
            "has_boxed": 0,
            "parse_error": 0,
            "parsed_answer": "",   
            "acc": score,
            "pred": "",
        }
    
    elif reward_fn_name in ["val_reward_fn"]:
        from . import val_reward_fn
        res = val_reward_fn.compute_score(solution_str, ground_truth, data_source, extra_info, config=config)

    elif reward_fn_name in ["final_reward_function"]:
        from . import final_reward_function
        res = final_reward_function.compute_score(solution_str, ground_truth, data_source, extra_info, config=config)
        # st()


    elif reward_fn_name in ["old_reward_function"]:
        if data_source in ["math_p_numeric_translated", "math_p_reverse_translated", "math_p_numeric", "math_p_reverse"]:
            from .math_p import compute_score as train_numeric_reward
            res = train_numeric_reward(solution_str, ground_truth)
        elif data_source in ["math_p_symbolic_translated", "math_p_symbolic"]:
            from . import final_reward_function
            res = final_reward_function.compute_score(solution_str, ground_truth, data_source, extra_info, config = config)
        elif "numeric_validation" in data_source:
            from .numerical_reward import compute_score as numeric_validation_reward
            res = numeric_validation_reward(solution_str, ground_truth)
        elif "symbolic_validation" in data_source:
            from . import final_reward_function
            res = final_reward_function.compute_score(solution_str, ground_truth, data_source, extra_info, config = config)
        elif "Types_1-5_Full_Dataset" in data_source:
            from . import final_reward_function
            res = final_reward_function.compute_score(solution_str, ground_truth, data_source, extra_info, config = config)
        else:
            raise NotImplementedError(f"Reward function is not implemented for data_source = {data_source}")
    
    elif reward_fn_name == "reward_conf1":
        from . import reward_conf1
        res = reward_conf1.compute_score(solution_str, ground_truth, data_source, extra_info, config = config)
    
    elif reward_fn_name == "reward_conf2":
        from . import reward_conf2
        res = reward_conf2.compute_score(solution_str, ground_truth, data_source, extra_info, config = config)

    elif reward_fn_name == "reward_conf3":
        from . import reward_conf3
        res = reward_conf3.compute_score(solution_str, ground_truth, data_source, extra_info, config = config)
    elif reward_fn_name == "aime_reward":
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)        
    elif reward_fn_name == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function {reward_fn_name} is not implemented.")
    
    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source == "math_p" or data_source == "hcv" or data_source == "ipho" or "numeric" in data_source or "reverse" in data_source:
        if reward_options == "math-verify_custom":
            score_val, parsed_answer = basic_utils.math_verify_numerical_answer_simple_relative(solution_str, ground_truth)
            res = {
                "score": score_val,
                "has_boxed": 0,
                "parse_error": 0,
                "parsed_answer": parsed_answer,                
            }
        elif reward_options=="math_verify_numerical_reward":
            from . import math_verify_numerical_reward

            res = math_verify_numerical_reward.compute_score(solution_str, ground_truth)
        else:
            from . import numerical_reward

            res = numerical_reward.compute_score(solution_str, ground_truth) # Synthetic numerical + revere
    elif "symbolic" in data_source:
        from . import symbolic_reward
        # st()

        res = symbolic_reward.compute_score(solution_str, ground_truth, extra_info) # Synthetic symbolic
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **reward_kwargs
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
