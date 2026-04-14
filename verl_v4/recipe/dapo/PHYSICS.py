import re, os
from typing import List, Any

from llm.utils.basic_utils import extract_bracket_content
from llm.utils import basic_utils

# Ensure you have math_verify installed: pip install math_verify
from math_verify import (
    parse,
    verify,
)
import pandas as pd

system_instruction = {
    "en": """Below is an open-ended problem in Physics. Please answer this problem adhering to the following rules:
1. Please use LaTeX format to represent the variables and formulas used in the solution process and results.
2. Please put the final answer(s) in \\boxed{}, note that the unit of the answer should not be included in \\boxed{}.
3. If there are multiple final answers, please seperated them by commas in \\boxed{}, e.g., \\boxed{answer 1, answer 2}.
Problem: """,
    "zh": """以下是一个开放式的物理问题。请遵守以下规则回答此问题：

1. 请使用 LaTeX 格式来表示解题过程和结果中使用的变量和公式。
2. 请将最终答案放在 \\boxed{} 中，注意答案的单位不应包含在 \\boxed{} 中。
3. 如果有多个最终答案，请在 \\boxed{} 中用逗号分隔，例如 \\boxed{答案 1, 答案 2}。
问题："""
}

def robust_reward_function(
    llm_generation: str,
    ground_truth_list: List[List[str]],
    answer_types: List[str],
) -> float:
    """
    Calculates a reward for an LLM's generation using math_verify.

    This function delegates all parsing and extraction to math_verify,
    which robustly handles various formats including boxed expressions.
    """

    """
    Example answer types and ground truths:
    Answer types: ['Equation'], example gt: [['\\boxed{\\Omega_{3}^{2}>\\frac{4 I_{1}^{\\prime} \\mu g l}{I_{3}^{2}}}']]
    Answer types: ['Numerical'], example gt: [['\\boxed{3.75 \\times 10^{4}}']]
    Answer types: ['Numerical', 'Expression', 'Interval'], example gt: [['5000 \\text{ N}'], ['T(x) = 5000 - 5x \\text{ N}, \\, 0 \\leq x \\leq 50'], ['t < 10 \\text{ s}']]
    Answer types: ['Open-end'], example gt: [['\\text{锂}']]
    Answer types: ['MCQ'], example gt: [['\\boxed{B}']]
    Answer types: ['Expression'], example gt: [['\\boxed{v_{0}^{\\prime}=\\sqrt{\\frac{2m v_{0}^{2}}{m+M}}}']]
    Answer types: ['Numerical', 'T/F'], example gt: [['g=\\frac{(\\omega C)^{2} R}{1+(\\omega C R)^{2}}'], ['b=\\frac{-\\omega C}{1+(\\omega C R)^{2}}']]
    """

    """
    How LLMs are prompted to generate llm_generation:
    "Below is an open-ended problem in Physics. Please answer this problem adhering to the following rules:\n"
        "1. Please use LaTeX format to represent the variables and formulas used in the solution process and results.\n"
        "2. Please put the final answer(s) in \\boxed{}, note that the unit of the answer should not be included in \\boxed{}.\n"
        "3. If there are multiple final answers, please seperated them by commas in \\boxed{}, e.g., \\boxed{answer 1, answer 2}.\n"
        "Problem:{{prompt}}
    """
    num_answers = len(answer_types)
    sol_content = extract_bracket_content(llm_generation)

    has_boxed = len(sol_content) > 0

    if not has_boxed:
        return {"reward": 0.0, "has_boxed": has_boxed}

    sol_content = sol_content[-1]

    llm_answers = sol_content.split(',')
    llm_answers = [parse('\\boxed{' + ans.strip() + '}') for ans in llm_answers if ans.strip()]

    matches = []
    for idx, gt in enumerate(ground_truth_list):
        for alternative in gt:
            match = False
            if "\\boxed" not in alternative and "$" not in alternative and "\\[" not in alternative:
                alternative = "\\boxed{" + alternative + "}"
            alternative = parse(alternative)
            
            for llm_answer in llm_answers:                
                if answer_types[idx] == "MCQ":
                    # Case insensitive
                    if isinstance(alternative, str): alternative = alternative.lower()
                    else: 
                        alternative[-1] = alternative[-1].lower()
                    if isinstance(llm_answer, str): llm_answer = llm_answer.lower()
                    else: 
                        llm_answer[-1] = llm_answer[-1].lower()
                
                print("Matching GT:", alternative, "with LLM answer:", llm_answer)
                
                try:
                    match = verify(alternative, llm_answer)
                except:
                    match = False
                if match: 
                    matches.append((alternative, llm_answer))
                    break
            if match: break
            
    score = len(matches) / num_answers
    return {"reward": score, "has_boxed": has_boxed}

def get_prompts(cfg):
    df = pd.read_json(cfg.data.val_files[0], lines=True)

    mechanical_questions = df[df["domain"] == "Mechanics"]

    mechanical_questions["prompts"] = mechanical_questions.apply(lambda row: [
                {
                    'role': 'user',
                    'content': system_instruction[row["language"]]+row["question"]
                }
            ], axis=1)

    return mechanical_questions

def generate_responses(cfg, llm_prompts, llm, sampling_params):
    output_texts = []
    if cfg.solve_locally:
        all_model_outputs, token_stats = basic_utils.call_model(llm, sampling_params, llm_prompts, cfg)
        for model_outputs in all_model_outputs:
            for output in model_outputs.outputs:
                output_texts.append(output.text)
    else:
        all_model_outputs, token_stats, error_generating = basic_utils.call_llm_api(cfg, cfg.model_name, llm_prompts, n=cfg.actor_rollout_ref.rollout.val_kwargs.n)
        for model_outputs in all_model_outputs:
            for output_text in model_outputs:
                output_texts.append(output_text)

    return output_texts, token_stats

def benchmark(cfg, llm, sampling_params, logger):
    mechanical_questions = get_prompts(cfg)

    output_texts, token_stats = generate_responses(cfg, mechanical_questions["prompts"].to_list(), llm, sampling_params)

    # Output_texts is a list of strings and token_stats is a dict[list] with keys: tokens/input, tokens/output, tokens/valid_end
    mechanical_questions["llm_output"] = output_texts
    for k in token_stats:
        mechanical_questions[k] = token_stats[k]

    # Evaluate
    f1 = lambda row: robust_reward_function(row["llm_output"], row["answer"], row["answer_type"])
    
    eval_results = [f1(row) for i, row in mechanical_questions.iterrows()]

    mechanical_questions["reward"] = [eval_results[i]["reward"] for i in range(len(eval_results))]
    mechanical_questions["has_boxed"] = [eval_results[i]["has_boxed"] for i in range(len(eval_results))]

    en = mechanical_questions[mechanical_questions["language"] == "en"]
    zh = mechanical_questions[mechanical_questions["language"] == "zh"]

    # Stats
    avg_reward = mechanical_questions["reward"].mean()
    frac_has_boxed = (mechanical_questions["has_boxed"] == True).mean()

    avg_reward_en = 0 if len(en)==0 else en["reward"].mean()
    avg_reward_zh = 0 if len(zh)==0 else zh["reward"].mean()
    frac_has_boxed_en = 0 if len(en)==0 else (en["has_boxed"] == True).mean()
    frac_has_boxed_zh = 0 if len(zh)==0 else (zh["has_boxed"] == True).mean()

    avg_input_length = mechanical_questions["tokens/input"].mean()
    avg_response_length = mechanical_questions["tokens/output"].mean()
    valid_end_rate = len(mechanical_questions[mechanical_questions["tokens/valid_end"]]) / len(mechanical_questions)

    avg_input_length_en = 0 if len(en)==0 else en["tokens/input"].mean()
    avg_input_length_zh = 0 if len(zh)==0 else zh["tokens/input"].mean()
    avg_response_length_en = 0 if len(en)==0 else en["tokens/output"].mean()
    avg_response_length_zh = 0 if len(zh)==0 else zh["tokens/output"].mean()
    valid_end_rate_en = 0 if len(en)==0 else len(en[en["tokens/valid_end"]]) / len(en)
    valid_end_rate_zh = 0 if len(zh)==0 else len(zh[zh["tokens/valid_end"]]) / len(zh)

    answer_types = {}
    for i, row in mechanical_questions.iterrows():
        for atype in row["answer_type"]:
            if atype not in answer_types:
                answer_types[atype] = 1
            answer_types[atype] += 1

    answer_types_en = {}
    for i, row in en.iterrows():
        for atype in row["answer_type"]:
            if atype not in answer_types_en:
                answer_types_en[atype] = 1
            answer_types_en[atype] += 1 
    answer_types_zh = {}
    for i, row in zh.iterrows():
        for atype in row["answer_type"]:
            if atype not in answer_types_zh:
                answer_types_zh[atype] = 1
            answer_types_zh[atype] += 1

    languages = {"en" : len(en), "zh": len(zh)}

    # Print stuff
    print(f"Avg input length: {avg_input_length}")
    print(f"Avg response length: {avg_response_length}")
    print(f"Valid end rate: {valid_end_rate}")
    print(f"Avg reward: {avg_reward}")
    print(f"Frac has boxed: {frac_has_boxed}")
    print(f"Dataset stats:")
    print("Answer types:", answer_types)
    print("Languages:", languages)
    print(f"\nEnglish stats (N={len(en)}):")
    print(f"Avg input length: {avg_input_length_en}")
    print(f"Avg response length: {avg_response_length_en}")
    print(f"Valid end rate: {valid_end_rate_en}")
    print(f"Avg reward: {avg_reward_en}")
    print(f"Frac has boxed: {frac_has_boxed_en}")
    print(f"\nChinese stats (N={len(zh)}):")
    print(f"Avg input length: {avg_input_length_zh}")
    print(f"Avg response length: {avg_response_length_zh}")
    print(f"Valid end rate: {valid_end_rate_zh}")
    print(f"Avg reward: {avg_reward_zh}")
    print(f"Frac has boxed: {frac_has_boxed_zh}")

    # Log stuff
    logger.log({f'tokens/input_length': avg_input_length, f'tokens/response_length': avg_response_length, f'tokens/valid_end': valid_end_rate}, step=0)
    logger.log({'dataset_stats/answer_types': answer_types}, step=0)
    logger.log({'reward/avg_reward': avg_reward, 'reward/frac_has_boxed': frac_has_boxed}, step=0)
    logger.log({'en/tokens/input_length': avg_input_length_en, 'zh/tokens/input_length': avg_input_length_zh}, step=0)
    logger.log({'en/tokens/response_length': avg_response_length_en, 'zh/tokens/response_length': avg_response_length_zh}, step=0)
    logger.log({'en/tokens/valid_end': valid_end_rate_en, 'zh/tokens/valid_end': valid_end_rate_zh}, step=0)
    logger.log({'en/reward/avg_reward': avg_reward_en, 'en/reward/frac_has_boxed': frac_has_boxed_en}, step=0)
    logger.log({'zh/reward/avg_reward': avg_reward_zh, 'zh/reward/frac_has_boxed': frac_has_boxed_zh}, step=0)
    logger.log({'en/dataset_stats/answer_types': answer_types_en}, step=0)
    logger.log({'zh/dataset_stats/answer_types': answer_types_zh}, step=0)


    # Store responses and results in generations
    model_name = cfg.model_name.replace("/", "_")
    os.makedirs(f"evals/PHYSICS/generations/{model_name}", exist_ok=True)
    mechanical_questions.to_json(f"evals/PHYSICS/generations/{model_name}/results.jsonl", lines=True, orient="records")

    logger.finish()

def test():
    # --- Example 1: Equation ---
    gt_equation = [['\\boxed{\\Omega_{3}^{2}>\\frac{4 I_{1}^{\\prime} \\mu g l}{I_{3}^{2}}}']]
    llm_output_eq_correct = "After careful calculation, we find that the condition is \\boxed{\\Omega_{3}^{2} > \\frac{4 g l \\mu I_{1}^{\\prime}}{I_{3}^{2}}}" # Note the reordered variables
    llm_output_eq_wrong = "The result is \\boxed{\\Omega_{3}^{2} < \\frac{4 I_{1}^{\\prime} \\mu g l}{I_{3}^{2}}}"
    print(f"Equation Correct: {robust_reward_function(llm_output_eq_correct, gt_equation, ['Equation'])}")
    print(f"Equation Wrong:   {robust_reward_function(llm_output_eq_wrong, gt_equation, ['Equation'])}")
    # Expected output: 1.0, 0.0

    # --- Example 2: Numerical ---
    gt_numerical = [['\\boxed{3.75 \\times 10^{4}}']]
    llm_output_num_correct = "The final force is \\boxed{37500}."
    llm_output_num_wrong = "The final force is \\boxed{3.75e3}."
    print(f"\nNumerical Correct: {robust_reward_function(llm_output_num_correct, gt_numerical, ['Numerical'])}")
    print(f"Numerical Wrong:   {robust_reward_function(llm_output_num_wrong, gt_numerical, ['Numerical'])}")
    # Expected output: 1.0, 0.0

    # --- Example 3: Multiple Answers ---
    gt_multi = [['5000'], ['T(x) = 5000 - 5x'], ['t < 10']]
    llm_output_multi_correct = "Therefore, the time is \\boxed{t<10, 5000, 5 \\times (1000-x)}."
    llm_output_multi_partial = "Therefore, the time is \\boxed{t<10, 5 \\times (1000-x)}."
    llm_output_multi_wrong = "Therefore, the time is \\boxed{t<10, 401, 5 * (1000-x)}."
    print(f"\nMulti-Answer Correct: {robust_reward_function(llm_output_multi_correct, gt_multi, ['Numerical', 'Expression', 'Interval'])}")
    print(f"Multi-Answer Partial: {robust_reward_function(llm_output_multi_partial, gt_multi, ['Numerical', 'Expression', 'Interval'])}")
    print(f"Multi-Answer Wrong: {robust_reward_function(llm_output_multi_wrong, gt_multi, ['Numerical', 'Expression', 'Interval'])}")
    # Expected output: 1.0, 0.666, 0.666

    # --- Example 4: MCQ ---
    gt_mcq = [['\\boxed{B}']]
    llm_output_mcq_correct = "The obvious choice is B. So the answer is \\boxed{ B }."
    llm_output_mcq_correct2 = "The obvious choice is B. So the answer is \\boxed{ b}."
    llm_output_mcq_wrong = "The obvious choice is B. So the answer is \\boxed{D}."
    llm_output_mcq_wrong2 = "The obvious choice is B. So the answer is \\boxed{BD}."
    print(f"\nMCQ Correct: {robust_reward_function(llm_output_mcq_correct, gt_mcq, ['MCQ'])}")
    print(f"MCQ Correct: {robust_reward_function(llm_output_mcq_correct2, gt_mcq, ['MCQ'])}")
    print(f"MCQ Wrong:   {robust_reward_function(llm_output_mcq_wrong, gt_mcq, ['MCQ'])}")
    print(f"MCQ Wrong:   {robust_reward_function(llm_output_mcq_wrong2, gt_mcq, ['MCQ'])}")
    # Expected output: 1.0, 1.0, 0.0, 0.0

def test_strict():
    """
    More exhaustive tests for the reward function covering many edge cases.
    Prints PASS/FAIL for each case and the returned reward.
    """
    def run_case(name, llm_output, gt, types, expected, tol=1e-6):
        reward = robust_reward_function(llm_output, gt, types)
        reward = reward["reward"]
        ok = abs(reward - expected) <= tol
        status = "PASS" if ok else "FAIL"
        print(f"{name}: reward={reward:.6f} expected={expected:.6f} => {status}")
        return ok

    all_ok = True

    # Equation tests (reordered tokens should match)
    gt_eq = [['\\boxed{\\Omega_{3}^{2}>\\frac{4 I_{1}^{\\prime} \\mu g l}{I_{3}^{2}}}']]
    out_eq_good = "\\boxed{\\Omega_{3}^{2} > \\frac{4 g l \\mu I_{1}^{\\prime}}{I_{3}^{2}}}"
    out_eq_bad = "\\boxed{\\Omega_{3}^{2} < \\frac{4 I_{1}^{\\prime} \\mu g l}{I_{3}^{2}}}"
    all_ok &= run_case("Equation - correct", out_eq_good, gt_eq, ['Equation'], 1.0)
    all_ok &= run_case("Equation - wrong", out_eq_bad, gt_eq, ['Equation'], 0.0)

    # Inequality flip (gold a < b, prediction b > a should match)
    gt_ineq = [['\\boxed{x < 10}']]
    out_ineq_flip = "\\boxed{10 > x}"
    all_ok &= run_case("Inequality flip", out_ineq_flip, gt_ineq, ['Equation'], 1.0)

    # Numerical tests (scientific / integer / latex)
    gt_num = [['\\boxed{3.75 \\times 10^{4}}']]
    out_num_plain = "\\boxed{37500}"
    out_num_sci = "\\boxed{3.75E4}"
    all_ok &= run_case("Numerical - plain", out_num_plain, gt_num, ['Numerical'], 1.0)
    all_ok &= run_case("Numerical - sci", out_num_sci, gt_num, ['Numerical'], 1.0)

    # Multi-answer: all, partial, none
    gt_multi = [['5000'], ['T(x) = 5000 - 5x'], ['t < 10']]
    out_multi_all = "\\boxed{t<10, 5000, 5 \\times (1000-x)}"
    out_multi_partial = "\\boxed{t<10, 5 \\times (1000-x)}"
    out_multi_none = "\\boxed{t<10, 401, 5 * (100-x)}"
    all_ok &= run_case("Multi - all", out_multi_all, gt_multi, ['Numerical','Expression','Interval'], 1.0)
    all_ok &= run_case("Multi - partial", out_multi_partial, gt_multi, ['Numerical','Expression','Interval'], 2.0/3.0)
    # 'none' should at least match the t<10 part -> partial credit 1/3
    all_ok &= run_case("Multi - none (one match)", out_multi_none, gt_multi, ['Numerical','Expression','Interval'], 1.0/3.0)

    # RHS-only: gold has equation, pred only RHS
    gt_rhs = [['T(x) = 5000 - 5x']]
    out_rhs_only = "\\boxed{5000 - 5x}"
    all_ok &= run_case("RHS-only match", out_rhs_only, gt_rhs, ['Expression'], 1.0)

    # Expression equivalence
    gt_expr = [['\\boxed{v_{0}^{\\prime}=\\sqrt{\\frac{2 m v_{0}^{2}}{m+M}}}']]
    out_expr_equiv = "\\boxed{v_{0}^{\\prime} = (2*m*v_0^2/(m+M))^0.5}"
    all_ok &= run_case("Expression equivalence", out_expr_equiv, gt_expr, ['Expression'], 1.0)

    # Interval / set handling
    gt_interval = [['\\boxed{0 < t < 10}']]
    out_interval_ineq = "\\boxed{(t > 0) \\wedge (t < 10)}"
    all_ok &= run_case("Interval vs inequalities", out_interval_ineq, gt_interval, ['Interval'], 1.0)

    # # MCQ robustness
    # gt_mcq = [['\\boxed{B}']]
    # out_mcq_good = "\\boxed{ B }"
    # out_mcq_lower = "\\boxed{ b }"
    # out_mcq_multi = "\\boxed{BD}"
    # out_mcq_all = "\\boxed{ABCD}"
    # all_ok &= run_case("MCQ - good", out_mcq_good, gt_mcq, ['MCQ'], 1.0)
    # all_ok &= run_case("MCQ - lower", out_mcq_lower, gt_mcq, ['MCQ'], 1.0)
    # all_ok &= run_case("MCQ - multi (should fail)", out_mcq_multi, gt_mcq, ['MCQ'], 0.0)
    # all_ok &= run_case("MCQ - all (should fail)", out_mcq_all, gt_mcq, ['MCQ'], 0.0)

    # Malformed LaTeX or spacing variants (best-effort)
    gt_mal = [['\\boxed{\\frac{1}{2}}']]
    out_malformed = "\\boxed{1/2}"
    all_ok &= run_case("Malformed LaTeX recovery", out_malformed, gt_mal, ['Numerical','Expression'], 1.0)

    print('\nSummary: ' + ('ALL PASS' if all_ok else 'SOME FAIL'))

if __name__ == "__main__":
    test()
    test_strict()