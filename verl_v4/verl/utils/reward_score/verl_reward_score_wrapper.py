from verl_v4.verl.utils.reward_score.math_combined import compute_score_combined_updated

def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Wrapper function for VERL that extracts answer_type from extra_info
    and calls your custom reward function.
    
    Args:
        solution_str: The model's generated solution
        ground_truth: The correct answer
        extra_info: Dictionary containing additional info like answer_type
        **kwargs: Additional keyword arguments (e.g., data_source) that VERL passes
    
    Returns:
        float: The reward score (between 0.0 and 1.0)
    """
    # Extract answer_type from extra_info
    answer_type = "uncategorizable"  # default fallback
    
    if extra_info and isinstance(extra_info, dict):
        answer_type = extra_info.get("answer_type", "uncategorizable")

    # Call your custom reward function
    result = compute_score_combined_updated(solution_str, ground_truth, answer_type)
    
    # Extract the score from the result dictionary and ensure it's a float
    if isinstance(result, dict) and "score" in result:
        score = result["score"]
    else:
        score = result
    
    # Ensure the score is a float (convert boolean True/False to 1.0/0.0)
    if isinstance(score, bool):
        return float(score)
    elif isinstance(score, (int, float)):
        return float(score)
    else:
        # Fallback to 0.0 if we can't parse the score
        return 0.0