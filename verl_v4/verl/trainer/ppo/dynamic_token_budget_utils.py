"""
Dynamic Token Budget utilities for WANDB visualization.

This module provides utilities for parsing and visualizing Dynamic Token Budget
two-stage generation results in WANDB.
"""

import torch
from typing import Dict, List, Any, Optional, Tuple
from verl import DataProto


def parse_two_stage_response(response_text: str, tokenizer, thinking_end_tokens: List[int] = None) -> Dict[str, Any]:
    """
    Parse a two-stage response into thinking and answer components.
    
    Args:
        response_text: The complete response text
        tokenizer: The tokenizer used for encoding
        thinking_end_tokens: List of token IDs that mark the end of thinking stage
                           (e.g., [151668] for </think> in Qwen models)
    
    Returns:
        Dictionary containing:
        - thinking_text: The thinking stage text
        - answer_text: The answer stage text
        - thinking_length: Number of tokens in thinking stage
        - answer_length: Number of tokens in answer stage
        - is_two_stage: Whether this is actually a two-stage response
        - has_early_stopping: Whether early stopping was used
    """
    if thinking_end_tokens is None:
        thinking_end_tokens = [151668]  # </think> token for Qwen models
    
    # Tokenize the response to get token IDs
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    
    # Find the thinking end position
    thinking_end_pos = None
    for i, token_id in enumerate(response_ids):
        if token_id in thinking_end_tokens:
            thinking_end_pos = i + 1  # Include the </think> token
            break
    
    if thinking_end_pos is None:
        # Single stage response or thinking didn't end
        return {
            "thinking_text": response_text,
            "answer_text": "",
            "thinking_length": len(response_ids),
            "answer_length": 0,
            "is_two_stage": False,
            "has_early_stopping": False
        }
    
    # Split into thinking and answer parts
    thinking_ids = response_ids[:thinking_end_pos]
    answer_ids = response_ids[thinking_end_pos:]
    
    thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=False)
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
    
    # Check for early stopping indicators
    early_stopping_indicators = [
        "Considering the limited time by the user",
        "I have to give the solution based on the thinking directly now"
    ]
    has_early_stopping = any(indicator in thinking_text for indicator in early_stopping_indicators)
    
    return {
        "thinking_text": thinking_text,
        "answer_text": answer_text,
        "thinking_length": len(thinking_ids),
        "answer_length": len(answer_ids),
        "is_two_stage": True,
        "has_early_stopping": has_early_stopping
    }


def compute_dynamic_token_budget_metrics(batch: DataProto, tokenizer, use_dynamic_budget: bool = False) -> Dict[str, Any]:
    """
    Compute metrics specific to Dynamic Token Budget two-stage generation.
    
    Args:
        batch: The batch data containing responses
        tokenizer: The tokenizer used for decoding
        use_dynamic_budget: Whether dynamic token budget is enabled
    
    Returns:
        Dictionary of metrics including:
        - thinking_stage/length/mean, max, min: Statistics about thinking stage lengths
        - answer_stage/length/mean, max, min: Statistics about answer stage lengths
        - dynamic_budget/two_stage_ratio: Ratio of responses that are two-stage
        - dynamic_budget/early_stopping_ratio: Ratio of responses with early stopping
    """
    if not use_dynamic_budget:
        return {}
    
    responses = batch.batch["responses"]
    batch_size = responses.shape[0]
    
    thinking_lengths = []
    answer_lengths = []
    two_stage_count = 0
    early_stopping_count = 0
    
    for i in range(batch_size):
        response_ids = responses[i].tolist()
        # Remove padding tokens
        response_ids = [token_id for token_id in response_ids if token_id != tokenizer.pad_token_id]
        
        if not response_ids:
            continue
            
        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
        parsed = parse_two_stage_response(response_text, tokenizer)
        
        thinking_lengths.append(parsed["thinking_length"])
        answer_lengths.append(parsed["answer_length"])
        
        if parsed["is_two_stage"]:
            two_stage_count += 1
        if parsed["has_early_stopping"]:
            early_stopping_count += 1
    
    if not thinking_lengths:
        return {}
    
    thinking_lengths = torch.tensor(thinking_lengths, dtype=torch.float32)
    answer_lengths = torch.tensor(answer_lengths, dtype=torch.float32)
    
    metrics = {
        # Thinking stage metrics
        "thinking_stage/length/mean": torch.mean(thinking_lengths).item(),
        "thinking_stage/length/max": torch.max(thinking_lengths).item(),
        "thinking_stage/length/min": torch.min(thinking_lengths).item(),
        
        # Answer stage metrics
        "answer_stage/length/mean": torch.mean(answer_lengths).item(),
        "answer_stage/length/max": torch.max(answer_lengths).item(),
        "answer_stage/length/min": torch.min(answer_lengths).item(),
        
        # Dynamic budget specific metrics
        "dynamic_budget/two_stage_ratio": two_stage_count / batch_size,
        "dynamic_budget/early_stopping_ratio": early_stopping_count / batch_size,
    }
    
    return metrics


def create_dynamic_budget_table_data(batch: DataProto, tokenizer, vis_examples: int, 
                                    use_dynamic_budget: bool = False) -> List[Dict[str, Any]]:
    """
    Create table data for WANDB visualization with Dynamic Token Budget information.
    
    Args:
        batch: The batch data
        tokenizer: The tokenizer
        vis_examples: Number of examples to visualize
        use_dynamic_budget: Whether dynamic token budget is enabled
    
    Returns:
        List of dictionaries containing table data with thinking/answer stages separated
    """
    prompts = batch.batch['prompts'][:vis_examples]
    responses = batch.batch['responses'][:vis_examples]

    data_sources = batch.non_tensor_batch.get("data_source", ["unknown"] * len(prompts))
    scene_types = batch.non_tensor_batch.get("scene_type", ["unknown"] * len(prompts))
    
    prompts_text = tokenizer.batch_decode(prompts, skip_special_tokens=False)
    responses_text = tokenizer.batch_decode(responses, skip_special_tokens=False)
    
    # Pre-calculate valid_end for consistency with original code
    valid_end = [int(tokenizer.eos_token in responses_text[i]) for i in range(len(prompts_text))]
    
    table_data = []
    
    for i in range(len(prompts_text)):
        # Basic data that always exists
        row_data = {
            "prompt": prompts_text[i],
            "response": responses_text[i],
            "reward": batch.batch["token_level_scores"][i].sum(-1).item(),
            "valid_end": valid_end[i],  # Use pre-calculated value like original code
        }
        
        # Add parsed answer if available
        if "parsed_answer" in batch.non_tensor_batch:
            row_data["parsed_answer"] = batch.non_tensor_batch["parsed_answer"][i]
        else:
            row_data["parsed_answer"] = "NaN"
        
        # Add ground truth if available - match original code behavior exactly
        if "reward_model" in batch.non_tensor_batch:
            row_data["ground_truth"] = batch.non_tensor_batch["reward_model"][i].get("ground_truth", "NaN")
        else:
            row_data["ground_truth"] = "NaN"
        
        # Add Dynamic Token Budget specific information
        if use_dynamic_budget:
            parsed = parse_two_stage_response(responses_text[i], tokenizer)
            row_data.update({
                "thinking_text": parsed["thinking_text"],
                "answer_text": parsed["answer_text"],
                "thinking_length": parsed["thinking_length"],
                "answer_length": parsed["answer_length"],
                "is_two_stage": parsed["is_two_stage"],
                "has_early_stopping": parsed["has_early_stopping"]
            })
        else:
            # Add default values for non-dynamic budget cases
            row_data.update({
                "thinking_text": "",
                "answer_text": responses_text[i],
                "thinking_length": 0,
                "answer_length": len(tokenizer.encode(responses_text[i], add_special_tokens=False)),
                "is_two_stage": False,
                "has_early_stopping": False
            })

        row_data.update({
            "data_source": data_sources[i],
            "scene_type": scene_types[i]
        })
        
        table_data.append(row_data)
    
    return table_data

def log_dynamic_budget_charts(metrics: Dict[str, Any], logger) -> None:
    """
    Log Dynamic Token Budget specific charts to WANDB.
    
    Args:
        metrics: Dictionary containing all metrics
        logger: The logger instance (e.g., wandb)
    """
    # Create charts for thinking vs answer stage lengths over time
    if "thinking_stage/length/mean" in metrics and "answer_stage/length/mean" in metrics:
        # These will be automatically plotted as line charts in WANDB
        pass  # The metrics are already in the correct format for WANDB
    
    # Could add more sophisticated visualizations here if needed
    # For example, histogram of length distributions, etc.
    pass 