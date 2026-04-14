import re, os, json
from typing import List, Any
import pandas as pd
import numpy as np

from llm.utils.basic_utils import extract_bracket_content
from llm.utils import basic_utils

# Ensure you have math_verify installed: pip install math_verify
from math_verify import (
    parse,
    verify,
)

# System instructions for different question types (from original JEEBench)
prompt_library = {
    "MCQ": "In this problem, only one option will be correct. Give a detailed solution and end the solution with the final answer.",
    "MCQ(multiple)": "In this problem, multiple options can be correct. Give a detailed solution and end the solution with the final answer.", 
    "Integer": "In this problem, the final answer will be a non-negative integer. Give a detailed solution and end the solution with the final answer.",
    "Numeric": "In this problem, the final will be a numeric value. Give the numerical answer correct upto the 2nd decimal digit. Give a detailed solution and end the solution with the final answer.",
}

# Strong mechanics keywords - clearly indicate mechanics-related vocabulary
STRONG_MECHANICS_KEYWORDS = [
    # Basic mechanics concepts
    'force', 'forces', 'friction', 'gravity', 'weight',
    'mass', 'momentum', 'impulse', 'collision', 'impact',
    'velocity', 'acceleration', 'motion', 'kinematics', 'dynamics',
    
    # Mechanical objects
    'stick', 'rod', 'beam', 'bar', 'block', 'particle', 'body',
    'ball', 'disk', 'cylinder', 'sphere', 'wheel', 'pulley',
    
    # Mechanical phenomena and concepts
    'equilibrium', 'balance', 'torque', 'moment', 'lever',
    'inclined plane', 'pendulum', 'spring', 'oscillation',
    'rotation', 'angular velocity', 'angular acceleration',
    'centripetal', 'centrifugal', 'circular motion',
    
    # Energy and work
    'kinetic energy', 'potential energy', 'work done',
    'elastic', 'inelastic', 'conservation of energy',
    'conservation of momentum',
    
    # Mechanical systems
    'uniform rod', 'rigid body', 'center of mass',
    'inclined', 'horizontal surface', 'vertical wall',
    'smooth surface', 'rough surface'
]

# Exclude keywords - clearly indicate non-mechanics fields
EXCLUDE_KEYWORDS = [
    # Atomic physics/quantum mechanics
    'hydrogen atom', 'electron', 'photoelectron', 'photon',
    'planck', 'quantum', 'orbital', 'nucleus', 'atomic',
    'rydberg', 'bohr', 'emission spectrum', 'absorption',
    
    # Optics
    'lens', 'mirror', 'refractive index', 'refraction',
    'reflection', 'optical', 'wavelength', 'frequency',
    'interference', 'diffraction', 'polarization',
    
    # Electromagnetism
    'electric field', 'magnetic field', 'capacitor',
    'inductor', 'resistance', 'current', 'voltage',
    'electromagnetic', 'dielectric', 'permittivity',
    
    # Thermodynamics
    'temperature', 'heat', 'thermal', 'furnace',
    'black-body radiation', 'stefan-boltzmann',
    'thermodynamic', 'entropy', 'enthalpy',
    
    # Nuclear physics
    'radioactive', 'decay', 'isotope', 'nuclear',
    'alpha particle', 'beta decay', 'gamma ray'
]

def is_mechanics_question(question_text: str) -> bool:
    """
    Determine if a question is mechanics-related
    
    Args:
        question_text: Question text content
        
    Returns:
        Whether the question is a mechanics problem
    """
    text_lower = question_text.lower()
    
    # Check exclude keywords first
    for keyword in EXCLUDE_KEYWORDS:
        if keyword.lower() in text_lower:
            return False
    
    # Check strong mechanics keywords
    strong_matches = []
    for keyword in STRONG_MECHANICS_KEYWORDS:
        if keyword.lower() in text_lower:
            strong_matches.append(keyword)
    
    # Require at least 2 strong keywords, or 1 very clear indicator
    very_strong = ['collision', 'friction', 'equilibrium', 'pendulum', 'pulley', 'lever']
    very_strong_matches = [kw for kw in strong_matches if kw.lower() in very_strong]
    
    if very_strong_matches:
        return True
    elif len(strong_matches) >= 2:
        return True
    else:
        return False

def extract_answer_from_generation(llm_generation: str, question_type: str) -> str:
    """
    Extract the final answer from LLM generation based on question type.
    
    Args:
        llm_generation: The LLM's complete response
        question_type: Type of question (MCQ, MCQ(multiple), Integer, Numeric)
        
    Returns:
        Extracted answer string, or "None" if no valid answer found
    """
    # First try to extract from \boxed{}
    boxed_content = extract_bracket_content(llm_generation)
    
    if boxed_content:
        answer = boxed_content[-1].strip()
        
        if question_type == "MCQ":
            # Single choice: should be A, B, C, or D
            answer = answer.upper()
            if answer in ["A", "B", "C", "D"]:
                return answer
                
        elif question_type == "MCQ(multiple)":
            # Multiple choice: should be combination of A, B, C, D
            answer = answer.upper()
            # Remove spaces and sort letters
            letters = ''.join(sorted(set(c for c in answer if c in "ABCD")))
            if letters:
                return letters
                
        elif question_type in ["Integer", "Numeric"]:
            # Try to parse as number
            try:
                # Remove any non-numeric characters except decimal point and minus
                cleaned = re.sub(r'[^\d.-]', '', answer)
                if cleaned:
                    return cleaned
            except:
                pass
    
    # Fallback: try to find patterns in the full text
    if question_type == "MCQ":
        # Look for patterns like "answer is A" or "option A"
        matches = re.findall(r'\b([ABCD])\b', llm_generation.upper())
        if matches:
            return matches[-1]  # Take the last one
            
    elif question_type == "MCQ(multiple)":
        # Look for multiple letters
        matches = re.findall(r'\b([ABCD])\b', llm_generation.upper())
        if matches:
            return ''.join(sorted(set(matches)))
            
    elif question_type in ["Integer", "Numeric"]:
        # Look for numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*', llm_generation)
        if numbers:
            return numbers[-1]  # Take the last number
    
    return "None"

def compute_score(gold: str, resp: str, question_type: str) -> float:
    """
    Original JEEBench evaluation function for exact reproduction.
    
    Args:
        gold: Gold standard answer
        resp: Predicted answer from LLM
        question_type: Type of question
        
    Returns:
        Score (float value based on original JEEBench logic)
    """
    QUES_TYPES = ['MCQ','MCQ(multiple)','Integer','Numeric']
    assert question_type in QUES_TYPES
    
    if question_type == 'MCQ(multiple)':
        gold_set = set([c for c in ['A', 'B', 'C', 'D'] if c in gold])
        resp_set = set([c for c in ['A', 'B', 'C', 'D'] if c in resp])
        if resp_set == gold_set:
            return 1.0
        else:
            if len(resp_set - gold_set) == 0:  # No wrong options selected
                return 0.25 * len(resp_set)  # Partial credit
            return 0.0  # Wrong options selected
    elif question_type == 'MCQ':
        gold_set = set([c for c in ['A', 'B', 'C', 'D'] if c in gold])
        resp_set = set([c for c in ['A', 'B', 'C', 'D'] if c in resp])
        return int(gold_set == resp_set)
    else:  # Integer or Numeric
        if resp == "None":
            return 0.0
        try:
            g, r = float(gold), float(resp)
            return int(abs(g - r) <= 0.01)  # Original JEEBench precision
        except:
            return 0.0

def get_prompts(cfg):
    """Load JEEBench dataset and prepare prompts with optional filtering."""
    import ipdb; ipdb.set_trace()
    with open(cfg.data.val_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Apply subject filtering (default to physics if filter_physics is True)
    filter_physics = getattr(cfg, 'filter_physics', True)  # Default to True
    if filter_physics:
        df = df[df['subject'] == 'phy']
        print(f"Filtered to physics questions: {len(df)} questions")
    
    # Apply mechanics filtering (default to True for physics questions)
    filter_mechanics = getattr(cfg, 'filter_mechanics', True)  # Default to True
    if filter_mechanics and filter_physics:
        original_count = len(df)
        mechanics_mask = df['question'].apply(lambda x: is_mechanics_question(x))
        df = df[mechanics_mask]
        print(f"Filtered to mechanics questions: {len(df)} questions (from {original_count} physics questions)")
    elif filter_mechanics and not filter_physics:
        print("Warning: filter_mechanics=True but filter_physics=False. Mechanics filtering works best with physics questions.")
        original_count = len(df)
        mechanics_mask = df['question'].apply(lambda x: is_mechanics_question(x))
        df = df[mechanics_mask]
        print(f"Filtered to mechanics questions: {len(df)} questions (from {original_count} total questions)")
    
    # Apply question type filtering if specified
    filter_type = getattr(cfg, 'filter_type', None)
    if filter_type:
        df = df[df['type'] == filter_type]
        print(f"Filtered to question type '{filter_type}': {len(df)} questions")
    
    # Add prompts field using original JEEBench prompt format
    df["prompts"] = df.apply(lambda row: [
        {
            'role': 'user',
            'content': prompt_library[row["type"]] + "\n\nProblem: " + row["question"].replace("\n\n", "\n").strip()
        }
    ], axis=1)
    
    return df

def generate_responses(cfg, llm_prompts, llm, sampling_params):
    """Generate responses using the LLM."""
    output_texts = []
    if cfg.solve_locally:
        all_model_outputs, token_stats = basic_utils.call_model(llm, sampling_params, llm_prompts, cfg)
        for model_outputs in all_model_outputs:
            for output in model_outputs.outputs:
                output_texts.append(output.text)
    else:
        all_model_outputs, token_stats, error_generating = basic_utils.call_llm_api(
            cfg, cfg.model_name, llm_prompts, n=cfg.actor_rollout_ref.rollout.val_kwargs.n
        )
        for model_outputs in all_model_outputs:
            for output_text in model_outputs:
                output_texts.append(output_text)

    return output_texts, token_stats

def benchmark(cfg, llm, sampling_params, logger):
    """Main benchmark function for JEEBench evaluation."""
    
    # Load dataset
    questions_df = get_prompts(cfg)
    
    print(f"Loaded {len(questions_df)} questions from JEEBench")
    print(f"Subject distribution:")
    subject_counts = questions_df['subject'].value_counts()
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count}")
    
    print(f"Question type distribution:")
    type_counts = questions_df['type'].value_counts()
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count}")
    
    # Generate responses
    output_texts, token_stats = generate_responses(
        cfg, questions_df["prompts"].to_list(), llm, sampling_params
    )
    
    # Handle multiple responses per question (if n > 1)
    n_responses = cfg.actor_rollout_ref.rollout.val_kwargs.n
    if n_responses > 1 and len(output_texts) == len(questions_df) * n_responses:
        # Multiple responses per question - take the first one for evaluation
        output_texts = output_texts[::n_responses]
        # Also adjust token stats
        for k in token_stats:
            if len(token_stats[k]) == len(questions_df) * n_responses:
                token_stats[k] = token_stats[k][::n_responses]
    
    # Add responses to dataframe
    questions_df["llm_output"] = output_texts
    for k in token_stats:
        questions_df[k] = token_stats[k]
    
    # Evaluate responses using original JEEBench logic
    predictions = []
    scores = []
    
    for i, row in questions_df.iterrows():
        predicted = extract_answer_from_generation(row["llm_output"], row["type"])
        score = compute_score(row["gold"], predicted, row["type"])
        
        predictions.append(predicted)
        scores.append(score)
    
    questions_df["predicted"] = predictions
    questions_df["score"] = scores
    
    # Compute overall statistics
    overall_accuracy = np.mean(scores)
    total_questions = len(questions_df)
    correct_answers = sum(scores)
    
    # Compute random baseline (from original JEEBench)
    random_scores = []
    for i, row in questions_df.iterrows():
        if row["type"] == "MCQ":
            random_scores.append(0.25)  # 1/4 chance
        elif row["type"] == "MCQ(multiple)":
            num_ans = len(row["gold"])
            if num_ans == 1:
                random_scores.append(0.0625)
            elif num_ans == 2:
                random_scores.append(0.09375)
            elif num_ans == 3:
                random_scores.append(0.203125)
            elif num_ans == 4:
                random_scores.append(0.5)
        else:  # Integer/Numeric
            random_scores.append(0.0)
    random_baseline = np.mean(random_scores)
    
    # Compute statistics by subject
    subject_stats = {}
    for subject in questions_df['subject'].unique():
        subject_df = questions_df[questions_df['subject'] == subject]
        subject_accuracy = subject_df['score'].mean()
        subject_stats[subject] = {
            'accuracy': subject_accuracy,
            'count': len(subject_df),
            'correct': subject_df['score'].sum()
        }
    
    # Compute statistics by question type
    type_stats = {}
    for qtype in questions_df['type'].unique():
        type_df = questions_df[questions_df['type'] == qtype]
        type_accuracy = type_df['score'].mean()
        type_stats[qtype] = {
            'accuracy': type_accuracy,
            'count': len(type_df),
            'correct': type_df['score'].sum()
        }
    
    # Compute token statistics
    avg_input_length = questions_df["tokens/input"].mean() if "tokens/input" in questions_df.columns else 0
    avg_response_length = questions_df["tokens/output"].mean() if "tokens/output" in questions_df.columns else 0
    valid_end_rate = (questions_df["tokens/valid_end"].sum() / len(questions_df)) if "tokens/valid_end" in questions_df.columns else 0
    
    # Print results in original JEEBench format
    print(f"\n=== JEEBench Results (Original Format) ===")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Random Baseline: {random_baseline:.3f}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    
    print(f"\n=== Results by Subject ===")
    for subject, stats in subject_stats.items():
        print(f"{subject}: {stats['accuracy']:.3f}")
    
    print(f"\n=== Results by Question Type ===")
    for qtype, stats in type_stats.items():
        print(f"{qtype}: {stats['accuracy']:.3f}")
    
    # Additional stats for comparison with original paper
    print(f"\n=== Token Statistics ===")
    print(f"Average Input Length: {avg_input_length:.1f} tokens")
    print(f"Average Response Length: {avg_response_length:.1f} tokens") 
    print(f"Valid End Rate: {valid_end_rate:.4f}")
    
    # Log to wandb (including random baseline for comparison)
    logger.log({
        'accuracy/overall': overall_accuracy,
        'accuracy/random_baseline': random_baseline,
        'stats/total_questions': total_questions,
        'stats/correct_answers': correct_answers,
        'tokens/input_length': avg_input_length,
        'tokens/response_length': avg_response_length,
        'tokens/valid_end_rate': valid_end_rate
    }, step=0)
    
    # Log subject-wise results
    for subject, stats in subject_stats.items():
        logger.log({
            f'accuracy/{subject}': stats['accuracy'],
            f'count/{subject}': stats['count']
        }, step=0)
    
    # Log question type results
    for qtype, stats in type_stats.items():
        logger.log({
            f'accuracy/{qtype}': stats['accuracy'],
            f'count/{qtype}': stats['count']
        }, step=0)
    
    # Save detailed results
    model_name = cfg.model_name.replace("/", "_")
    os.makedirs(f"evals/JEEBench/generations/{model_name}", exist_ok=True)
    questions_df.to_json(f"evals/JEEBench/generations/{model_name}/results.jsonl", 
                        lines=True, orient="records")
    
    print(f"\nDetailed results saved to: evals/JEEBench/generations/{model_name}/results.jsonl")
    
    logger.finish()

def test():
    """Test function to verify answer extraction and evaluation (original JEEBench format)."""
    
    # Test MCQ
    mcq_generation = "After solving step by step, the answer is B"
    predicted = extract_answer_from_generation(mcq_generation, "MCQ")
    score = compute_score("B", predicted, "MCQ")
    print(f"MCQ Test: predicted='{predicted}', score={score}")
    
    # Test MCQ(multiple) - full credit
    mcq_multi_generation = "The correct options are A and D. So AD"
    predicted = extract_answer_from_generation(mcq_multi_generation, "MCQ(multiple)")
    score = compute_score("AD", predicted, "MCQ(multiple)")
    print(f"MCQ(multiple) Test (full): predicted='{predicted}', score={score}")
    
    # Test MCQ(multiple) - partial credit
    mcq_partial_generation = "I think A is correct"
    predicted = extract_answer_from_generation(mcq_partial_generation, "MCQ(multiple)")
    score = compute_score("AD", predicted, "MCQ(multiple)")
    print(f"MCQ(multiple) Test (partial): predicted='{predicted}', score={score}")
    
    # Test Integer
    integer_generation = "The final answer is 42"
    predicted = extract_answer_from_generation(integer_generation, "Integer")
    score = compute_score("42", predicted, "Integer")
    print(f"Integer Test: predicted='{predicted}', score={score}")
    
    # Test Numeric
    numeric_generation = "Therefore, the value is 3.14"
    predicted = extract_answer_from_generation(numeric_generation, "Numeric")
    score = compute_score("3.14", predicted, "Numeric")
    print(f"Numeric Test: predicted='{predicted}', score={score}")
    
    # Test Numeric with tolerance
    numeric_close_generation = "The answer is approximately 3.141"
    predicted = extract_answer_from_generation(numeric_close_generation, "Numeric")
    score = compute_score("3.14", predicted, "Numeric")
    print(f"Numeric Test (tolerance): predicted='{predicted}', score={score}")

if __name__ == "__main__":
    test() 