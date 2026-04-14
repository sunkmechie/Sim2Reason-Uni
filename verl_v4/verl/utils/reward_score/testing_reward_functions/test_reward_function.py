'''
This file tests the different reward functions.

The main functions are:
- generate_new_question_and_answer_sample: generates a new question and answer sample output in a json file.
- test_reward_functions: tests the reward functions on the sample question and
  answer and writes the results to a file.
'''

import os, glob, re, time, json, shutil, tqdm, queue, threading, sys, random, logging, traceback, textwrap
from io import StringIO
from tabulate import tabulate
from google import genai
from google.genai import types

from verl.utils.reward_score.math_combined import compute_score_combined_original, compute_score_combined_updated

testing_prompts = False # True for testing prompts. False for batch processing.
GEMINI_KEY_SOURCE = "GEMINI_KEY2"
# client = genai.Client(api_key=os.environ.get(GEMINI_KEY_SOURCE))  # Initialize only when needed
testing_model_name = "gemini-2.5-flash-preview-05-20"
batch_model_name = "gemini-2.5-flash-preview-05-20"
model_name = testing_model_name if testing_prompts else batch_model_name
NUM_THREADS = 50  # Number of concurrent threads

def extract_path_from_pho(file_path):
    """
    Extract the file path starting from 'PHO' for cleaner logging output.
    
    Args:
        file_path (str): The full file path
        
    Returns:
        str: The path starting from 'PHO', or the original path if 'PHO' is not found
    """
    if not file_path:
        return file_path
    
    pho_index = str(file_path).find('PHO')
    if pho_index != -1:
        return str(file_path)[pho_index:]
    else:
        return str(file_path)

def get_next_log_file():
    """Get the next available log file name with 4-digit number."""
    log_dir = "Log-Files/TestRewardFunction"
    os.makedirs(log_dir, exist_ok=True)
    
    # Get existing log files with pattern log-XXXX.log
    existing_logs = glob.glob(os.path.join(log_dir, "log-????.log"))
    existing_numbers = []
    
    for log_file in existing_logs:
        match = re.search(r'log-(\d{4})\.log$', log_file)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # Find the lowest unused number
    next_number = 1
    while next_number < 10000:
        if next_number not in existing_numbers:
            break
        next_number += 1
    
    return os.path.join(log_dir, f"log-{next_number:04d}.log")

def call_gemini_api(prompt, image_data=None):
    """
    Call the Gemini API with the provided prompt and optional image
    
    Args:
        prompt (str): The prompt to send to the Gemini API
        image_data (str or bytes, optional): Base64 encoded image data, Python string representation, or raw binary data
        
    Returns:
        str: The response from the Gemini API
    """
    try:
        # Check if API key is available
        api_key = os.environ.get(GEMINI_KEY_SOURCE)
        if not api_key:
            print(f"ERROR: Gemini API key not found!")
            print(f"Please set the environment variable '{GEMINI_KEY_SOURCE}' with your Gemini API key.")
            return None
        
        # Initialize the client
        client = genai.Client(api_key=api_key)
        
        if image_data:
            # For now, just use text-only since we don't expect images in this use case
            response = client.models.generate_content(
                model=model_name,
                contents=prompt + " (Note: Image processing not implemented in this context)"
            )
        else:
            # Send text-only prompt to the API
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
        
        # Add a sleep to avoid rate limiting
        time.sleep(3)
        
        # Return the response text
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return None
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        print(traceback.format_exc())
        return None

def generate_new_question_and_answer_sample(num_questions, output_file):
    '''
    Generates a new question and answer sample output in a json file.
    If the output file already exists, it will be moved to Old_QA_Pairs folder
    with a numbered name before creating the new file.

    Args:
        num_questions (int): The number of questions to generate.
        output_file (str): The file to write the sample to.

    Returns:
        None - the sample QA pairs are written to the output file.
    '''
    # Set random seed based on current time for different samples each run
    random.seed(int(time.time() * 1000) % 2**32)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    
    # Check if output file already exists and move it to Old_QA_Pairs folder
    if os.path.exists(output_path):
        # Check if the existing file contains any LLM predicted answers
        has_llm_answers = False
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    # If it's not a JSON array, try reading line by line (JSONL format)
                    f.seek(0)
                    existing_data = []
                    for line in f:
                        try:
                            existing_data.append(json.loads(line))
                        except:
                            continue
            
            # Check if any item has LLM predicted answer fields
            llm_answer_pattern = re.compile(r'LLM_\d{2}_predicted_answer')
            for item in existing_data:
                if isinstance(item, dict):
                    for key in item.keys():
                        if llm_answer_pattern.match(key):
                            has_llm_answers = True
                            break
                if has_llm_answers:
                    break
                    
        except Exception as e:
            print(f"Warning: Could not read existing file {extract_path_from_pho(output_path)}: {e}")
            # If we can't read the file, assume it doesn't have LLM answers
            has_llm_answers = False
        
        # Only move the file if it contains LLM answers
        if has_llm_answers:
            # Create Old_QA_Pairs directory if it doesn't exist
            old_qa_dir = os.path.join(script_dir, "Old_QA_Pairs")
            os.makedirs(old_qa_dir, exist_ok=True)
            
            # Find the next available number for the old file
            existing_files = glob.glob(os.path.join(old_qa_dir, "???_Old_QA_Pairs.json.log"))
            existing_numbers = []
            
            for old_file in existing_files:
                match = re.search(r'(\d{3})_Old_QA_Pairs\.json\.log$', old_file)
                if match:
                    existing_numbers.append(int(match.group(1)))
            
            # Find the lowest unused number
            next_number = 1
            while next_number < 1000:
                if next_number not in existing_numbers:
                    break
                next_number += 1
            
            # Move the existing file to Old_QA_Pairs with the new name
            old_file_name = f"{next_number:03d}_Old_QA_Pairs.json.log"
            old_file_path = os.path.join(old_qa_dir, old_file_name)
            shutil.move(output_path, old_file_path)
            # Extract path starting from PHO for cleaner output
            display_path = extract_path_from_pho(old_file_path)
            print(f"Moved existing QA pairs file with LLM answers to: {display_path}")
        else:
            # Remove the file since it doesn't contain LLM answers
            os.remove(output_path)
            # Extract path starting from PHO for cleaner output
            display_path = extract_path_from_pho(output_path)
            print(f"Removed existing QA pairs file (no LLM answers found): {display_path}")
    
    # Go up 5 levels to reach the root directory (PHO)
    root_dir = script_dir
    for _ in range(5):
        root_dir = os.path.dirname(root_dir)

    # Commented out Type 3A and 3B since all of the sources in those folders
    # have files which contain misaligned questions and answers.
    dirs_to_process = [
        os.path.join(root_dir, "llm", "Type_2_Sources"),
        # os.path.join(root_dir, "llm", "Type_3A_Sources"),
        # os.path.join(root_dir, "llm", "Type_3B_Sources"),
        os.path.join(root_dir, "llm", "Type_5_Sources"),
    ]
    # Initialize lists for different answer types
    numerical_questions = []
    equation_questions = []
    letter_single_questions = []  # Letter answers with single string or single-element list standardized_answer
    letter_multi_questions = []   # Letter answers with multi-element list standardized_answer
    
    # Loop through each directory and process JSON files
    for dir_path in dirs_to_process:
        if os.path.exists(dir_path):
            # Use glob to recursively find all JSON files
            json_pattern = os.path.join(dir_path, "**", "*.json")
            json_files = glob.glob(json_pattern, recursive=True)
            # Extract path starting from PHO for cleaner output
            display_dir = extract_path_from_pho(dir_path)
            print(f"Found {len(json_files)} JSON files in {display_dir}")
            
            # Process each JSON file
            for json_file in json_files:
                try:
                    # Extract source number from file path (e.g., "050" from path containing "/050/")
                    path_parts = str(json_file).split(os.sep)
                    source_number = None
                    for part in path_parts:
                        if part.isdigit() and len(part) == 3:  # Looking for 3-digit source numbers
                            source_number = part
                            break
                    
                    # First read the raw content to check for escaped Unicode sequences
                    with open(json_file, 'r', encoding='utf-8') as f:
                        raw_content = f.read()
                    
                    # Skip files that contain escaped Unicode sequences
                    # Be very strict - no Unicode escapes allowed at all
                    if '\\u' in raw_content:
                        continue
                    
                    # Now load the JSON data
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Handle both single objects and arrays
                        items_to_process = [data] if isinstance(data, dict) else data
                        
                        for item in items_to_process:
                            if isinstance(item, dict):
                                # Check if question meets our criteria
                                if (item.get('verifiableAnswer', False) and 
                                    item.get('answerType') in ['numerical', 'equation', 'letter'] and
                                    item.get('text', '') and item.get('answer', '')):
                                    
                                    # Check for Unicode characters in the actual text content
                                    # This catches both decoded Unicode characters and escaped sequences
                                    def contains_unicode_chars(text):
                                        if not isinstance(text, str):
                                            return False
                                        
                                        # Check for escaped Unicode sequences (like \u00b0)
                                        if '\\u' in text:
                                            return True
                                        
                                        # Check for non-ASCII characters
                                        for char in text:
                                            if ord(char) > 127:  # Non-ASCII character
                                                # Be more restrictive - only allow very common math symbols
                                                # Remove degree symbol and other potentially problematic characters
                                                if char not in ['π', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'ρ', 'σ', 'τ', 'φ', 'ω']:
                                                    return True
                                        return False
                                    
                                    # Check all string fields for problematic Unicode
                                    # Also check the raw JSON representation of this item for escaped Unicode
                                    item_json = json.dumps(item)
                                    if '\\u' in item_json:
                                        continue
                                    
                                    skip_item = False
                                    for key, value in item.items():
                                        if isinstance(value, str) and contains_unicode_chars(value):
                                            skip_item = True
                                            break
                                        elif isinstance(value, list):
                                            for list_item in value:
                                                if isinstance(list_item, str) and contains_unicode_chars(list_item):
                                                    skip_item = True
                                                    break
                                            if skip_item:
                                                break
                                    
                                    if skip_item:
                                        continue
                                    
                                    # Add source information to the item
                                    item_with_source = item.copy()
                                    # Extract path starting from PHO
                                    display_path = extract_path_from_pho(json_file)
                                    item_with_source['source_file'] = display_path
                                    item_with_source['source_number'] = source_number
                                    
                                    # Categorize by answer type
                                    answer_type = item.get('answerType')
                                    if answer_type == 'numerical':
                                        numerical_questions.append(item_with_source)
                                    elif answer_type == 'equation':
                                        equation_questions.append(item_with_source)
                                    elif answer_type == 'letter':
                                        # Check standardized_answer field structure
                                        std_answer = item.get('standardized_answer', item.get('answer', ''))
                                        
                                        # Determine if it's single or multi based on standardized_answer structure
                                        if isinstance(std_answer, str):
                                            # Single string - goes to single list
                                            letter_single_questions.append(item_with_source)
                                        elif isinstance(std_answer, list):
                                            if len(std_answer) == 1:
                                                # List with single element - goes to single list
                                                letter_single_questions.append(item_with_source)
                                            else:
                                                # List with multiple elements - goes to multi list
                                                letter_multi_questions.append(item_with_source)
                                        else:
                                            # Fallback: if standardized_answer is neither string nor list,
                                            # use the answer field and treat as single
                                            letter_single_questions.append(item_with_source)
                                            
                except Exception as e:
                    # Skip files that can't be processed
                    continue
        else:
            # Extract path starting from PHO for cleaner output
            display_dir = extract_path_from_pho(dir_path)
            print(f"Warning: Directory {display_dir} does not exist")
    
    print(f"Categorized questions:")
    print(f"  Numerical: {len(numerical_questions)}")
    print(f"  Equation: {len(equation_questions)}")
    print(f"  Letter (single): {len(letter_single_questions)}")
    print(f"  Letter (multi): {len(letter_multi_questions)}")
    
    # Calculate target numbers for each category
    numerical_target = num_questions // 3
    equation_target = num_questions // 3
    letter_single_target = num_questions // 3
    #letter_multi_target = num_questions - numerical_target - equation_target - letter_single_target
    
    print(f"Target samples:")
    print(f"  Numerical: {numerical_target}")
    print(f"  Equation: {equation_target}")
    print(f"  Letter (single): {letter_single_target}")
    # print(f"  Letter (multi): {letter_multi_target}")
    
    def sample_by_source_proportion(questions_list, target_count):
        """Sample questions proportionally by source number"""
        if not questions_list or target_count == 0:
            return []
        
        # Count questions by source
        source_counts = {}
        for q in questions_list:
            source = q.get('source_number', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        total_questions = len(questions_list)
        sampled_questions = []
        # Use current time for truly random sampling
        random.seed(int(time.time() * 1000) % 2**32)
        
        # Sample proportionally from each source
        for source, count in source_counts.items():
            proportion = count / total_questions
            source_target = max(1, round(target_count * proportion))  # At least 1 if source exists
            
            # Get questions from this source
            source_questions = [q for q in questions_list if q.get('source_number') == source]
            
            # Sample from this source
            if len(source_questions) >= source_target:
                sampled = random.sample(source_questions, source_target)
            else:
                sampled = source_questions  # Take all if not enough
            
            sampled_questions.extend(sampled)
        
        # If we have too many, randomly remove some; if too few, randomly add more
        if len(sampled_questions) > target_count:
            sampled_questions = random.sample(sampled_questions, target_count)
        elif len(sampled_questions) < target_count:
            remaining_needed = target_count - len(sampled_questions)
            remaining_questions = [q for q in questions_list if q not in sampled_questions]
            if remaining_questions:
                additional = random.sample(remaining_questions, 
                                         min(remaining_needed, len(remaining_questions)))
                sampled_questions.extend(additional)
        
        return sampled_questions
    
    # Sample from each category proportionally by source
    sampled_numerical = sample_by_source_proportion(numerical_questions, numerical_target)
    sampled_equation = sample_by_source_proportion(equation_questions, equation_target)
    sampled_letter_single = sample_by_source_proportion(letter_single_questions, letter_single_target)
    # sampled_letter_multi = sample_by_source_proportion(letter_multi_questions, letter_multi_target)
    
    # Combine all sampled questions
    final_sample = sampled_numerical + sampled_equation + sampled_letter_single # + sampled_letter_multi
    
    print(f"Final sample:")
    print(f"  Numerical: {len(sampled_numerical)}")
    print(f"  Equation: {len(sampled_equation)}")
    print(f"  Letter (single): {len(sampled_letter_single)}")
    #print(f"  Letter (multi): {len(sampled_letter_multi)}")
    print(f"  Total: {len(final_sample)}")
    
    # Write the results to the output file with 4 spaces indent
    # Use ensure_ascii=False to properly handle Unicode characters
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_sample, f, indent=4, ensure_ascii=False)
    
    # Extract path starting from PHO for cleaner output
    display_path = extract_path_from_pho(output_path)
    print(f"Saved {len(final_sample)} questions to {display_path}")
    return final_sample

def get_prompt(item, llm_name):
    '''
    Generate a prompt for the LLM based on the question and answer type.
    
    Args:
        item (dict): JSON object with question, answer type, and other metadata
        llm_name (str): Name of the LLM being used
        
    Returns:
        str: Prompt for the LLM
    '''
    question_text = item.get('text', '')
    answer_type = item.get('answerType', 'uncategorizable')
    descriptions = item.get('descriptions', [])
    
    # Add image descriptions if available
    context = ""
    if descriptions:
        context = "Descriptions or answer choices:" + ", ".join(descriptions)
    
    base_prompt = f"Please solve the following physics problem: Question: {question_text}. {context}. "
    
    if answer_type == "numerical":
        type_specific = "Provide a numerical answer with SI units, in decimal digit notation, not scientific notation, where there is a space between the number and the unit. If the answer is something like x = 5 m, just put 5 m. Show your work step by step and give the final numerical value with units, using as many digits as possible to be most accurate. Use proper mathematical notation and LaTeX formatting where appropriate. Convert your final answer into SI units if possible. Box the final answer (in SI units if applicable). Do not put the boxed final numerical answer in \\text{}, only put the unit in \\text{} if there is a unit of measurement. If the answer is a multiple choice question, provide the index of the answer choice that is correct or box -1.2345 as your final answer if you are not sure. Always box the final answer, even it is the index of the answer choice for a multiple choice question."
    elif answer_type == "letter":
        # Check if this is a single or multiple answer letter question
        std_answer = item.get('standardized_answer', item.get('answer', ''))
        
        if isinstance(std_answer, list) and len(std_answer) > 1:
            # Multiple letter answers
            type_specific = "This is a multiple choice question with multiple correct answers. Provide all the letters of the correct answers (in a comma separated list. e.g. A,D,E) and briefly explain your reasoning for each. Use proper mathematical notation and LaTeX formatting where appropriate. Box the final answer which should be a comma separated list of letters corresponding to the correct answers."
        else:
            # Single letter answer (either string or single-element list)
            type_specific = "This is a multiple choice question. Provide the letter of the correct answer (A, B, C, D, etc.) and briefly explain your reasoning. Use proper mathematical notation and LaTeX formatting where appropriate. Box the final answer which should just be the letter or answer choice, or if you are not sure, box -1.2345 as your final answer."
    elif answer_type == "equation":
        type_specific = "Provide the equation or mathematical expression that answers the question in LaTeX format with proper latex delimiters. Don't use \\text{} besides for units of measurement if applicable. Use proper mathematical notation and LaTeX formatting where appropriate. Box the final answer."
    else:
        type_specific = "Provide a clear and complete answer to the question. Show your work and reasoning."
    
    prompt = base_prompt + type_specific
    
    return prompt

def process_llm_item(item, llm_list, thread_print):
    """
    Process a single JSON item to get answers from all LLMs
    
    Args:
        item (dict): The JSON object to process
        llm_list (list): List of LLM configurations
        thread_print: Function to print thread-specific logs
        
    Returns:
        dict: The processed JSON object with LLM answers added
    """
    try:
        for i, llm_config in enumerate(llm_list):
            llm_name = llm_config.get('name', f'LLM_{i+1}')
            field_name = f"LLM_{i+1:02d}_predicted_answer"
            
            # Check if this LLM should be used
            if not llm_config.get('use_LLM', False):
                thread_print(f"Skipping {llm_name} (use_LLM=False)")
                item[field_name] = ""  # Set empty string for skipped LLMs
                continue
            
            thread_print(f"Getting answer from {llm_name} for item")
            
            # Get the prompt for this LLM and item
            prompt = get_prompt(item, llm_name)
            
            # Call the LLM API using the specified api_function
            api_function = llm_config.get('api_function', call_gemini_api)
            llm_response = api_function(prompt)
            
            # Store the response in the appropriate field
            if llm_response:
                item[field_name] = llm_response
                thread_print(f"Got response from {llm_name}: {llm_response[:100]}...")
            else:
                item[field_name] = ""
                thread_print(f"No response from {llm_name}")
        
        return item
    except Exception as e:
        thread_print(f"Error processing LLM item: {e}")
        thread_print(traceback.format_exc())
        return item

def worker_process_llms(thread_id, item_queue, processed_items, progress_bar, progress_lock, stats_lock, stats, llm_list):
    """
    Worker function to process JSON items from the queue for LLM answer generation
    
    Args:
        thread_id (int): Thread identifier
        item_queue (queue.Queue): Queue of JSON items to process
        processed_items (list): Shared list to store processed items
        progress_bar (tqdm.tqdm): Progress bar for visual feedback
        progress_lock (threading.Lock): Lock for updating progress bar
        stats_lock (threading.Lock): Lock for updating statistics
        stats (dict): Statistics dictionary
        llm_list (list): List of LLM configurations
    """
    # Set up thread-specific logging
    log_dir = f"Log-Files/LLMAnswerGeneration"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"thread-{thread_id:03d}.log")
    
    # Create a custom logger for this thread
    logger = logging.getLogger(f"llm-thread-{thread_id}")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    
    # Define a thread-local print function
    def thread_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        logger.info(message)
    
    try:
        thread_print(f"LLM answer generation thread {thread_id} started")
        
        while True:
            try:
                # Get next item from queue with timeout
                item_with_index = item_queue.get(timeout=0.1)
                
                if item_with_index is None:  # Sentinel value
                    item_queue.task_done()
                    break
                
                index, item = item_with_index
                
                # Process the item
                thread_print(f"Processing item {index}")
                start_time = time.time()
                processed_item = process_llm_item(item, llm_list, thread_print)
                
                # Add to shared list of processed items
                with stats_lock:
                    processed_items[index] = processed_item
                    stats['processed_count'] += 1
                    
                    # Update LLM response stats
                    for i, llm_config in enumerate(llm_list):
                        field_name = f"LLM_{i+1:02d}_predicted_answer"
                        if processed_item.get(field_name):
                            stats['llm_responses'][field_name] = stats['llm_responses'].get(field_name, 0) + 1
                        # Track which LLMs were used vs skipped
                        if llm_config.get('use_LLM', False):
                            stats['llm_used'][field_name] = stats['llm_used'].get(field_name, 0) + 1
                        else:
                            stats['llm_skipped'][field_name] = stats['llm_skipped'].get(field_name, 0) + 1
                
                processing_time = time.time() - start_time
                thread_print(f"Completed processing item {index} in {processing_time:.2f} seconds")
                
                # Update progress bar
                with progress_lock:
                    progress_bar.update(1)
                
                # Mark task as done
                item_queue.task_done()
                
            except queue.Empty:
                # Queue is temporarily empty but may receive more items
                continue
                
    except Exception as e:
        thread_print(f"Unhandled error in LLM thread {thread_id}: {e}")
        thread_print(traceback.format_exc())
        
    finally:
        # Clean up resources
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        
        thread_print(f"LLM thread {thread_id} completed")

def generate_answers_for_sample(qa_pairs_file, num_threads=NUM_THREADS):
    '''
    Tests the reward functions on the sample question and answer and writes the results to a file.
    Runs LLM calls in parallel using threads, and then write results back to json file.
    The first LLM's answers are written to field LLM_01_predicted_answer, the second LLM's
    answers are written to field LLM_02_predicted_answer, etc.

    Args:
        qa_pairs_file (str): The file to read the sample from.
        num_threads (int): Number of threads to use for processing

    Returns:
        None - the results are written to the qa_pairs file.
    '''
    # Define list of LLMs to use (currently just Gemini, but can be extended)
    llm_list = [
        {
            'name': 'Gemini-2.5-Flash',
            'model': 'gemini-2.5-flash-preview-05-20',
            'api_function': call_gemini_api,
            'use_LLM': True  # Set to False to skip this LLM
        }
        # Add more LLMs here as needed:
        # {
        #     'name': 'GPT-4',
        #     'model': 'gpt-4',
        #     'api_function': call_openai_api,
        #     'use_LLM': True
        # }
    ]
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qa_pairs_path = os.path.join(script_dir, qa_pairs_file)
    
    if not os.path.exists(qa_pairs_path):
        print(f"Error: QA pairs file not found: {extract_path_from_pho(qa_pairs_path)}")
        return
    
    # Set up main logging
    main_log_file = get_next_log_file()
    print(f"LLM Answer Generation: Processing with multi-threading")
    print(f"Main log file: {extract_path_from_pho(main_log_file)}")
    
    # Redirect stdout and stderr to the main log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    main_log_fd = open(main_log_file, 'w', buffering=1)
    sys.stdout = main_log_fd
    sys.stderr = main_log_fd
    
    try:
        print(f"\nProcessing file: {qa_pairs_file}")
        print("-" * 80)
        
        # Load all items from the input file
        all_items = []
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            try:
                all_items = json.load(f)
            except json.JSONDecodeError:
                # If it's not a JSON array, try reading line by line (JSONL format)
                f.seek(0)
                for line in f:
                    try:
                        all_items.append(json.loads(line))
                    except Exception as e:
                        print(f"Error parsing line: {e}")
                        # Keep the original line as a string for integrity
                        all_items.append({"_original_line": line.strip()})
        
        total_items = len(all_items)
        print(f"Loaded {total_items} items for processing")
        
        # Show which LLMs are enabled/disabled
        enabled_llms = [llm['name'] for llm in llm_list if llm.get('use_LLM', False)]
        disabled_llms = [llm['name'] for llm in llm_list if not llm.get('use_LLM', False)]
        print(f"Total LLMs configured: {len(llm_list)}")
        print(f"Enabled LLMs: {enabled_llms}")
        if disabled_llms:
            print(f"Disabled LLMs: {disabled_llms}")
        
        # Create queue for items to process
        item_queue = queue.Queue()
        for i, item in enumerate(all_items):
            item_queue.put((i, item))
        
        # Create shared list for processed items
        processed_items = [None] * total_items
        
        # Shared locks and statistics
        stats_lock = threading.Lock()
        progress_lock = threading.Lock()
        stats = {
            'total_items': total_items,
            'processed_count': 0,
            'llm_responses': {},
            'llm_used': {},
            'llm_skipped': {}
        }
        
        # Create progress bar
        progress_bar = tqdm.tqdm(total=total_items, desc=f"Generating LLM answers", unit="item", 
                              file=original_stdout)
        
        # Create and start worker threads
        threads = []
        num_workers = min(num_threads, total_items)
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=worker_process_llms,
                args=(i+1, item_queue, processed_items, progress_bar, progress_lock, stats_lock, stats, llm_list)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for all tasks to complete
        item_queue.join()
        
        # Add sentinel values to stop worker threads
        for _ in range(num_workers):
            item_queue.put(None)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        progress_bar.close()
        
        # Write processed items back to the original file
        with open(qa_pairs_path, 'w', encoding='utf-8') as f_out:
            # Determine output format based on original format
            if qa_pairs_file.endswith('.jsonl'):
                # Write as JSONL
                for item in processed_items:
                    if item is not None:
                        # Handle original lines that couldn't be parsed
                        if "_original_line" in item:
                            f_out.write(item["_original_line"] + '\n')
                        else:
                            f_out.write(json.dumps(item) + '\n')
            else:
                # Write as JSON array with proper formatting
                valid_items = [item for item in processed_items if item is not None and "_original_line" not in item]
                json.dump(valid_items, f_out, indent=4)
        
        # Print statistics
        print("\n" + "="*80)
        print(f"LLM ANSWER GENERATION SUMMARY")
        print("="*80)
        print(f"Total items processed: {stats['processed_count']} / {stats['total_items']}")
        print(f"LLMs configured: {len(llm_list)}")
        print(f"LLMs enabled: {len([llm for llm in llm_list if llm.get('use_LLM', False)])}")
        print("\nLLM Response Statistics:")
        for field_name, count in sorted(stats['llm_responses'].items()):
            print(f"  {field_name}: {count} responses ({count/stats['total_items']*100:.1f}%)")
        print("\nLLM Usage Statistics:")
        for field_name in sorted(set(list(stats['llm_used'].keys()) + list(stats['llm_skipped'].keys()))):
            used_count = stats['llm_used'].get(field_name, 0)
            skipped_count = stats['llm_skipped'].get(field_name, 0)
            status = "ENABLED" if used_count > 0 else "DISABLED"
            print(f"  {field_name}: {status} (processed: {used_count}, skipped: {skipped_count})")
        print(f"Results written back to: {extract_path_from_pho(qa_pairs_path)}")
    
    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        main_log_fd.close()
        print(f"LLM answer generation complete. See {extract_path_from_pho(main_log_file)} for details.")
    
    print("\nLLM answer generation complete.")
    
def wrap_text(text, width=60):
    """Wrap text to specified width"""
    if isinstance(text, str) and len(text) > width:
        return '\n'.join(textwrap.wrap(text, width=width))
    return str(text)

def print_stats(stats_data, stats_file, max_chars_per_line=80):
    """
    Print statistics in a formatted table similar to print_new_verification_format
    
    Args:
        stats_data (dict): Dictionary containing all computed statistics
        stats_file (str): File to write the statistics to
        max_chars_per_line (int): Maximum characters per line before wrapping
    """    
    output = StringIO()
    
    def write_line(text=""):
        output.write(text + "\n")
        print(text)
    
    def wrap_text_local(text, width=max_chars_per_line):
        """Wrap text to specified width, preserving line breaks between items."""
        if not text:
            return text
            
        # Split by existing line breaks (between different items)
        lines = text.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if line.strip():  # Only wrap non-empty lines
                # Wrap each line individually
                wrapped = textwrap.fill(line, width=width, 
                                      break_long_words=False, 
                                      break_on_hyphens=False)
                wrapped_lines.append(wrapped)
            else:
                wrapped_lines.append(line)
        
        return '\n'.join(wrapped_lines)
    
    write_line("=" * 120)
    write_line("REWARD FUNCTION EVALUATION STATISTICS")
    write_line("=" * 120)
    
    # Get reward function names for headers
    rf_names = [rf_name for rf_name, _ in stats_data['reward_functions']]
    
    # Create main table data
    table_data = []
    
    # Add header row with reward function names
    header_row = ["Categories"]
    for rf_name in rf_names:
        header_row.append(f"{rf_name}")
    
    # Add difference column if we have exactly 2 reward functions
    if len(rf_names) == 2:
        header_row.append(f"Diff ({rf_names[1]} - {rf_names[0]})")
    
    table_data.append(header_row)
    
    # Section 1: Overall LLM Accuracy
    llm_rf1_list = []
    llm_rf2_list = []
    llm_diff_list = []
    
    for llm_name in sorted(stats_data['llms']):
        accuracies = []
        for rf_name in rf_names:
            accuracy = stats_data['overall_accuracy'][llm_name][rf_name]
            accuracies.append(accuracy)
            
        if len(rf_names) >= 1:
            llm_rf1_list.append(f"{llm_name}: {accuracies[0]:.3f}")
        if len(rf_names) >= 2:
            llm_rf2_list.append(f"{llm_name}: {accuracies[1]:.3f}")
            diff = accuracies[1] - accuracies[0]
            llm_diff_list.append(f"{llm_name}: {diff:+.3f}")
    
    # Add LLM accuracy row
    llm_row = ["LLM ACCURACY:"]
    if llm_rf1_list:
        llm_row.append(wrap_text_local("\n".join(llm_rf1_list)))
    if llm_rf2_list:
        llm_row.append(wrap_text_local("\n".join(llm_rf2_list)))
    if llm_diff_list and len(rf_names) == 2:
        llm_row.append(wrap_text_local("\n".join(llm_diff_list)))
    
    table_data.append(llm_row)
    
    # Section 2: Answer Types
    answer_type_rf1_list = []
    answer_type_rf2_list = []
    answer_type_diff_list = []
    
    for answer_type in sorted(stats_data['answer_types']):
        # Calculate average accuracy across all LLMs for this answer type
        avg_accuracies = []
        for rf_name in rf_names:
            total_accuracy = sum(stats_data['accuracy_by_type'][llm_name][answer_type][rf_name] 
                               for llm_name in stats_data['llms'])
            avg_accuracy = total_accuracy / len(stats_data['llms']) if stats_data['llms'] else 0.0
            avg_accuracies.append(avg_accuracy)
        
        if len(rf_names) >= 1:
            answer_type_rf1_list.append(f"{answer_type}: {avg_accuracies[0]:.3f}")
        if len(rf_names) >= 2:
            answer_type_rf2_list.append(f"{answer_type}: {avg_accuracies[1]:.3f}")
            diff = avg_accuracies[1] - avg_accuracies[0]
            answer_type_diff_list.append(f"{answer_type}: {diff:+.3f}")
    
    # Add answer types row
    answer_type_row = ["ANSWER TYPES:"]
    if answer_type_rf1_list:
        answer_type_row.append(wrap_text_local("\n".join(answer_type_rf1_list)))
    if answer_type_rf2_list:
        answer_type_row.append(wrap_text_local("\n".join(answer_type_rf2_list)))
    if answer_type_diff_list and len(rf_names) == 2:
        answer_type_row.append(wrap_text_local("\n".join(answer_type_diff_list)))
    
    table_data.append(answer_type_row)
    
    # Section 3: Sources (Top 10 by sample count)
    source_counts = {}
    for source in stats_data['sources']:
        total_count = sum(stats_data['accuracy_by_source'][llm_name][source]['count'] 
                         for llm_name in stats_data['llms'])
        source_counts[source] = total_count
    
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    source_rf1_list = []
    source_rf2_list = []
    source_diff_list = []
    
    for source, count in top_sources:
        # Calculate average accuracy across all LLMs for this source
        avg_accuracies = []
        for rf_name in rf_names:
            total_accuracy = sum(stats_data['accuracy_by_source'][llm_name][source][rf_name] 
                               for llm_name in stats_data['llms'])
            avg_accuracy = total_accuracy / len(stats_data['llms']) if stats_data['llms'] else 0.0
            avg_accuracies.append(avg_accuracy)
        
        if len(rf_names) >= 1:
            source_rf1_list.append(f"Source {source} (n={count}): {avg_accuracies[0]:.3f}")
        if len(rf_names) >= 2:
            source_rf2_list.append(f"Source {source} (n={count}): {avg_accuracies[1]:.3f}")
            diff = avg_accuracies[1] - avg_accuracies[0]
            source_diff_list.append(f"Source {source} (n={count}): {diff:+.3f}")
    
    # Add sources row
    source_row = ["TOP SOURCES:"]
    if source_rf1_list:
        source_row.append(wrap_text_local("\n".join(source_rf1_list)))
    if source_rf2_list:
        source_row.append(wrap_text_local("\n".join(source_rf2_list)))
    if source_diff_list and len(rf_names) == 2:
        source_row.append(wrap_text_local("\n".join(source_diff_list)))
    
    table_data.append(source_row)
    
    # Print the main table
    table = tabulate(
        table_data,
        headers="firstrow",
        tablefmt="grid",
        stralign="left"
    )
    write_line(table)
    
    # Summary Statistics
    write_line("\n" + "="*120)
    write_line("SUMMARY STATISTICS")
    write_line("="*120)
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Questions Evaluated", f"{stats_data['total_questions']}"],
        ["Total LLMs", f"{len(stats_data['llms'])}"],
        ["Total Reward Functions", f"{len(stats_data['reward_functions'])}"],
        ["Answer Types", f"{', '.join(sorted(stats_data['answer_types']))}"],
        ["Total Sources", f"{len(stats_data['sources'])}"],
        ["LLMs Evaluated", f"{', '.join(sorted(stats_data['llms']))}"],
        ["Reward Functions", f"{', '.join(rf_names)}"]
    ]
    
    summary_table = tabulate(
        summary_data,
        headers="firstrow",
        tablefmt="grid",
        stralign="left"
    )
    write_line(summary_table)
    
    # Detailed breakdown by LLM and Answer Type
    write_line("\n" + "="*120)
    write_line("DETAILED BREAKDOWN BY LLM AND ANSWER TYPE")
    write_line("="*120)
    
    for llm_name in sorted(stats_data['llms']):
        write_line(f"\nLLM: {llm_name}")
        write_line("-" * 60)
        
        detailed_data = [["Answer Type"] + rf_names]
        if len(rf_names) == 2:
            detailed_data[0].append(f"Diff ({rf_names[1]} - {rf_names[0]})")
        
        for answer_type in sorted(stats_data['answer_types']):
            row = [answer_type]
            accuracies = []
            for rf_name in rf_names:
                accuracy = stats_data['accuracy_by_type'][llm_name][answer_type][rf_name]
                row.append(f"{accuracy:.3f}")
                accuracies.append(accuracy)
            
            if len(accuracies) == 2:
                diff = accuracies[1] - accuracies[0]
                row.append(f"{diff:+.3f}")
            
            detailed_data.append(row)
        
        detailed_table = tabulate(
            detailed_data,
            headers="firstrow",
            tablefmt="grid",
            stralign="left"
        )
        write_line(detailed_table)
    
    # Accuracy per Answer Type per Source
    write_line("\n" + "="*120)
    write_line("ACCURACY PER ANSWER TYPE PER SOURCE")
    write_line("="*120)
    
    for answer_type in sorted(stats_data['answer_types']):
        write_line(f"\nAnswer Type: {answer_type.upper()}")
        write_line("-" * 80)
        
        # Create table for this answer type
        type_source_data = [["Source", "Count"] + rf_names]
        if len(rf_names) == 2:
            type_source_data[0].append(f"Diff ({rf_names[1]} - {rf_names[0]})")
        
        # Collect data for this answer type across all sources
        type_source_combinations = {}
        
        # Find all type-source combinations for this answer type
        for llm_name in stats_data['llms']:
            for type_source_key in stats_data['accuracy_by_type_source'][llm_name]:
                if type_source_key.startswith(f"{answer_type}_"):
                    source = type_source_key[len(f"{answer_type}_"):]
                    if source not in type_source_combinations:
                        type_source_combinations[source] = {
                            'counts': [],
                            'accuracies': {rf_name: [] for rf_name, _ in stats_data['reward_functions']}
                        }
                    
                    # Get count and accuracies for this LLM
                    count = stats_data['accuracy_by_type_source'][llm_name][type_source_key]['count']
                    type_source_combinations[source]['counts'].append(count)
                    
                    for rf_name, _ in stats_data['reward_functions']:
                        accuracy = stats_data['accuracy_by_type_source'][llm_name][type_source_key][rf_name]
                        type_source_combinations[source]['accuracies'][rf_name].append(accuracy)
        
        # Create rows for each source with data
        sources_with_data = []
        for source, data in type_source_combinations.items():
            if data['counts']:
                # Calculate average count and accuracies across LLMs
                avg_count = sum(data['counts']) / len(data['counts'])
                total_count = sum(data['counts'])  # Total across all LLMs
                
                avg_accuracies = []
                for rf_name, _ in stats_data['reward_functions']:
                    if data['accuracies'][rf_name]:
                        avg_accuracy = sum(data['accuracies'][rf_name]) / len(data['accuracies'][rf_name])
                        avg_accuracies.append(avg_accuracy)
                    else:
                        avg_accuracies.append(0.0)
                
                # Create row
                row = [f"Source {source}", f"{int(total_count)}"]
                for accuracy in avg_accuracies:
                    row.append(f"{accuracy:.3f}")
                
                if len(avg_accuracies) == 2:
                    diff = avg_accuracies[1] - avg_accuracies[0]
                    row.append(f"{diff:+.3f}")
                
                sources_with_data.append((source, total_count, row))
        
        # Sort by total count (descending) and add to table
        sources_with_data.sort(key=lambda x: x[1], reverse=True)
        for _, _, row in sources_with_data:
            type_source_data.append(row)
        
        if len(type_source_data) > 1:  # Only show table if there's data
            type_source_table = tabulate(
                type_source_data,
                headers="firstrow",
                tablefmt="grid",
                stralign="left"
            )
            write_line(type_source_table)
        else:
            write_line(f"No data available for answer type: {answer_type}")
    
    # Write to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(script_dir, stats_file)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
    
    write_line(f"\nStatistics written to: {extract_path_from_pho(stats_path)}")
    
def compute_stats(qa_pairs_file, stats_file, answer_types_to_include):
    '''
    Computes the accuracies of each LLM's based on its answers for each reward 
    function. Also computes most in depth stats for each reward function such
    as accuracy per answer type, per source, etc.

    Args:
        qa_pairs_file (str): The file to read the sample from.
        stats_file (str): The file to write the results to.

    Returns:
        None - the results are written to the stats file.
    '''
    # Define reward functions as tuples of (name, callable)
    reward_functions = [
        ("Original Reward Function", compute_score_combined_original),
        ("Math Verify Reward Function", compute_score_combined_updated),
    ]
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qa_pairs_path = os.path.join(script_dir, qa_pairs_file)
    
    if not os.path.exists(qa_pairs_path):
        print(f"Error: QA pairs file not found: {extract_path_from_pho(qa_pairs_path)}")
        return
    
    print(f"Computing statistics for reward functions...")
    print(f"Reward functions: {[name for name, _ in reward_functions]}")
    
    # Load QA pairs data
    all_items = []
    with open(qa_pairs_path, 'r', encoding='utf-8') as f:
        try:
            all_items = json.load(f)
        except json.JSONDecodeError:
            # If it's not a JSON array, try reading line by line (JSONL format)
            f.seek(0)
            for line in f:
                try:
                    all_items.append(json.loads(line))
                except:
                    continue
    
    if not all_items:
        print("No data found in QA pairs file")
        return
    
    print(f"Loaded {len(all_items)} QA pairs")
    
    filtered_items = []
    for item in all_items:
        if item.get('answerType', 'uncategorizable') in answer_types_to_include:
            filtered_items.append(item)
    
    all_items = filtered_items
    print(f"Filtered to {len(all_items)} QA pairs")
    
    # Find all LLMs that have predicted answers
    llm_pattern = re.compile(r'LLM_(\d{2})_predicted_answer')
    llms_found = set()
    
    for item in all_items:
        if isinstance(item, dict):
            for key in item.keys():
                match = llm_pattern.match(key)
                if match:
                    llms_found.add(key)
    
    llms_found = sorted(list(llms_found))
    print(f"Found LLMs: {llms_found}")
    
    if not llms_found:
        print("No LLM predicted answers found in the data")
        return
    
    # Initialize statistics structure
    stats_data = {
        'reward_functions': reward_functions,
        'llms': llms_found,
        'answer_types': set(),
        'sources': set(),
        'total_questions': len(all_items),
        'overall_accuracy': {},
        'accuracy_by_type': {},
        'accuracy_by_source': {},
        'accuracy_by_type_source': {}  # New: accuracy by answer type and source combination
    }
    
    # Initialize nested dictionaries
    for llm_name in llms_found:
        stats_data['overall_accuracy'][llm_name] = {}
        stats_data['accuracy_by_type'][llm_name] = {}
        stats_data['accuracy_by_source'][llm_name] = {}
        stats_data['accuracy_by_type_source'][llm_name] = {}
        
        for rf_name, _ in reward_functions:
            stats_data['overall_accuracy'][llm_name][rf_name] = 0.0
    
    # Process each item and compute statistics
    valid_items = 0
    
    for item in all_items:
        if not isinstance(item, dict):
            continue
            
        # Get ground truth answer and metadata
        answer_type = item.get('answerType', 'uncategorizable')
        if answer_type == 'numerical':
            ground_truth = item.get('numerical_answer', item.get('standardized_answer', item.get('answer', '')))
        else:
            ground_truth = item.get('standardized_answer', item.get('answer', ''))
        # Try both 'source' and 'source_number' fields, fallback to 'unknown'
        source = item.get('source', item.get('source_number', 'unknown'))
        
        if not ground_truth:
            continue
            
        valid_items += 1
        stats_data['answer_types'].add(answer_type)
        stats_data['sources'].add(source)
        
        # Initialize nested structures for this answer type and source
        for llm_name in llms_found:
            if answer_type not in stats_data['accuracy_by_type'][llm_name]:
                stats_data['accuracy_by_type'][llm_name][answer_type] = {}
                for rf_name, _ in reward_functions:
                    stats_data['accuracy_by_type'][llm_name][answer_type][rf_name] = []
            
            if source not in stats_data['accuracy_by_source'][llm_name]:
                stats_data['accuracy_by_source'][llm_name][source] = {}
                for rf_name, _ in reward_functions:
                    stats_data['accuracy_by_source'][llm_name][source][rf_name] = []
            
            # Initialize type-source combination
            type_source_key = f"{answer_type}_{source}"
            if type_source_key not in stats_data['accuracy_by_type_source'][llm_name]:
                stats_data['accuracy_by_type_source'][llm_name][type_source_key] = {}
                for rf_name, _ in reward_functions:
                    stats_data['accuracy_by_type_source'][llm_name][type_source_key][rf_name] = []
        
        # Evaluate each LLM's answer with each reward function
        for llm_name in llms_found:
            predicted_answer = item.get(llm_name, '')
            
            for rf_name, rf_function in reward_functions:
                # Call the reward function
                result = rf_function(predicted_answer, ground_truth, answer_type)
                
                # Extract score from result (handle both dict and numeric returns)
                if isinstance(result, dict):
                    score = result.get('score', 0.0)
                else:
                    score = float(result) if result is not None else 0.0
                
                # Add to overall accuracy
                stats_data['overall_accuracy'][llm_name][rf_name] += score
                
                # Add to type-specific accuracy
                stats_data['accuracy_by_type'][llm_name][answer_type][rf_name].append(score)
                
                # Add to source-specific accuracy
                stats_data['accuracy_by_source'][llm_name][source][rf_name].append(score)
                
                # Add to type-source combination accuracy
                type_source_key = f"{answer_type}_{source}"
                stats_data['accuracy_by_type_source'][llm_name][type_source_key][rf_name].append(score)
    
    # Convert lists to averages and counts
    stats_data['total_questions'] = valid_items
    
    # Overall accuracy
    for llm_name in llms_found:
        for rf_name, _ in reward_functions:
            if valid_items > 0:
                stats_data['overall_accuracy'][llm_name][rf_name] /= valid_items
    
    # Type-specific accuracy
    for llm_name in llms_found:
        for answer_type in stats_data['answer_types']:
            for rf_name, _ in reward_functions:
                scores = stats_data['accuracy_by_type'][llm_name][answer_type][rf_name]
                stats_data['accuracy_by_type'][llm_name][answer_type][rf_name] = sum(scores) / len(scores) if scores else 0.0
    
    # Source-specific accuracy
    for llm_name in llms_found:
        for source in stats_data['sources']:
            for rf_name, _ in reward_functions:
                scores = stats_data['accuracy_by_source'][llm_name][source][rf_name]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                stats_data['accuracy_by_source'][llm_name][source][rf_name] = avg_score
                # Also store count for reference
                stats_data['accuracy_by_source'][llm_name][source]['count'] = len(scores)
    
    # Type-source combination accuracy
    for llm_name in llms_found:
        for type_source_key in stats_data['accuracy_by_type_source'][llm_name]:
            for rf_name, _ in reward_functions:
                scores = stats_data['accuracy_by_type_source'][llm_name][type_source_key][rf_name]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                stats_data['accuracy_by_type_source'][llm_name][type_source_key][rf_name] = avg_score
                # Also store count for reference
                stats_data['accuracy_by_type_source'][llm_name][type_source_key]['count'] = len(scores)
    
    # Convert sets to sorted lists for consistent output
    stats_data['answer_types'] = sorted(list(stats_data['answer_types']))
    stats_data['sources'] = sorted(list(stats_data['sources']))
    
    print(f"Processed {valid_items} valid items")
    print(f"Answer types found: {stats_data['answer_types']}")
    print(f"Sources found: {len(stats_data['sources'])}")
    
    # Print and save statistics
    print_stats(stats_data, stats_file)

if __name__ == "__main__":
    qa_pairs_file = "qa_pairs_with_predicted_answers_sample.json"
    stats_file = "Log-Files/Reward_Function_Stats/Reward_Functions_Stats.txt"
    num_questions_in_sample = 500

    # Generate a new question and answer sample.
    # generate_new_question_and_answer_sample(num_questions_in_sample, qa_pairs_file)
    
    # Test the reward functions and compute their stats.
    # generate_answers_for_sample(qa_pairs_file)
    answer_types_to_include = [
        "numerical",
        "letter",
        "equation"
    ]
    compute_stats(qa_pairs_file, stats_file, answer_types_to_include)
