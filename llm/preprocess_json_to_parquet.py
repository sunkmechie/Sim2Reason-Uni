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
"""
Preprocess qa pair json files to parquet format for LLM evaluation.
"""
import ipdb
from tqdm import tqdm
import pandas as pd
st = ipdb.set_trace
import os
import json
import argparse
import random
import re
from transformers import AutoTokenizer
from typing import Dict


def index_json_files_relative(root_dir: str) -> Dict[str, str]:
    """
    Returns a dict mapping each JSON filename to a relative path
    that starts with root_dir (e.g., "/home/mprabhud/datasets/physics_sim_data/PHO_repo_datasets/llm/Types_1-5_Full_Dataset/Type_2_Sources/002/002-002A-2024-USAPhO-Exam_solutions-001_responses.json").
    """
    root_dir = os.path.normpath(root_dir)  # remove trailing slash if any
    json_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        
        for fname in filenames:
            if fname.lower().endswith('.json'):
                full_path = os.path.join(dirpath, fname)
                # Keep the relative path starting from current directory (which includes root_dir)
                rel_path = os.path.relpath(full_path, start=".")
                json_map[fname] = rel_path
    return json_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=os.environ["PHO_DATA"])
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--json_names', nargs='+', default=['Synthetic_problems.json'], help='Name(s) of the json file')
    parser.add_argument('--disable_train', action='store_true', help='Disable processing of the train set')
    parser.add_argument('--disable_test', action='store_true', help='Disable processing of the test set')
    parser.add_argument('--data_source', default='math_p', help='Data source')
    parser.add_argument('--data_version', default='v_testing', help='Data version')
    parser.add_argument('--no_extra_instruction', action='store_true', help='do not add instruction_following: Let\'s think step by step and output the final answer within \\boxed{}')
    parser.add_argument('--translated', action='store_true', help='translated the problems to English')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--for_sft', action='store_true', help='whether to preprocess the data for SFT or RL')

    args = parser.parse_args()
    # st()
    
    text_key = 'text' if not args.translated else 'translated_text'
    # args.local_dir = os.path.join(args.local_dir, args.data_version)
    train_data = []
    for json_name in args.json_names:
        with open(os.path.join(args.local_dir, args.data_version, json_name), 'r') as f:
            _data = json.load(f)
            print("[DATA] Loaded data from: ", os.path.join(args.local_dir, args.data_version, json_name), "num examples: ", len(_data))
            train_data.extend(_data)
    # st()
    # remove duplicates by using pandas
    df = pd.DataFrame(train_data)
    print('Original length: ', len(df))

    # count the number of samples according to the data_source
    # duplicates = df[df.duplicated(subset=[text_key])]
    # dupe_texts = duplicates[text_key].unique()
    # Remove duplicate rows based on both text content and descriptions (if present)
    # This ensures we don't have multiple identical questions with the same descriptions in our dataset
    
    if not args.for_sft:
        duplicate_columns = [text_key]

        if 'descriptions' in df.columns:
            df['descriptions_tuple'] = df['descriptions'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
            duplicate_columns.append('descriptions_tuple')

        df = df.drop_duplicates(subset=duplicate_columns)
        if 'descriptions_tuple' in df.columns:
            df.drop(columns='descriptions_tuple', inplace=True)

        print('After removing duplicates: ', len(df))
    else:
        # Filter overlong prompts
        k = 4096
        def filter_correct_cot(row):
            return [i for i in row['correct_cot'] if len(row['cot'][i]) < k]
    
        print("Num correct COT:", sum([len(row['correct_cot']) for _, row in df.iterrows()]))

        df['correct_cot'] = df.apply(filter_correct_cot, axis=1)

        print("Num correct COT after removing overlong prompts:", sum([len(row['correct_cot']) for _, row in df.iterrows()]))
    # exit()
    
    train_data = df.to_dict(orient='records')
    random.shuffle(train_data)

    if args.disable_train:
        test_data = train_data
    else:
        # Do 90-10 split
        train_size = int(0.9 * len(train_data))
        test_data = train_data[train_size:]
        train_data = train_data[:train_size]
        print("Number of train examples: ", len(train_data))
        print("Number of test examples: ", len(test_data))
    
    # if not args.disable_test:
    #     with open(os.path.join(args.local_dir, 'ipho_test.json'), 'r') as f:
    #         test_data = json.load(f)
        
    data_source = args.data_source
    numeric_instruction_following = " Let's think step by step and output the final answer within \\boxed{}. The final answer should be expressed using the International System of Units (SI) unless stated otherwise."
    symbolic_instruction_following = " Let's think step by step and output the final answer within \\boxed{}. Present the final answer in LaTeX format."    
    file_name_to_relative_path = index_json_files_relative(args.local_dir)

    def make_map_fn_rl(split):
        def process_fn(example, idx):
            question = example[text_key]
            # keep only one space if there are consecutive spaces
            question = re.sub(r'\s+', ' ', question).strip()
            answer = example['answer']
            if "math_p" in data_source:
                if example['is_symbolic']:
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = data_source + '_symbolic' if not args.translated else data_source + '_symbolic_translated'
                elif example.get('is_reverse', False):
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = data_source + '_reverse' if not args.translated else data_source + '_reverse_translated'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = data_source + '_numeric' if not args.translated else data_source + '_numeric_translated'
            elif data_source == "hcv":
                if "answerType" in example and example['answerType'] == "equation" and "error" in example and not example['error']:
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = 'hcv_symbolic'
                elif "answerType" in example and example['answerType'] == "numerical":
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'hcv_numeric'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'hcv'
                if "description" in example and str(example['description']) != "" and str(example['description']).lower() != "nan":
                    description = str(example['description'])
                    question = f"Here is a description of the image: {description}. \nBased on the above description, answer the following question.\n{question}" 
            elif data_source == "ipho":
                if "is_symbolic" in example and example['is_symbolic'] and "error" in example and not example['error']:
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = 'ipho_symbolic'
                elif "is_numerical" in example and example['is_numerical']:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'ipho_numeric'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'ipho'
                if "description" in example and str(example['description']) != "" and str(example['description']).lower() != "nan":
                    description = str(example['description'])
                    question = f"Here is a description of the image: {description}. \nBased on the above description, answer the following question.\n{question}"
            elif data_source == "Types_1-5_Full_Dataset":
                try:
                    if example.get('answerType', '') == "letter" or example.get('answerType', '') == "equation":
                        answer = example['standardized_answer']
                    elif example.get('answerType', '') == "numerical":
                        answer = example.get('numerical_answer', '')
                    else:
                        answer = example.get("standardized_answer", example.get("answer", ""))
                except Exception as e:
                    pass

                if example.get('answerType', '') == "equation":
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = f"Types_1-5_Full_Dataset_equation_{example.get('source', example.get('source_number', '000'))}"
                elif example.get('answerType', '') == "numerical":
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = f'Types_1-5_Full_Dataset_numerical_{example.get("source", example.get("source_number", "000"))}'
                elif example.get('answerType', '') == "letter":
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = f'Types_1-5_Full_Dataset_letter_{example.get("source", example.get("source_number", "000"))}'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = f'Types_1-5_Full_Dataset_other_{example.get("source", example.get("source_number", "000"))}'
                if "descriptions" in example and example['descriptions'] and isinstance(example['descriptions'], list):
                    description = "\n".join(example['descriptions'])
                    question = f"Here is a description of the image or context or answer choices: {description}. \nBased on the above description, answer the following question.\n{question}"
        
            if 'reference' in example.keys():
                if "math_p" in data_source_type:
                    data_type = example['reference']    # e.g. "/datasets/physics_sim_data/v6/AdvancedInclinedPlaneFriction/scene_938/scene_output"
                    data_type = data_type.split('/')[-3]  # e.g. "AdvancedInclinedPlaneFriction"
                elif 'hcv' in data_source_type:
                    data_type = example['chapter']
                else:
                    data_type = 'unknown'
            else:
                data_type = 'unknown'

            data = {
                "scene_type": data_type,
                "data_source": data_source_type,
                "prompt": [{
                    "role": "user", 
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(answer)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer_type': example.get('answerType', None),
                    'file_name': file_name_to_relative_path.get(example.get('file_name', None), None),
                    'json_index': example.get('idx', None)
                }
            }
            if "math_p" in data_source_type and "symbolic" in data_source_type:
                data['extra_info']['given_variable_mapping'] = example['given_variable_mapping']
            else:
                data['extra_info']['given_variable_mapping'] = None
            return data
        return process_fn

    def make_map_fn_sft(split):
        def process_fn(example, idx):
            question = example[text_key]
            # keep only one space if there are consecutive spaces
            question = re.sub(r'\s+', ' ', question).strip()
            answer = example['answer']
            if "math_p" in data_source:
                if example['is_symbolic']:
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = data_source + '_symbolic' if not args.translated else data_source + '_symbolic_translated'
                elif example.get('is_reverse', False):
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = data_source + '_reverse' if not args.translated else data_source + '_reverse_translated'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = data_source + '_numeric' if not args.translated else data_source + '_numeric_translated'
            elif data_source == "hcv":
                if "answerType" in example and example['answerType'] == "equation" and "error" in example and not example['error']:
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = 'hcv_symbolic'
                elif "answerType" in example and example['answerType'] == "numerical":
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'hcv_numeric'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'hcv'
                if "description" in example and str(example['description']) != "" and str(example['description']).lower() != "nan":
                    description = str(example['description'])
                    question = f"Here is a description of the image: {description}. \nBased on the above description, answer the following question.\n{question}" 
            elif data_source == "ipho":
                if "is_symbolic" in example and example['is_symbolic'] and "error" in example and not example['error']:
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = 'ipho_symbolic'
                elif "is_numerical" in example and example['is_numerical']:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'ipho_numeric'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = 'ipho'
                if "description" in example and str(example['description']) != "" and str(example['description']).lower() != "nan":
                    description = str(example['description'])
                    question = f"Here is a description of the image: {description}. \nBased on the above description, answer the following question.\n{question}"
            elif data_source == "Types_1-5_Full_Dataset":
                try:
                    if example.get('answerType', '') == "letter" or example.get('answerType', '') == "equation":
                        answer = example['standardized_answer']
                    elif example.get('answerType', '') == "numerical":
                        answer = example.get('numerical_answer', '')
                    else:
                        answer = example.get("standardized_answer", example.get("answer", ""))
                except Exception as e:
                    pass

                if example.get('answerType', '') == "equation":
                    question = question if args.no_extra_instruction else question + symbolic_instruction_following
                    data_source_type = f"Types_1-5_Full_Dataset_equation_{example.get('source', example.get('source_number', '000'))}"
                elif example.get('answerType', '') == "numerical":
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = f'Types_1-5_Full_Dataset_numerical_{example.get("source", example.get("source_number", "000"))}'
                elif example.get('answerType', '') == "letter":
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = f'Types_1-5_Full_Dataset_letter_{example.get("source", example.get("source_number", "000"))}'
                else:
                    question = question if args.no_extra_instruction else question + numeric_instruction_following
                    data_source_type = f'Types_1-5_Full_Dataset_other_{example.get("source", example.get("source_number", "000"))}'
                if "descriptions" in example and example['descriptions'] and isinstance(example['descriptions'], list):
                    description = "\n".join(example['descriptions'])
                    question = f"Here is a description of the image or context or answer choices: {description}. \nBased on the above description, answer the following question.\n{question}"
        
            if 'reference' in example.keys():
                if "math_p" in data_source_type:
                    data_type = example['reference']    # e.g. "/datasets/physics_sim_data/v6/AdvancedInclinedPlaneFriction/scene_938/scene_output"
                    data_type = data_type.split('/')[-3]  # e.g. "AdvancedInclinedPlaneFriction"
                elif 'hcv' in data_source_type:
                    data_type = example['chapter']
                else:
                    data_type = 'unknown'
            else:
                data_type = 'unknown'

            correct_cot = [example['cot'][k] for k in example['correct_cot']]

            data = [{
                "scene_type": data_type,
                "data_source": data_source_type,
                "question": question,
                "ability": "math",
                "answer": str(cot),
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer_type': example.get('answerType', None),
                    'file_name': file_name_to_relative_path.get(example.get('file_name', None), None),
                    'json_index': example.get('idx', None)
                }
            } for cot in correct_cot]
            if "math_p" in data_source_type and "symbolic" in data_source_type:
                for row in data:
                    row['extra_info']['given_variable_mapping'] = example['given_variable_mapping']
                # data['extra_info']['given_variable_mapping'] = example['given_variable_mapping']
            else:
                for row in data:
                    row['extra_info']['given_variable_mapping'] = None
            return data
        return process_fn
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B", trust_remote_code=True)

    # Initialize variables
    train_valid_data = []
    test_valid_data = []
    
    if not args.disable_train:
        # Process train data
        train_valid_data = [example for example in train_data if text_key in example.keys()]
        print("Number of valid train examples: ", len(train_valid_data))

        if len(train_valid_data) > 0:
            train_token_lengths = []
            for example in tqdm(train_valid_data, desc="Tokenizing train examples"):
                token_length = len(tokenizer.encode(example[text_key]))
                train_token_lengths.append(token_length)

            train_valid_data = [example for i, example in enumerate(train_valid_data) if train_token_lengths[i] <= 1900]
            train_token_lengths = [length for length in train_token_lengths if length <= 1900]

            print("Number of valid train examples after filtering: ", len(train_valid_data))
            if len(train_token_lengths) > 0:
                print("Train average token length:", sum(train_token_lengths) / len(train_token_lengths))
                print("Train max token length:", max(train_token_lengths))
                print("Train min token length:", min(train_token_lengths))
            else:
                print("No valid train examples found")
        else:
            print("No valid train examples found")

    # Process test data
    if not args.disable_test:
        test_valid_data = [example for example in test_data if text_key in example.keys()]
        print("Number of valid test examples: ", len(test_valid_data))

        if len(test_valid_data) > 0:
            test_token_lengths = []
            for example in tqdm(test_valid_data, desc="Tokenizing test examples"):
                token_length = len(tokenizer.encode(example[text_key]))
                test_token_lengths.append(token_length)

            test_valid_data = [example for i, example in enumerate(test_valid_data) if test_token_lengths[i] <= 1900]
            test_token_lengths = [length for length in test_token_lengths if length <= 1900]

            print("Number of valid test examples after filtering: ", len(test_valid_data))
            if len(test_token_lengths) > 0:
                print("Test average token length:", sum(test_token_lengths) / len(test_token_lengths))
                print("Test max token length:", max(test_token_lengths))
                print("Test min token length:", min(test_token_lengths))
            else:
                print("No valid test examples found")
        else:
            print("No valid test examples found")

    map_fn = make_map_fn_sft if args.for_sft else make_map_fn_rl
    if not args.disable_train and len(train_valid_data) > 0:
        # Create final datasets
        train_dataset = [map_fn('train')(example, idx) 
                        for idx, example in enumerate(train_valid_data)]
        if args.for_sft:
            # Flatten the list of lists for SFT
            train_dataset = sum(train_dataset, [])
    if not args.disable_test and len(test_valid_data) > 0:
        test_dataset = [map_fn('test')(example, idx)
                    for idx, example in enumerate(test_valid_data)]
        if args.for_sft:
            # Flatten the list of lists for SFT
            test_dataset = sum(test_dataset, [])

    local_dir = args.local_dir
    local_dir = os.path.join(local_dir, args.data_version)
    hdfs_dir = args.hdfs_dir
    if not args.disable_train and len(train_valid_data) > 0:
        train_df = pd.DataFrame(train_dataset)
    if not args.disable_test and len(test_valid_data) > 0:
        test_df = pd.DataFrame(test_dataset)

    file_suffix = "_sft" if args.for_sft else "_rl"
    
    if not args.disable_train and len(train_valid_data) > 0:
        if args.translated:
            train_df.to_parquet(os.path.join(local_dir, f'train_translated_{args.data_version}{file_suffix}.parquet'))
        else:
            train_df.to_parquet(os.path.join(local_dir, f'train_{args.data_version}{file_suffix}.parquet'))
    if not args.disable_test and len(test_valid_data) > 0:
        if args.translated:
            test_df.to_parquet(os.path.join(local_dir, f'test_translated_{args.data_version}{file_suffix}.parquet'))
        else:
            test_df.to_parquet(os.path.join(local_dir, f'test_{args.data_version}{file_suffix}.parquet'))
    # Also save as JSON for easier inspection
    if not args.disable_train and len(train_valid_data) > 0:  
        if args.translated:
            train_df.to_json(os.path.join(local_dir, f'train_translated_{args.data_version}{file_suffix}.json'), orient='records', indent=2)
        else:
            train_df.to_json(os.path.join(local_dir, f'train_{args.data_version}{file_suffix}.json'), orient='records', indent=2)
    if not args.disable_test and len(test_valid_data) > 0:
        if args.translated:
            test_df.to_json(os.path.join(local_dir, f'test_translated_{args.data_version}{file_suffix}.json'), orient='records', indent=2)
        else:
            test_df.to_json(os.path.join(local_dir, f'test_{args.data_version}{file_suffix}.json'), orient='records', indent=2)

    print(f"local_dir: ", os.path.join(local_dir, f'test_{args.data_version}{file_suffix}.json'))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
