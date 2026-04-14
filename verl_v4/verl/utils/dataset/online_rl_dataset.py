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

from omegaconf import ListConfig, OmegaConf
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

from sim.scene_generator import SceneGenerator
from omegaconf import DictConfig
from sim.qa_gen_rule import data_gen
import datetime
import random
import sys
import psutil

def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class OnlineRLHFDataset(Dataset):
    """
    In this dataset, we would generate data on the fly.
    We will generate MuJoCo scenes and then generate Questions and Answers for the scenes.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=2048,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 main_cfg: DictConfig = None,
                 length: int = int(1e7)):

        self.qa_gen_cfg = main_cfg
        self.scene_generator_config = main_cfg.scene_generation

        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._length = length   # default is int(1e7), otherwise it is the number of generated data (let's say for validation)
        

    def __len__(self):
        return self._length

    
    def _make_map_fn(self, split):
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        def process_fn(example, idx):
            question = example['text']
            question = question + 'Assume acceleration due to gravity as 9.81 m/s^2, all strings inextensible, and all surfaces frictionless unless otherwise stated. ' + instruction_following
            answer = example['answer']
            solution = answer
            scene_type = example['scene_type']
            data = {
                "data_source": 'math_p',
                'scene_type': scene_type,
                "prompt": np.array([{
                    "role": "user", 
                    "content": question
                }]),
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        ''' Step 1: Generate Scene '''
        current_time = datetime.datetime.now().microsecond
        seed = current_time + item  # enrich the seed with item to make it unique
        random.seed(seed)
        subtype = random.choice(self.scene_generator_config['scene_types'])
        scene_generator = SceneGenerator(subtype=subtype, seed=datetime.datetime.now().microsecond + item)
        scene_yaml = scene_generator.generate_scene_yaml()


        ''' Step 2: Generate Question and Answer '''
        recorder_cfg = self.qa_gen_cfg.recorder
        Final_Problem_with_Answer = data_gen(scene_yaml, self.qa_gen_cfg, recorder_cfg, seed_offset=item)
        Final_Problem_with_Answer['scene_type'] = subtype
        ''' Step 3: Tokenize Question and Answer '''
        
        # # return Final_Problem_with_Answer
        # token_length = len(self.tokenizer.encode(Final_Problem_with_Answer['text']))
        # if token_length > self.max_prompt_length: # reproduce main_p.py
        #     return self.__getitem__(random.randint(0, len(self)))
        
        row_dict = self._make_map_fn('train')(Final_Problem_with_Answer, item)
        

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = chat[0]['content']
        # prompt_with_chat_template = chat

        model_inputs = self.tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation
            )
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self)))

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


def monitor_resources():
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    memory_info = psutil.virtual_memory()
    memory_usage_gb = memory_info.used / (1024 ** 3)
    print(f"Memory Usage: {memory_usage_gb:.2f} GB")

if __name__ == "__main__":
    cfg = OmegaConf.load('config/config.yaml')
    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs('Qwen/Qwen2.5-3B')

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    dataset = OnlineRLHFDataset(tokenizer, main_cfg=cfg)
    
    # import ipdb; ipdb.set_trace()
    dataloader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collate_fn,
                            num_workers=4,
                            prefetch_factor=1,
                            )

    print(len(dataloader))
    print("Number of CPU:", len(os.sched_getaffinity(0)))

    # test the dataloader
    # count time
    seconds_per_batch = [0.0]
    import time
    start_time = time.time()
    i = 0
    monitor_resources()
    for data in dataloader:
        monitor_resources()
        i += 1
        print(i)
        print(data)
        time.sleep(10)
        end_time = time.time()
        seconds_per_batch.append(end_time - start_time)
        print(f"Time taken: {end_time - start_time} seconds")
        if i >= 256:
            break
    import matplotlib.pyplot as plt
    plt.plot(seconds_per_batch)
    plt.savefig('seconds_per_batch.png')
    plt.close()