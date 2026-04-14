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

from collections import defaultdict

import torch
import ipdb
st = ipdb.set_trace
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        simple_eval=False,
        overlong_buffer_cfg=None,
        config=None,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.simple_eval = simple_eval
        self.config = config
        # st()

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False, reward_fn_name=None, config=None):
        """We will expand this function gradually based on the available datasets"""

        if config is None:
            config = self.config
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]


        if self.simple_eval:
            reward_tensor = torch.zeros(len(data.non_tensor_batch["responses"]), dtype=torch.float32)[:, None]
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
            

        already_print_data_sources = {}
        if config.actor_rollout_ref.rollout.val_kwargs.n >  1:
            data = data.repeat_except_responses(repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n)

        for i in range(len(data)):
            # print("DEBUG: Within DAPO reward manager i: ", i, " / ", len(data), flush=True)
            if config.debug_print:
                print("DEBUG: Within DAPO reward manager i: ", i, " / ", len(data))
            data_item = data[i]  # DataProtoItem

            if self.simple_eval:
                prompt_str = str(data_item.non_tensor_batch["raw_prompt"])
                response_str = data_item.non_tensor_batch["responses"]
            else:
                prompt_ids = data_item.batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                eos_token = self.tokenizer.eos_token
                if response_str.endswith(eos_token):
                    response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            if config.debug_print:
                print("DEBUG: Computing score for data_source", data_source)
            # st()

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                reward_fn_name=reward_fn_name,
                config=config,
            )
            # if result["score"] == -1:
            #     st()
            # st()

            if config.debug_print:
                print("DEBUG: Score computed")

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["acc"].append(score)

            reward = score

            if not self.simple_eval and self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)
            # st()
            if self.simple_eval:
                reward_tensor[i,-1] = reward
            else:
                reward_tensor[i, valid_response_length - 1] = reward
            # st()
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            if config.debug_print: print("DEBUG: Returning reward tensor and reward extra info")
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            if config.debug_print: print("DEBUG: Returning reward tensor")
            return reward_tensor
