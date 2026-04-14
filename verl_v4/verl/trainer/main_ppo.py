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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket
import ipdb
st = ipdb.set_trace

import hydra
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type

from importlib_metadata import version, PackageNotFoundError

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)        

    print(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    import os
    
    
    assert version("math-verify") == "0.7.0"
    assert version("antlr4-python3-runtime") == "4.11.0"
    assert version("latex2sympy2_extended") == "1.0.9"

    
    print("PATH", os.environ['PATH'])
    try:
        print("LD_LIBRARY_PATH", os.environ['LD_LIBRARY_PATH'])
    except:
        print("LD_LIBRARY_PATH not set.")

    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)        
        
        runtime_env["env_vars"]["TOKENIZERS_PARALLELISM"] = "true"
        runtime_env["env_vars"]["NCCL_DEBUG"] = "INFO"
        runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] = "WARN"

        runtime_env["env_vars"]["HYDRA_FULL_ERROR"] = "1"
        runtime_env["env_vars"]["VLLM_USE_V1"] = "1"

        hostname = socket.gethostname()
        if "mgx" in hostname or "gb" in hostname:
            runtime_env["env_vars"]["GLOO_SOCKET_IFNAME"] = "bond0"
        if "gb" in hostname:
            runtime_env["env_vars"]["NCCL_DEBUG"] = "INFO"

        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, reward_options=config.reward_model.get("reward_options", None), **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, reward_options=config.reward_model.get("reward_options", None), **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        
        if not config.get("run_eval_mode", False):
            # Start the training process.
            trainer.fit()
            return

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=trainer.config.trainer.project_name,
            experiment_name=trainer.config.trainer.experiment_name,
            default_backend=trainer.config.trainer.logger,
            config=OmegaConf.to_container(trainer.config, resolve=True),
        )

        if 'wandb' in logger.logger:
            trainer.config.trainer.default_local_dir = os.path.join(trainer.config.trainer.default_local_dir,logger.logger['wandb'].run.name)
        
        if not config.get("one_time_eval", False):
            # We evaluate pretrained model
            folder = '/'.join(config.trainer.resume_from_path.split("/")[:-1])
            sub_folders = os.listdir(folder)
            sub_folders = [f for f in sub_folders if f.startswith('global_step_') and os.path.isdir(os.path.join(folder, f))]
            
            sub_folders_sorted = sorted(sub_folders, key=lambda x: int(x.split('_')[-1]))

            outs = []
            for sub_folder in sub_folders_sorted:
                print(f"Resuming from {sub_folder}")
                trainer.config.trainer.resume_from_path = os.path.join(folder, sub_folder)
                metrics, step = trainer.run_eval_mode()

                print(f"Done validate, appending to local list")
                outs.append((step, metrics))

                print(f"Done appending to local list, logging to wandb now.")

                logger.log(data=metrics, step=step)

                print(f"Done logging in wandb")
            
        else:
            # We evaluate base model
            metrics, step = trainer.run_eval_mode()
            logger.log(data=metrics, step=step)
            print(f"Done logging in wandb")
        
        # flush
        logger.logger['wandb'].finish()

def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        is_train=is_train,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler

def create_rl_sampler_weighted(data_config, dataset):
    """
    Create a WeightedRandomSampler based on data_config.weights dict.
    Supports partial weights and normalizes unspecified keys.

    Args:
        data_config: dict with 'shuffle', 'seed', and optional 'weights' field:
                     weights = {
                         "data_source": {"A": 0.5},
                         "scene_type": {"urban": 2, "highway": 1}
                     }
        dataset: Dataset with 'data_source' and/or 'scene_type' fields per item.

    Returns:
        torch.utils.data.Sampler
    """
    import torch
    from torch.utils.data import SequentialSampler, WeightedRandomSampler
    from collections import defaultdict

    if not data_config.get("shuffle", False):
        return SequentialSampler(dataset)

    weights_cfg = data_config.get("weights", {})
    seed = data_config.get("seed", 1)

    def normalize_and_fill_weights(key_name, values_list, user_weights):
        """
        Fill in missing weights uniformly, normalize all weights to sum to 1.
        """
        unique_keys = sorted(set(values_list))
        user_weights = user_weights or {}
        user_weights = {k: v for k, v in user_weights.items() if k in unique_keys}
        
        # Extract user weights and mark unspecified keys
        specified_keys = set(user_weights.keys())
        unspecified_keys = [k for k in unique_keys if k not in specified_keys]

        # Sum of specified weights
        total_specified = sum(user_weights.get(k, 0.0) for k in specified_keys)

        # If total_specified == 0, fallback to uniform
        remaining_weight = max(1.0 - total_specified, 0.0)
        uniform_fill = remaining_weight / len(unspecified_keys) if unspecified_keys else 0.0

        # Combine weights
        final_weights = {
            k: user_weights.get(k, uniform_fill) for k in unique_keys
        }

        # Normalize so weights sum to 1 over all keys
        total = sum(final_weights.values())
        final_weights = {k: v / total if total > 0 else 1.0 / len(final_weights) for k, v in final_weights.items()}

        return final_weights

    # Extract data_source and scene_type per item
    # data_sources = [dataset[i]["data_source"] for i in range(len(dataset))]
    # scene_types = [dataset[i].get("scene_type", None) for i in range(len(dataset))]
    
    data_sources = dataset.dataframe["data_source"]
    if "scene_type" not in dataset.dataframe.features:
        scene_types = ["no_scene_type"] * len(dataset.dataframe)
    else:
        scene_types = dataset.dataframe["scene_type"]

    ds_weights = normalize_and_fill_weights("data_source", data_sources, weights_cfg.get("data_source"))
    st_weights = normalize_and_fill_weights("scene_type", [s for s in scene_types if s is not None], weights_cfg.get("scene_type"))
    # import ipdb
    # ipdb.set_trace()
    missing_ds = set(data_sources) - set(ds_weights)
    missing_st = set(scene_types) - set(st_weights)

    if missing_ds:
        raise ValueError(f"Missing weights for data_sources: {sorted(missing_ds)}")
    if missing_st:
        raise ValueError(f"Missing weights for scene_types: {sorted(missing_st)}")
    
    # Assign weight per sample
    sample_weights = []
    for ds, st in tqdm(zip(data_sources, scene_types), desc="Weighing samples", total=len(data_sources)):
        w_ds = ds_weights.get(ds, 0.0)
        w_st = st_weights.get(st, 0.0)
        sample_weights.append(w_ds * w_st)

    # Final normalization across samples
    total_weight = sum(sample_weights)
    sample_weights = [w / total_weight if total_weight > 0 else 1.0 / len(sample_weights) for w in sample_weights]

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator
    )


if __name__ == "__main__":
    main()
