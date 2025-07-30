# Copyright 2023 The Alpaca Team
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
import os
import io
import functools
import json
import time
import types
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import warnings
import transformers
import numpy as np
from typing import Callable, Optional, Sequence, Union
import log_utils

# Create directories if they don't exist
make_directories = functools.partial(os.makedirs, exist_ok=True)
logger = log_utils.get_logger(__name__)

def all_elements_equal(sequence: Sequence, 
                      comparator: Optional[Callable] = lambda x, y: x == y) -> bool:
    """Check if all elements in a sequence are equal using a comparator function"""
    return all(comparator(sequence[0], element) for element in sequence[1:])

def safe_zip(*sequences: Sequence) -> list:
    """Zip sequences safely after validating equal lengths"""
    if not sequences:
        return []
    if not all_elements_equal(sequences, lambda x, y: len(x) == len(y)):
        raise ValueError("All sequences must have the same length")
    return list(zip(*sequences))

def calculate_mean(values: Sequence[Union[int, float]]) -> Union[float, list]:
    """Calculate mean of numeric sequences"""
    means = [float(np.mean(seq)) for seq in values]
    return means[0] if len(values) == 1 else means

def open_file_for_reading(file_path: str, mode: str = "r") -> io.TextIOWrapper:
    """Open a file for reading, handling both paths and file objects"""
    if not isinstance(file_path, io.IOBase):
        return open(file_path, mode=mode)
    return file_path

def open_file_for_writing(file_path: str, mode: str = "w") -> io.TextIOWrapper:
    """Open a file for writing, creating directories if needed"""
    if not isinstance(file_path, io.IOBase):
        directory = os.path.dirname(file_path)
        if directory:
            make_directories(directory)
        return open(file_path, mode=mode)
    return file_path

def load_json(file_path: str) -> Union[dict, list]:
    """Load JSON data from a file"""
    with open_file_for_reading(file_path) as file:
        return json.load(file)

def save_json(data: Union[dict, list, str], 
             file_path: str, 
             indent: int = 4, 
             default: Callable = str) -> None:
    """Save data to a JSON file with proper formatting"""
    with open_file_for_writing(file_path) as file:
        if isinstance(data, (dict, list)):
            json.dump(data, file, indent=indent, default=default)
        elif isinstance(data, str):
            file.write(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

def to_json_string(data: Union[dict, list], 
                  indent: int = 4, 
                  default: Callable = str) -> str:
    """Convert data to a formatted JSON string"""
    return json.dumps(data, indent=indent, default=default)

def resize_token_embeddings(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    special_tokens: dict,
    jitter_new_embeddings: bool = False
) -> None:
    """Resize token embeddings and add special tokens"""
    tokenizer.add_special_tokens(special_tokens)
    target_size = len(tokenizer)
    current_size = model.get_input_embeddings().weight.size(0)
    
    if target_size <= current_size:
        return
    
    model.resize_token_embeddings(target_size)
    num_new_tokens = target_size - current_size
    
    # Initialize new embeddings
    with torch.inference_mode():
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        
        for embedding in [input_embeddings, output_embeddings]:
            embedding_data = embedding.weight.data
            existing_mean = embedding_data[:current_size].mean(dim=0, keepdim=True)
            embedding_data[current_size:] = existing_mean
            
            if jitter_new_embeddings:
                existing_std = embedding_data[:current_size].std(dim=0, keepdim=True)
                noise = torch.randn_like(embedding_data[current_size:]) * existing_std
                embedding_data[current_size:] += noise

class StaggeredCreation:
    """Context manager for staggered object creation in distributed environments"""
    def __init__(self, local_rank: int, world_size: int):
        self.local_rank = local_rank
        self.world_size = world_size

    def __enter__(self):
        if self.world_size > 1 and self.local_rank % 2 == 0:
            dist.barrier()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_size > 1:
            if self.local_rank % 2 == 1:
                dist.barrier()
            dist.barrier()

    def __call__(self, func: Callable) -> Callable:
        """Decorator for staggered function execution"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

def optimize_memory_usage(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize gradient memory usage by setting gradients to None"""
    def efficient_zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()
    
    model.zero_grad = types.MethodType(efficient_zero_grad, model)
    return model

def save_model_checkpoint(
    trainer: transformers.Trainer, 
    output_dir: str, 
    set_permissions: bool = True,
    rank0_only: bool = True
) -> None:
    """Save model checkpoint with distributed training support"""
    start_time = time.perf_counter()
    
    if trainer.is_fsdp_enabled:
        # Full-state saving for FSDP models
        save_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_config):
            state_dict = trainer.model.state_dict()
            if trainer.args.should_save:
                trainer._save(output_dir, state_dict=state_dict)
    
    elif trainer.is_deepspeed_enabled:
        # DeepSpeed has its own saving mechanism
        if trainer.args.should_save:
            trainer._save(output_dir)
    
    else:
        # Standard model saving
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            # Move to CPU to save memory
            cpu_state = {k: v.cpu() for k, v in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state)
    
    # Post-save operations
    if trainer.args.should_save:
        if set_permissions:
            try:
                os.chmod(output_dir, 0o777)  # rwx for all users
                for root, dirs, files in os.walk(output_dir):
                    for name in dirs + files:
                        os.chmod(os.path.join(root, name), 0o777)
            except OSError as e:
                logger.error(f"Permission setting failed: {e}")
        
        elapsed = time.perf_counter() - start_time
        logger.info(f"Model saved in {elapsed:.2f} seconds to {output_dir}")