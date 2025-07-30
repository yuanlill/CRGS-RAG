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
from typing import Literal
from dataclasses import dataclass, field

import torch
import transformers
from transformers import Trainer, AutoModelForCausalLM

import log_utils
import common_utils
import data_utils

logger = log_utils.get_logger(__name__)

@dataclass
class ModelArguments:
    """Configuration for model-related parameters"""
    model_path: str = field(
        default=None, 
        metadata={"help": "Path to pretrained model or HuggingFace model name"}
    )

@dataclass
class DataArguments:
    """Configuration for dataset-related parameters"""
    dataset_name: str = field(
        default=None,
        metadata={"help": "Name of the dataset to use"}
    )
    prompt_config_path: str = field(
        default="xx_path",
        metadata={"help": "Path to prompt configuration file"}
    )
    num_documents: int = field(
        default=5, 
        metadata={"help": "Number of documents to use per sample"}
    )

@dataclass
class TrainingConfig(transformers.TrainingArguments):
    """Configuration for training parameters"""
    cache_dir: str = field(default=None)
    optimizer: str = field(default="adamw_torch")
    max_sequence_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for input truncation"}
    )
    padding_strategy: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={"help": "Padding strategy: fixed length or longest in batch"}
    )
    resume_from_checkpoint: bool = field(
        default=False, 
        metadata={"help": "Resume training from saved checkpoint"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Use fast tokenizer implementation"}
    )
    save_fp16: bool = field(
        default=True,
        metadata={"help": "Save model weights in float16 format"}
    )

def configure_model(model_args: ModelArguments, training_config: TrainingConfig) -> AutoModelForCausalLM:
    """Initialize and configure the model with memory optimizations"""
    model_config = transformers.AutoConfig.from_pretrained(model_args.model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        attn_implementation="flash_attention_2",
        config=model_config,
        cache_dir=training_config.cache_dir,
        low_cpu_mem_usage=True,
        device_map={"": training_config.device.index},
    )
    
    common_utils.optimize_memory_usage(model)
    return model

def configure_tokenizer(model_args: ModelArguments, training_config: TrainingConfig) -> transformers.PreTrainedTokenizer:
    """Initialize and configure the tokenizer"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        cache_dir=training_config.cache_dir,
        model_max_length=training_config.max_sequence_length,
        padding_side="right",
        truncation_side="left",
        use_fast=training_config.use_fast_tokenizer,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def prepare_training_environment(training_config: TrainingConfig):
    """Configure training environment settings"""
    training_config.gradient_checkpointing = True
    training_config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_config.save_strategy = "no"

def train_model():
    """Execute the full model training workflow"""
    # Parse command-line arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingConfig))
    model_args, data_args, training_config = parser.parse_args_into_dataclasses()
    
    # Configure training environment
    prepare_training_environment(training_config)
    
    # Initialize model with distributed context
    with common_utils.staggered_object_creation(
        local_rank=training_config.local_rank, 
        world_size=training_config.world_size
    ):
        model = configure_model(model_args, training_config)
    
    # Initialize tokenizer
    tokenizer = configure_tokenizer(model_args, training_config)
    
    # Prepare training data
    data_module = data_utils.create_supervised_dataset(
        tokenizer=tokenizer,
        data_config=data_args,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_config,
        **data_module,
    )
    
    # Execute training
    trainer.train(resume_from_checkpoint=training_config.resume_from_checkpoint)
    logger.info("Training completed successfully")
    
    # Save final model
    common_utils.save_model_checkpoint(
        trainer=trainer, 
        output_dir=training_config.output_dir,
        save_fp16=True
    )
    logger.info("Model saved successfully")

if __name__ == "__main__":
    train_model()