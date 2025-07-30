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
# import os
# import copy
# import json
# import dataclasses
# from tqdm import tqdm
# from functools import partial
# from typing import Dict, Sequence, Union, List

# import torch
# import transformers
# import log_utils
# import common_utils

# IGNORE_INDEX = -100
# logger = log_utils.get_logger(__name__)

# class SupervisedFineTuningDataset(torch.utils.data.Dataset):
#     def __init__(self, data_list: List[dict], prompt_config: dict, 
#                  tokenizer: transformers.PreTrainedTokenizer, num_documents: int):
#         super().__init__()
#         processed_data = prepare_rag_data(
#             data_list=data_list, 
#             prompt_config=prompt_config,
#             tokenizer=tokenizer,
#             num_documents=num_documents
#         )

#         self.input_ids = processed_data["input_ids"]
#         self.labels = processed_data["labels"]
#         self.metadata = processed_data["metadata"]
#         self.tokenization_metadata = processed_data["tokenization_metadata"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, index) -> Dict[str, torch.Tensor]:
#         return {"input_ids": self.input_ids[index], "labels": self.labels[index]}

# @dataclasses.dataclass
# class SupervisedDataCollator:
#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids = [item["input_ids"] for item in batch]
#         labels = [item["labels"] for item in batch]
        
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=IGNORE_INDEX
#         )
        
#         attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
#         return {
#             "input_ids": input_ids,
#             "labels": labels,
#             "attention_mask": attention_mask,
#         }

# def create_supervised_dataset(
#     tokenizer: transformers.PreTrainedTokenizer,
#     data_config,
# ):
#     prompt_config = common_utils.load_json(data_config.prompt_dict_path)
#     data_path = "xx_path"
#     logger.info(f"Loading training data from: {data_path}")
#     data_list = common_utils.load_json(data_path)

#     train_dataset = SupervisedFineTuningDataset(
#         data_list=data_list,       
#         prompt_config=prompt_config,    
#         tokenizer=tokenizer,        
#         num_documents=data_config.num_documents,    
#     )

#     data_collator = SupervisedDataCollator(tokenizer=tokenizer)
    
#     return {"train_dataset": train_dataset, "data_collator": data_collator}

# def normalize_question(question: str) -> str:
#     """Format questions consistently"""
#     if not question.endswith("?"):
#         question += "?"
#     if question.startswith("."):
#         question = question.lstrip(". ")
#     return question[0].lower() + question[1:]

# def build_context_string(example: dict, num_documents: int) -> str:
#     """Build context string from documents"""
#     ctxs = example["ctxs"][:num_documents]
#     if ctxs and ctxs[0]["score"] > ctxs[1]["score"]:
#         ctxs = ctxs[::-1]
    
#     return "\n\n".join(
#         f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}"
#         for idx, ctx in enumerate(ctxs)
#     ) + "\n\n"

# def prepare_rag_data(
#     data_list: List[dict],
#     prompt_config: dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     num_documents: int,
#     verbose: bool = True,
# ) -> dict:
#     """Prepare data for RAG training"""
#     sources, targets = [], []
#     assistant_prefix = prompt_config['assistant_prefix']
#     user_prefix = prompt_config['user_prefix']
    
#     # Precompute tokenized prefixes
#     assistant_prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
#     user_prefix_ids = tokenizer.encode(user_prefix, add_special_tokens=True)
#     assistant_prefix_len = len(assistant_prefix_ids)
#     user_prefix_len = len(user_prefix_ids)

#     for sample in data_list:
#         question = normalize_question(sample['question'])
#         context = build_context_string(sample, num_documents)
#         query = prompt_config['query_prompt'] + question
        
#         sources.append(context + query)
#         targets.append(assistant_prefix + sample['rationale'] + tokenizer.eos_token)
    
#     if verbose and sources:
#         logger.info("\n=== Sample Input Example ===")
#         logger.info(f"Input: {sources[0]}")
#         logger.info(f"Target Output: {targets[0]}")
#         logger.info("=== Sample End ===\n")

#     # Tokenize full examples
#     examples = [source + target for source, target in zip(sources, targets)]
#     tokenized_examples = tokenize_texts(
#         texts=examples, 
#         tokenizer=tokenizer, 
#         max_length_offsets=[user_prefix_len] * len(examples),
#         add_special_tokens=False
#     )

#     # Prepare input IDs and labels
#     input_ids = [torch.cat([torch.tensor(user_prefix_ids), ids]) 
#                  for ids in tokenized_examples["input_ids"]]
    
#     tokenized_targets = tokenize_texts(
#         texts=targets, 
#         tokenizer=tokenizer, 
#         add_special_tokens=False
#     )
    
#     # Create labels with ignore index for non-target parts
#     labels = []
#     for idx, input_seq in enumerate(input_ids):
#         target_seq = tokenized_targets["input_ids"][idx]
#         label_seq = torch.full_like(input_seq, IGNORE_INDEX)
#         label_seq[-len(target_seq) + assistant_prefix_len:] = target_seq[assistant_prefix_len:]
#         labels.append(label_seq)

#     result = {
#         "input_ids": input_ids,
#         "labels": labels,
#         "metadata": {},
#         "tokenization_metadata": tokenized_examples["tokenization_metadata"],
#     }

#     if verbose:
#         logger.info(f"Tokenization metadata:\n{json.dumps(result['tokenization_metadata'])}")

#     return result

# def tokenize_texts(
#     texts: Sequence[str], 
#     tokenizer: transformers.PreTrainedTokenizer, 
#     max_length_offsets: Optional[List[int]] = None,
#     add_special_tokens: bool = True
# ) -> dict:
#     """Tokenize a batch of texts with optional length offsets"""
#     padding = getattr(tokenizer, "padding", "longest")
#     tokenized = []
    
#     for i, text in enumerate(texts):
#         max_length = tokenizer.model_max_length
#         if max_length_offsets:
#             max_length -= max_length_offsets[i]
        
#         tokenized.append(tokenizer(
#             text,
#             return_tensors="pt",
#             padding=padding,
#             max_length=max_length,
#             truncation=True,
#             add_special_tokens=add_special_tokens,
#         ))

#     input_ids = [t.input_ids[0] for t in tokenized]
#     input_lens = [t.input_ids.ne(tokenizer.pad_token_id).sum().item() for t in tokenized]
    
#     return {
#         "input_ids": input_ids,
#         "input_ids_lens": input_lens,
#         "tokenization_metadata": {
#             "num_examples": len(tokenized),
#             "input_ids_avg_len": sum(input_lens) / len(input_lens),
#             "input_ids_max_len": max(input_lens),
#             "input_ids_min_len": min(input_lens),
#             "model_max_length": tokenizer.model_max_length,
#         },
#     }

# def clean_document_text(text: str) -> str:
#     """Clean document text by removing special characters"""
#     return text.translate(str.maketrans('', '', '*#"\'\\\t\r\f\v')).replace('\n\n', '\n')

# def format_prompt_for_task(
#     example: dict, 
#     prompt_config: dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     task_mode: str,
#     num_documents: int = 5,
#     demos: List[dict] = [],
# ) -> str:
#     """Format prompt based on specified task mode"""
#     # Normalize question and initialize components
#     example['question'] = normalize_question(example['question'])
#     context = build_context_string(example, num_documents)
#     query = prompt_config['query_prompt'].format_map(example)
#     prefix = prompt_config['user_prefix']
#     target_prefix = ""
    
#     # Handle different task modes
#     if task_mode == "rationale_generation":
#         prefix += prompt_config['demo_prefix'].format_map(example)
#         target_prefix += prompt_config['rationale_generation_instruction'].format_map(example)
#         target_prefix += prompt_config['rationale_generation_postfix']
        
#     elif task_mode == "fused_rationale_generation":
#         context = example['fused_document']
#         prefix += prompt_config['demo_prefix'].format_map(example)
#         target_prefix += prompt_config['rationale_generation_instruction'].format_map(example)
#         target_prefix += prompt_config['rationale_generation_postfix']
        
#     elif task_mode == "document_generation":
#         context = ""
#         prefix += prompt_config['document_generation_prefix'].format_map(example)
        
#     elif task_mode == "supplementary_document_generation":
#         prefix += prompt_config['supplementary_document_generation_prefix'].format_map(example)
        
#     elif task_mode == "origin_document_evaluation":
#         cleaned_doc = clean_document_text(example['document'])
#         context += f"Supplementary Document: \n{cleaned_doc}\n\n"
        
#     elif task_mode == "fused_document_evaluation":
#         context = example['fused_document'] + "\n\n"
        
#     elif task_mode == "supplementary_document_evaluation":
#         cleaned_doc = clean_document_text(example['document'])
#         context += f"{cleaned_doc}\n\n"
        
#     elif task_mode.startswith("fuse_document_"):
#         fuse_type = task_mode.split("_")[-1]
#         prefix += prompt_config[f'document_fuse_prefix_{fuse_type}'].format_map(example)
#         cleaned_doc = clean_document_text(example['document'])
#         context += f"{cleaned_doc}\n\n"
    
#     # Add demonstration examples if provided
#     if demos:
#         prefix += prompt_config['demo_task_instruction']
#         for idx, demo in enumerate(demos):
#             demo_question = normalize_question(demo['question'])
#             prefix += f"###\n\nExample {idx+1}\n\nQuestion: {demo_question}\n\nAnswer: {demo['rationale']}\n\n"
#         prefix += prompt_config['demo_postfix']
    
#     # Tokenize and truncate
#     prefix_tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
#     max_length = tokenizer.model_max_length
#     content = context + query + target_prefix + prompt_config['assistant_prefix']
#     content_tokens = tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
    
#     if content_tokens.shape[1] > max_length - prefix_tokens.shape[1]:
#         content_tokens = content_tokens[:, -(max_length - prefix_tokens.shape[1]):]
    
#     full_tokens = torch.cat([prefix_tokens, content_tokens], dim=1)
#     return tokenizer.decode(full_tokens[0], skip_special_tokens=False)

# def format_prompts_for_dataset(
#     data_list: List[dict],
#     prompt_config: dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     task_mode: str,
#     num_documents: int = 5,
#     demos: List[dict] = [],
# ) -> List[str]:
#     """Format prompts for an entire dataset"""
#     logger.info(f"Formatting prompts for {len(data_list)} examples...")
#     return [
#         format_prompt_for_task(
#             example=copy.deepcopy(example),
#             prompt_config=prompt_config,
#             tokenizer=tokenizer,
#             task_mode=task_mode,
#             num_documents=num_documents,
#             demos=demos
#         )
#         for example in tqdm(data_list)
#     ]

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
import copy
import json
import dataclasses
from tqdm import tqdm
from functools import partial
from typing import Dict, Sequence, Union, List

import torch
import transformers
import log_utils
import common_utils

IGNORE_INDEX = -100
logger = log_utils.get_logger(__name__)

class SupervisedFineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, data_list: List[dict], prompt_config: dict, 
                 tokenizer: transformers.PreTrainedTokenizer, num_documents: int):
        super().__init__()
        processed_data = prepare_rag_data(
            data_list=data_list, 
            prompt_config=prompt_config,
            tokenizer=tokenizer,
            num_documents=num_documents
        )

        self.input_ids = processed_data["input_ids"]
        self.labels = processed_data["labels"]
        self.metadata = processed_data["metadata"]
        self.tokenization_metadata = processed_data["tokenization_metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}

@dataclasses.dataclass
class SupervisedDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

def create_supervised_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    data_config,
):
    prompt_config = common_utils.load_json(data_config.prompt_dict_path)
    data_path = "xx_path"
    logger.info(f"Loading training data from: {data_path}")
    data_list = common_utils.load_json(data_path)

    train_dataset = SupervisedFineTuningDataset(
        data_list=data_list,       
        prompt_config=prompt_config,    
        tokenizer=tokenizer,        
        num_documents=data_config.num_documents,    
    )

    data_collator = SupervisedDataCollator(tokenizer=tokenizer)
    
    return {"train_dataset": train_dataset, "data_collator": data_collator}

def normalize_question(question: str) -> str:
    """Format questions consistently"""
    if not question.endswith("?"):
        question += "?"
    if question.startswith("."):
        question = question.lstrip(". ")
    return question[0].lower() + question[1:]

def build_context_string(example: dict, num_documents: int) -> str:
    """Build context string from documents"""
    ctxs = example["ctxs"][:num_documents]
    if ctxs and ctxs[0]["score"] > ctxs[1]["score"]:
        ctxs = ctxs[::-1]
    
    return "\n\n".join(
        f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}"
        for idx, ctx in enumerate(ctxs)
    ) + "\n\n"

def prepare_rag_data(
    data_list: List[dict],
    prompt_config: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    num_documents: int,
    verbose: bool = True,
) -> dict:
    """Prepare data for RAG training"""
    sources, targets = [], []
    assistant_prefix = prompt_config['assistant_prefix']
    user_prefix = prompt_config['user_prefix']
    
    # Precompute tokenized prefixes
    assistant_prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    user_prefix_ids = tokenizer.encode(user_prefix, add_special_tokens=True)
    assistant_prefix_len = len(assistant_prefix_ids)
    user_prefix_len = len(user_prefix_ids)

    for sample in data_list:
        question = normalize_question(sample['question'])
        context = build_context_string(sample, num_documents)
        query = prompt_config['query_prompt'] + question
        
        sources.append(context + query)
        targets.append(assistant_prefix + sample['rationale'] + tokenizer.eos_token)
    
    if verbose and sources:
        logger.info("\n=== Sample Input Example ===")
        logger.info(f"Input: {sources[0]}")
        logger.info(f"Target Output: {targets[0]}")
        logger.info("=== Sample End ===\n")

    # Tokenize full examples
    examples = [source + target for source, target in zip(sources, targets)]
    tokenized_examples = tokenize_texts(
        texts=examples, 
        tokenizer=tokenizer, 
        max_length_offsets=[user_prefix_len] * len(examples),
        add_special_tokens=False
    )

    # Prepare input IDs and labels
    input_ids = [torch.cat([torch.tensor(user_prefix_ids), ids]) 
                 for ids in tokenized_examples["input_ids"]]
    
    tokenized_targets = tokenize_texts(
        texts=targets, 
        tokenizer=tokenizer, 
        add_special_tokens=False
    )
    
    # Create labels with ignore index for non-target parts
    labels = []
    for idx, input_seq in enumerate(input_ids):
        target_seq = tokenized_targets["input_ids"][idx]
        label_seq = torch.full_like(input_seq, IGNORE_INDEX)
        label_seq[-len(target_seq) + assistant_prefix_len:] = target_seq[assistant_prefix_len:]
        labels.append(label_seq)

    result = {
        "input_ids": input_ids,
        "labels": labels,
        "metadata": {},
        "tokenization_metadata": tokenized_examples["tokenization_metadata"],
    }

    if verbose:
        logger.info(f"Tokenization metadata:\n{json.dumps(result['tokenization_metadata'])}")

    return result

def tokenize_texts(
    texts: Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer, 
    max_length_offsets: Optional[List[int]] = None,
    add_special_tokens: bool = True
) -> dict:
    """Tokenize a batch of texts with optional length offsets"""
    padding = getattr(tokenizer, "padding", "longest")
    tokenized = []
    
    for i, text in enumerate(texts):
        max_length = tokenizer.model_max_length
        if max_length_offsets:
            max_length -= max_length_offsets[i]
        
        tokenized.append(tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            max_length=max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        ))

    input_ids = [t.input_ids[0] for t in tokenized]
    input_lens = [t.input_ids.ne(tokenizer.pad_token_id).sum().item() for t in tokenized]
    
    return {
        "input_ids": input_ids,
        "input_ids_lens": input_lens,
        "tokenization_metadata": {
            "num_examples": len(tokenized),
            "input_ids_avg_len": sum(input_lens) / len(input_lens),
            "input_ids_max_len": max(input_lens),
            "input_ids_min_len": min(input_lens),
            "model_max_length": tokenizer.model_max_length,
        },
    }

def clean_document_text(text: str) -> str:
    """Clean document text by removing special characters"""
    return text.translate(str.maketrans('', '', '*#"\'\\\t\r\f\v')).replace('\n\n', '\n')

def format_prompt_for_task(
    example: dict, 
    prompt_config: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    task_mode: str,
    num_documents: int = 5,
    demos: List[dict] = [],
) -> str:
    """Format prompt based on specified task mode"""
    # Normalize question and initialize components
    example['question'] = normalize_question(example['question'])
    context = build_context_string(example, num_documents)
    query = prompt_config['query_prompt'].format_map(example)
    prefix = prompt_config['user_prefix']
    target_prefix = ""
    
    # Handle different task modes
    if task_mode == "rationale_generation":
        prefix += prompt_config['demo_prefix'].format_map(example)
        target_prefix += prompt_config['rationale_generation_instruction'].format_map(example)
        target_prefix += prompt_config['rationale_generation_postfix']
        
    elif task_mode == "fused_rationale_generation":
        context = example['fused_document']
        prefix += prompt_config['demo_prefix'].format_map(example)
        target_prefix += prompt_config['rationale_generation_instruction'].format_map(example)
        target_prefix += prompt_config['rationale_generation_postfix']
        
    elif task_mode == "document_generation":
        context = ""
        prefix += prompt_config['document_generation_prefix'].format_map(example)
        
    elif task_mode == "supplementary_document_generation":
        prefix += prompt_config['supplementary_document_generation_prefix'].format_map(example)
        
    elif task_mode == "origin_document_evaluation":
        cleaned_doc = clean_document_text(example['document'])
        context += f"Supplementary Document: \n{cleaned_doc}\n\n"
        
    elif task_mode == "fused_document_evaluation":
        context = example['fused_document'] + "\n\n"
        
    elif task_mode == "supplementary_document_evaluation":
        cleaned_doc = clean_document_text(example['document'])
        context += f"{cleaned_doc}\n\n"
        
    elif task_mode.startswith("fuse_document_"):
        fuse_type = task_mode.split("_")[-1]
        prefix += prompt_config[f'document_fuse_prefix_{fuse_type}'].format_map(example)
        cleaned_doc = clean_document_text(example['document'])
        context += f"{cleaned_doc}\n\n"
    
    # Add demonstration examples if provided
    if demos:
        prefix += prompt_config['demo_task_instruction']
        for idx, demo in enumerate(demos):
            demo_question = normalize_question(demo['question'])
            prefix += f"###\n\nExample {idx+1}\n\nQuestion: {demo_question}\n\nAnswer: {demo['rationale']}\n\n"
        prefix += prompt_config['demo_postfix']
    
    # Tokenize and truncate
    prefix_tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids
    max_length = tokenizer.model_max_length
    content = context + query + target_prefix + prompt_config['assistant_prefix']
    content_tokens = tokenizer(content, return_tensors="pt", add_special_tokens=False).input_ids
    
    if content_tokens.shape[1] > max_length - prefix_tokens.shape[1]:
        content_tokens = content_tokens[:, -(max_length - prefix_tokens.shape[1]):]
    
    full_tokens = torch.cat([prefix_tokens, content_tokens], dim=1)
    return tokenizer.decode(full_tokens[0], skip_special_tokens=False)

def format_prompts_for_dataset(
    data_list: List[dict],
    prompt_config: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    task_mode: str,
    num_documents: int = 5,
    demos: List[dict] = [],
) -> List[str]:
    """Format prompts for an entire dataset"""
    logger.info(f"Formatting prompts for {len(data_list)} examples...")
    return [
        format_prompt_for_task(
            example=copy.deepcopy(example),
            prompt_config=prompt_config,
            tokenizer=tokenizer,
            task_mode=task_mode,
            num_documents=num_documents,
            demos=demos
        )
        for example in tqdm(data_list)
    ]