import os
import sys
import argparse
import data_utils
import common_utils
from metrics import calculate_metrics
from vllm import LLM, SamplingParams

def run_generation_task(args, task_type: str):
    """Run a generation task based on the specified type"""
    # Common setup for all generation tasks
    data_path = 'xx_path'
    print(f"Loading dataset from: {data_path}")
    data = common_utils.load_json(data_path)[:args.max_instances]
    
    # Configure LLM with common parameters
    llm_config = {
        "model": args.model_name_or_path,
        "download_dir": args.cache_dir,
        "max_model_len": args.max_tokens,
        "tensor_parallel_size": args.gpu_count,
    }
    
    # Add memory optimization for document-related tasks
    if task_type in ["document", "fuse_document"]:
        llm_config["gpu_memory_utilization"] = 0.95
    
    llm = LLM(**llm_config)
    tokenizer = llm.get_tokenizer()
    prompt_config = common_utils.load_json(args.prompt_dict_path)
    
    # Generate prompts based on task type
    prompts = generate_prompts_for_task(
        data_list=data,
        task_type=task_type,
        prompt_config=prompt_config,
        tokenizer=tokenizer,
        num_documents=args.n_docs,
        dataset_name=args.dataset_name
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    
    # Execute generation
    outputs = llm.generate(prompts, sampling_params)
    
    # Determine output file based on task type
    output_file = determine_output_file(args, task_type)
    
    # Save results with task-specific formatting
    return save_generation_results(
        outputs=outputs,
        data=data,
        output_file=output_file,
        num_documents=args.n_docs,
        task_type=task_type
    )

def generate_prompts_for_task(data_list: list, task_type: str, prompt_config: dict, 
                             tokenizer: transformers.PreTrainedTokenizer, num_documents: int, 
                             dataset_name: str) -> list:
    """Generate prompts based on task type"""
    task_to_flag_map = {
        "rationale": {"do_rationale_generation": True},
        "fused_rationale": {"do_fused_rationale_generation": True},
        "document": {"do_document_generation": True},
        "supplementary_document": {"do_supplementary_document_generation": True},
        "fuse_document": {"fuse_document": True},
        "fuse_document_cooperative": {"fuse_document_Cooperative": True},
        "fuse_document_complementary": {"fuse_document_Complementary": True},
        "fuse_document_competitive": {"fuse_document_Competitive": True},
        "fuse_document_fallback": {"fuse_document_Fallback": True},
    }
    
    if task_type not in task_to_flag_map:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return data_utils.format_prompts_for_dataset(
        data_list=data_list,
        prompt_config=prompt_config,
        tokenizer=tokenizer,
        task_mode=task_type,
        num_documents=num_documents,
        dataset_name=dataset_name
    )

def run_evaluation_task(args):
    """Run model evaluation task"""
    data_path = 'xx_path'
    print(f"Loading evaluation dataset: {data_path}")
    eval_data = common_utils.load_json(data_path)[:args.max_instances]
    
    print(f'Loading model {args.rag_model}...')
    
    # Model configuration based on RAG type
    model_configs = {
        'XXX': {
            "model_path": 'xx_path' if args.load_local_model else 'xx_path',
            "demos": []
        },
        'XXX': {
            "model_path": 'xx_path' if args.load_local_model else 'xx_path',
            "demos": common_utils.load_json('xx_path')
        },
        'BaseRAG': {
            "model_path": 'xx_path',
            "demos": []
        }
    }
    
    if args.rag_model not in model_configs:
        print(f"Unsupported RAG model: {args.rag_model}")
        sys.exit(1)
    
    config = model_configs[args.rag_model]
    
    # Initialize LLM
    llm = LLM(
        model=config["model_path"],
        max_model_len=args.max_tokens,
        tensor_parallel_size=args.gpu_count
    )
    
    tokenizer = llm.get_tokenizer()
    prompt_config = common_utils.load_json(args.prompt_dict_path)
    
    # Determine evaluation mode
    eval_mode = determine_evaluation_mode(args)
    
    # Generate prompts
    prompts = data_utils.format_prompts_for_dataset(
        data_list=eval_data,
        prompt_config=prompt_config,
        tokenizer=tokenizer,
        task_mode=eval_mode,
        num_documents=args.n_docs,
        demos=config["demos"],
        dataset_name=args.dataset_name
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    
    # Execute generation
    outputs = llm.generate(prompts, sampling_params)
    
    # Save results
    output_file = os.path.join(args.output_dir, "result.json")
    results = save_generation_results(
        outputs=outputs,
        data=eval_data,
        output_file=output_file,
        num_documents=args.n_docs,
        task_type="evaluation"
    )
    
    # Calculate metrics
    calculate_metrics(
        results=results,
        output_dir=args.output_dir,
        is_asqa=(args.dataset_name == 'ASQA')
    )

def determine_evaluation_mode(args) -> str:
    """Determine evaluation mode based on arguments"""
    if args.eval_with_origin_document:
        return "origin_document_evaluation"
    if args.eval_with_fused_document:
        return "fused_document_evaluation"
    if args.eval_with_supplementary_document:
        return "supplementary_document_evaluation"
    return "default_evaluation"

def determine_output_file(args, task_type: str) -> str:
    """Determine output file based on task type"""
    file_map = {
        "rationale": "xx_path",
        "fused_rationale": "xx_path",
        "document": "test.json",
        "supplementary_document": "test.json",
        "fuse_document": "test.json",
        "fuse_document_cooperative": "test.json",
        "fuse_document_complementary": "test.json",
        "fuse_document_competitive": "test.json",
        "fuse_document_fallback": "test.json",
    }
    return os.path.join(args.output_dir, file_map.get(task_type, "output.json"))

def save_generation_results(outputs, data, output_file: str, num_documents: int, task_type: str) -> list:
    """Save generation results with task-specific formatting"""
    result_data = []
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        sample = data[i]
        
        # Format documents based on score ordering
        documents = sample["ctxs"][:num_documents]
        if documents and documents[0]['score'] > documents[1]['score']:
            documents = documents[::-1]
        
        # Create base result structure
        result = {
            "question": sample["question"],
            "answers": sample["answers"],
            "qa_pairs": sample.get("qa_pairs"),
            "ctxs": documents,
            "prompt": prompt,
        }
        
        # Add task-specific fields
        if task_type == "rationale":
            result["rationale"] = generated_text
        elif task_type == "fused_rationale":
            result.update({
                "document": sample.get("document"),
                "fused_document": sample.get("fused_document"),
                "rationale": generated_text
            })
        elif task_type == "document":
            result["document"] = generated_text
        elif task_type == "supplementary_document":
            result["supplementary_document"] = generated_text
        elif task_type == "fuse_document":
            result.update({
                "document": sample.get("document"),
                "fused_document": generated_text
            })
        elif task_type.startswith("fuse_document_"):
            result.update({
                "document": sample.get("document"),
                "fused_document": generated_text
            })
        elif task_type == "evaluation":
            result["rationale"] = generated_text
        
        result_data.append(result)
    
    # Save results
    common_utils.save_json(result_data, output_file)
    print(f"Results saved to {output_file}")
    return result_data

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--rag_model', type=str, choices=['BaseRAG'], 
                       default='BaseRAG', help='RAG model type')
    parser.add_argument('--model_name_or_path', type=str, 
                       default='meta-llama/Meta-Llama-3-8B-Instruct', 
                       help='Model name or path')
    parser.add_argument('--load_local_model', action='store_true', 
                       help='Load model from local path')
    parser.add_argument('--n_docs', type=int, default=5, 
                       help='Number of retrieved documents')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--cache_dir', type=str, default=None, 
                       help='Model cache directory')
    parser.add_argument('--prompt_dict_path', type=str, default="xx_path",
                       help='Path to prompt configuration')
    parser.add_argument('--temperature', type=float, default=0, 
                       help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=4096, 
                       help='Maximum tokens to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize,
                       help='Maximum instances to process')
    parser.add_argument('--gpu_count', type=int, default=2, 
                       help='Number of GPUs to use')
    
    # Task flags
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument('--do_rationale_generation', action='store_true',
                          help='Generate rationales')
    task_group.add_argument('--do_fused_rationale_generation', action='store_true',
                          help='Generate fused rationales')
    task_group.add_argument('--do_document_generation', action='store_true',
                          help='Generate documents')
    task_group.add_argument('--do_supplementary_document_generation', action='store_true',
                          help='Generate supplementary documents')
    task_group.add_argument('--fuse_document', action='store_true',
                          help='Fuse documents')
    task_group.add_argument('--fuse_document_cooperative', action='store_true',
                          help='Fuse documents cooperatively')
    task_group.add_argument('--fuse_document_complementary', action='store_true',
                          help='Fuse documents complementarily')
    task_group.add_argument('--fuse_document_competitive', action='store_true',
                          help='Fuse documents competitively')
    task_group.add_argument('--fuse_document_fallback', action='store_true',
                          help='Fuse documents with fallback strategy')
    
    # Evaluation flags
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument('--eval_with_origin_document', action='store_true',
                          help='Evaluate with original document')
    eval_group.add_argument('--eval_with_fused_document', action='store_true',
                          help='Evaluate with fused document')
    eval_group.add_argument('--eval_with_supplementary_document', action='store_true',
                          help='Evaluate with supplementary document')
    
    return parser.parse_args()

def main():
    """Main function to execute tasks based on arguments"""
    args = parse_arguments()
    
    # Determine task type from arguments
    task_mapping = {
        "do_rationale_generation": ("rationale", run_generation_task),
        "do_fused_rationale_generation": ("fused_rationale", run_generation_task),
        "do_document_generation": ("document", run_generation_task),
        "do_supplementary_document_generation": ("supplementary_document", run_generation_task),
        "fuse_document": ("fuse_document", run_generation_task),
        "fuse_document_cooperative": ("fuse_document_cooperative", run_generation_task),
        "fuse_document_complementary": ("fuse_document_complementary", run_generation_task),
        "fuse_document_competitive": ("fuse_document_competitive", run_generation_task),
        "fuse_document_fallback": ("fuse_document_fallback", run_generation_task),
    }
    
    for flag, (task_type, task_func) in task_mapping.items():
        if getattr(args, flag):
            task_func(args, task_type)
            return
    
    # If no generation task was specified, run evaluation
    run_evaluation_task(args)

if __name__ == "__main__":
    main()