import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def count_tokens(text):
    return len(text.split())

def analyze_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_lengths = []
    question_lengths = []
    ctx_title_lengths = []
    ctx_text_lengths = []
    rationale_lengths = []
    
    for item in tqdm(data, desc="Analyzing dataset"):
        question = item.get("question", "")
        question_lengths.append(count_tokens(question))
        
        rationale = item.get("rationale", "")
        rationale_lengths.append(count_tokens(rationale))
        
        ctx_titles = []
        ctx_texts = []
        
        if "ctxs" in item:
            for ctx in item["ctxs"]:
                title = ctx.get("title", "")
                text = ctx.get("text", "")
                if title != "null":
                    ctx_titles.append(title)
                if text != "null":
                    ctx_texts.append(text)
        
        ctx_title_length = sum(count_tokens(title) for title in ctx_titles)
        ctx_text_length = sum(count_tokens(text) for text in ctx_texts)
        
        ctx_title_lengths.append(ctx_title_length)
        ctx_text_lengths.append(ctx_text_length)
        
        total_length = question_lengths[-1] + ctx_title_length + ctx_text_length + rationale_lengths[-1]
        total_lengths.append(total_length)
    
    stats = {
        "total": {
            "avg": np.mean(total_lengths),
            "max": np.max(total_lengths),
            "min": np.min(total_lengths),
            "coverage_4096": np.mean([l <= 4096 for l in total_lengths]) * 100,
            "coverage_3584": np.mean([l <= 3584 for l in total_lengths]) * 100,
            "coverage_3072": np.mean([l <= 3072 for l in total_lengths]) * 100,
            "coverage_2560": np.mean([l <= 2560 for l in total_lengths]) * 100,
            "coverage_2048": np.mean([l <= 2048 for l in total_lengths]) * 100,
            "coverage_1536": np.mean([l <= 1536 for l in total_lengths]) * 100,
            "coverage_1024": np.mean([l <= 1024 for l in total_lengths]) * 100,
        },
        "question": {
            "avg": np.mean(question_lengths),
            "max": np.max(question_lengths),
            "min": np.min(question_lengths),
        },
        "ctx_title": {
            "avg": np.mean(ctx_title_lengths),
            "max": np.max(ctx_title_lengths),
            "min": np.min(ctx_title_lengths),
        },
        "ctx_text": {
            "avg": np.mean(ctx_text_lengths),
            "max": np.max(ctx_text_lengths),
            "min": np.min(ctx_text_lengths),
        },
        "rationale": {
            "avg": np.mean(rationale_lengths),
            "max": np.max(rationale_lengths),
            "min": np.min(rationale_lengths),
        }
    }
    
    return stats, total_lengths

def plot_length_distribution(total_lengths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sample_indices = list(range(1, len(total_lengths) + 1))
    
    # 1. Bar chart
    plt.figure(figsize=(15, 8))
    plt.bar(sample_indices, total_lengths, alpha=0.7)
    plt.xlabel('Sample index')
    plt.ylabel('Length (tokens)')
    plt.title('Total length distribution (Bar chart)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_bar_chart.png'), dpi=300)
    plt.close()
    
    # 2. Line chart
    plt.figure(figsize=(15, 8))
    plt.plot(sample_indices, total_lengths, linewidth=1)
    plt.xlabel('Sample index')
    plt.ylabel('Length (tokens)')
    plt.title('Total length distribution (Line chart)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_line_chart.png'), dpi=300)
    plt.close()
    
    print(f"Charts saved to: {output_dir}")
