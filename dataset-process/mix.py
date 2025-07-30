import json
import random
import argparse
import os
from typing import List, Dict, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data: List[Dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def mix_datasets(input_files: List[str], ratios: List[float], output_file: str) -> None:
    datasets = []
    for file_path in input_files:
        try:
            data = load_json_data(file_path)
            datasets.append(data)
        except Exception as e:
            print(f"Failed to load dataset {file_path}: {e}")
            return
    
    samples_per_dataset = []
    for i, (data, ratio) in enumerate(zip(datasets, ratios)):
        num_samples = int(len(data) * ratio)
        if num_samples <= 0:
            num_samples = 1
        elif num_samples > len(data):
            num_samples = len(data)
        samples_per_dataset.append(num_samples)
    
    mixed_data = []
    for i, (data, num_samples) in enumerate(zip(datasets, samples_per_dataset)):
        selected = random.sample(data, num_samples)
        for item in selected:
            item['dataset_source'] = 'xx_path'
        mixed_data.extend(selected)
    
    random.shuffle(mixed_data)
    
    save_json_data(mixed_data, output_file)
    
    source_counts = {}
    for item in mixed_data:
        source = item.get('dataset_source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("Sample counts per source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count} samples ({count/len(mixed_data)*100:.2f}%)")
