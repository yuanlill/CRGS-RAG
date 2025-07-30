import json
import random
import argparse
import copy
import re
from typing import List, Dict, Any
from tqdm import tqdm

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def filter_eligible_samples(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter samples that have both relevant and irrelevant contexts"""
    return [
        item for item in data
        if item.get('useful_ctxs_indices') and 
           len(item.get('useful_ctxs_indices', [])) < len(item.get('ctxs', []))
    ]

def get_irrelevant_indices(item: Dict[str, Any]) -> List[int]:
    """Get indices of irrelevant contexts"""
    useful_indices = {idx - 1 for idx in item.get('useful_ctxs_indices', [])}
    return [idx for idx in range(len(item.get('ctxs', []))) if idx not in useful_indices]

def remove_irrelevant_contexts(item: Dict[str, Any], indices_to_remove: List[int]) -> Dict[str, Any]:
    """Remove specified irrelevant contexts by setting them to null"""
    new_item = copy.deepcopy(item)
    for idx in indices_to_remove:
        new_item['ctxs'][idx]['title'] = "null"
        new_item['ctxs'][idx]['text'] = "null"
    return new_item

def create_counterfactual_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create counterfactual sample by removing irrelevant contexts"""
    irrelevant_indices = get_irrelevant_indices(item)
    if not irrelevant_indices:
        return item
    
    if random.random() < 0.8 and len(irrelevant_indices) > 1:
        # Remove partial irrelevant contexts
        num_to_remove = random.randint(1, len(irrelevant_indices) - 1)
        indices_to_remove = random.sample(irrelevant_indices, num_to_remove)
        cf_type = 'remove_partial_irrelevant'
    else:
        # Remove all irrelevant contexts
        indices_to_remove = irrelevant_indices
        cf_type = 'remove_all_irrelevant'
    
    new_item = remove_irrelevant_contexts(item, indices_to_remove)
    new_item['counterfactual_type'] = cf_type
    new_item['removed_indices'] = indices_to_remove
    return new_item

def build_counterfactual_dataset(input_file: str, output_file: str) -> None:
    """Construct counterfactual dataset by removing irrelevant contexts"""
    data = load_json_data(input_file)
    eligible_samples = filter_eligible_samples(data)
    print(f"Eligible sample count: {len(eligible_samples)}/{len(data)}")
    
    counterfactual_samples = [
        create_counterfactual_sample(item)
        for item in tqdm(eligible_samples, desc="Processing samples")
    ]
    
    save_json_data(counterfactual_samples, output_file)
    print(f"Counterfactual dataset saved to: {output_file}")
    
    # Collect statistics
    counterfactual_stats = {}
    for item in counterfactual_samples:
        cf_type = item.get('counterfactual_type', 'unknown')
        counterfactual_stats[cf_type] = counterfactual_stats.get(cf_type, 0) + 1
    
    print("Counterfactual method statistics:")
    for cf_type, count in counterfactual_stats.items():
        print(f"  {cf_type}: {count}")

def split_text_into_chunks(text: str) -> List[str]:
    """Split text into chunks using multiple methods"""
    # First try splitting by sentence endings
    chunks = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    if len(chunks) > 1:
        return chunks
    
    # Then try splitting by commas
    chunks = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|,)\s', text)
    if len(chunks) > 1:
        return chunks
    
    # Finally, split by word count
    words = text.split()
    chunk_size = max(1, len(words) // 4)
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def swap_chunks_in_document(item: Dict[str, Any]) -> Dict[str, Any]:
    """Swap chunks within a single document"""
    new_item = copy.deepcopy(item)
    irrelevant_indices = get_irrelevant_indices(new_item)
    if not irrelevant_indices:
        return new_item
    
    doc_idx = random.choice(irrelevant_indices)
    text = new_item['ctxs'][doc_idx]['text']
    chunks = split_text_into_chunks(text)
    
    if len(chunks) < 2:
        return new_item
    
    n_chunks = min(4, len(chunks))
    chunk_size = len(chunks) // n_chunks
    chunk_list = [
        chunks[i*chunk_size:(i+1)*chunk_size] 
        for i in range(n_chunks-1)
    ]
    chunk_list.append(chunks[(n_chunks-1)*chunk_size:])
    
    idx1, idx2 = random.sample(range(n_chunks), 2)
    chunk_list[idx1], chunk_list[idx2] = chunk_list[idx2], chunk_list[idx1]
    
    new_text = ' '.join(' '.join(chunk) for chunk in chunk_list)
    new_item['ctxs'][doc_idx]['text'] = new_text
    new_item['intervention_type'] = 'swap_chunks_internal'
    new_item['intervention_indices'] = [doc_idx]
    return new_item

def swap_chunks_between_documents(item: Dict[str, Any]) -> Dict[str, Any]:
    """Swap chunks between multiple documents"""
    new_item = copy.deepcopy(item)
    irrelevant_indices = get_irrelevant_indices(new_item)
    if len(irrelevant_indices) < 2:
        return new_item
    
    selected_docs = random.sample(irrelevant_indices, min(2, len(irrelevant_indices)))
    texts = [new_item['ctxs'][idx]['text'] for idx in selected_docs]
    chunk_lists = [split_text_into_chunks(text) for text in texts]
    
    # Skip if any document has no chunks
    if any(not chunks for chunks in chunk_lists):
        return new_item
    
    swap_positions = [random.randint(0, len(chunks)-1) for chunks in chunk_lists]
    temp = chunk_lists[0][swap_positions[0]]
    
    # Perform chain swap
    for i in range(len(selected_docs)-1):
        chunk_lists[i][swap_positions[i]] = chunk_lists[i+1][swap_positions[i+1]]
    chunk_lists[-1][swap_positions[-1]] = temp
    
    # Update document texts
    for i, idx in enumerate(selected_docs):
        new_item['ctxs'][idx]['text'] = ' '.join(chunk_lists[i])
    
    new_item['intervention_type'] = 'swap_chunks_cross'
    new_item['intervention_indices'] = selected_docs
    return new_item

def replace_documents(item: Dict[str, Any], all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Replace irrelevant documents with random documents from other samples"""
    new_item = copy.deepcopy(item)
    irrelevant_indices = get_irrelevant_indices(new_item)
    if not irrelevant_indices or not all_data:
        return new_item
    
    num_replace = min(random.randint(1, 3), len(irrelevant_indices))
    replace_indices = random.sample(irrelevant_indices, num_replace)
    
    other_samples = [
        s for s in all_data 
        if s != new_item and s.get('ctxs')
    ]
    
    for idx in replace_indices:
        if not other_samples:
            continue
            
        other_sample = random.choice(other_samples)
        random_doc = random.choice(other_sample['ctxs'])
        original_score = new_item['ctxs'][idx]['score']
        
        new_doc = copy.deepcopy(random_doc)
        new_doc['score'] = original_score
        new_doc['is_relevant'] = False
        new_item['ctxs'][idx] = new_doc
    
    new_item['intervention_type'] = f'document_replacement_{num_replace}'
    new_item['intervention_indices'] = replace_indices
    return new_item

def inject_noise(item: Dict[str, Any], noise_dataset: List[str]) -> Dict[str, Any]:
    """Inject noise fragments into irrelevant documents"""
    new_item = copy.deepcopy(item)
    irrelevant_indices = get_irrelevant_indices(new_item)
    if not irrelevant_indices or not noise_dataset:
        return new_item
    
    num_inject = min(random.randint(1, 3), len(irrelevant_indices))
    inject_indices = random.sample(irrelevant_indices, num_inject)
    
    for idx in inject_indices:
        text = new_item['ctxs'][idx]['text']
        sentences = split_text_into_chunks(text) or [text]
        insert_pos = random.randint(0, len(sentences))
        noise_fragment = random.choice(noise_dataset)
        sentences.insert(insert_pos, noise_fragment)
        new_item['ctxs'][idx]['text'] = ' '.join(sentences)
    
    new_item['intervention_type'] = f'noise_injection_{num_inject}'
    new_item['intervention_indices'] = inject_indices
    return new_item

def reorder_documents(item: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly reorder irrelevant documents"""
    new_item = copy.deepcopy(item)
    irrelevant_indices = get_irrelevant_indices(new_item)
    if len(irrelevant_indices) < 2:
        return new_item
    
    useful_indices = {idx - 1 for idx in item.get('useful_ctxs_indices', [])}
    ctxs = copy.deepcopy(new_item['ctxs'])
    
    # Create mapping for irrelevant documents
    irrelevant_docs = {idx: ctxs[idx] for idx in irrelevant_indices}
    permuted_order = irrelevant_indices.copy()
    random.shuffle(permuted_order)
    
    # Build new document order
    new_ctxs = []
    counter = 0
    for i, doc in enumerate(ctxs):
        if i in useful_indices:
            new_ctxs.append(doc)
        else:
            new_ctxs.append(irrelevant_docs[permuted_order[counter]])
            counter += 1
    
    new_item['ctxs'] = new_ctxs
    new_item['intervention_type'] = 'document_reordering'
    new_item['original_order'] = irrelevant_indices
    new_item['new_order'] = permuted_order
    
    # Update useful indices if needed
    if item.get('useful_ctxs_indices'):
        new_useful_indices = [i + 1 for i, doc in enumerate(new_ctxs) if i in useful_indices]
        new_item['useful_ctxs_indices'] = new_useful_indices
    
    return new_item

def apply_intervention(item: Dict[str, Any], all_data: List[Dict[str, Any]], noise_data: List[str]) -> Dict[str, Any]:
    """Apply random intervention to a sample"""
    interventions = [
        swap_chunks_in_document,
        lambda i: replace_documents(i, all_data),
        lambda i: inject_noise(i, noise_data),
        reorder_documents
    ]
    return random.choice(interventions)(item)

def build_intervention_dataset(input_path: str, output_path: str, noise_data_path: str) -> None:
    """Construct intervention dataset"""
    data = load_json_data(input_path)
    eligible_samples = filter_eligible_samples(data)
    noise_data = load_json_data(noise_data_path)
    
    # Extract noise fragments
    noise_texts = []
    for item in noise_data:
        for ctx in item.get('ctxs', []):
            text = ctx.get('text', '')
            noise_texts.extend(split_text_into_chunks(text))
    
    # Apply interventions
    intervened_samples = [
        apply_intervention(item, data, noise_texts)
        for item in tqdm(eligible_samples, desc="Applying interventions")
    ]
    
    save_json_data(intervened_samples, output_path)
    
    # Collect statistics
    intervention_stats = {}
    for item in intervened_samples:
        int_type = item.get('intervention_type', 'no_intervention')
        intervention_stats[int_type] = intervention_stats.get(int_type, 0) + 1