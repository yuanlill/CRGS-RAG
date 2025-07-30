import re
import json
import string
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Optional

# Answer normalization
def normalize_answer(text: str) -> str:
    """Normalize text for comparison by removing punctuation, articles, and extra spaces"""
    text = text.lower()
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^`{|}~_]', ' ', text)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())

# Evaluation metrics
def check_answer_in_context(answer_list: List[str], context: str) -> bool:
    """Check if any normalized answer appears in normalized context"""
    normalized_context = normalize_answer(context)
    return any(normalize_answer(ans) in normalized_context for ans in answer_list)

def calculate_string_exact_match_metrics(data: List[Dict]) -> Dict[str, float]:
    """Calculate string-based exact match metrics"""
    if not data or 'qa_pairs' not in data[0] or not data[0]['qa_pairs']:
        return {"accuracy": 0.0, "hit_rate": 0.0}
    
    accuracy_scores = []
    perfect_hits = []
    
    for item in data:
        item_scores = [
            int(check_answer_in_context(qa['answers'], item["rationale"]))
            for qa in item['qa_pairs']
        ]
        accuracy_scores.append(np.mean(item_scores))
        perfect_hits.append(int(np.mean(item_scores) == 1))
    
    return {
        "accuracy": 100 * np.mean(accuracy_scores),
        "hit_rate": 100 * np.mean(perfect_hits)
    }

def calculate_metrics(data: List[Dict], save_dir: Optional[str] = None, is_asqa: bool = False) -> Dict:
    """Calculate and save evaluation metrics"""
    total_examples = len(data)
    if is_asqa:
        metrics = calculate_string_exact_match_metrics(data)
        result = {"EM": metrics["accuracy"], "num_examples": total_examples}
    else:
        correct_count = sum(
            int(check_answer_in_context(item['answers'], item['rationale']))
            for item in tqdm(data, desc="Evaluating")
        )
        result = {
            "accuracy": 100 * correct_count / total_examples,
            "num_examples": total_examples
        }
    
    if save_dir:
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(result, f)
    
    return result

# # F1 Metric Calculation
# class F1ScoreCalculator:
#     @staticmethod
#     def calculate_precision_recall_f1(predicted_tokens: List[str], reference_tokens: List[str]) -> tuple:
#         """Calculate precision, recall, and F1 score between token lists"""
#         common_tokens = Counter(reference_tokens) & Counter(predicted_tokens)
#         num_common = sum(common_tokens.values())
        
#         if num_common == 0:
#             return 0.0, 0.0, 0.0
        
#         precision = num_common / len(predicted_tokens)
#         recall = num_common / len(reference_tokens)
#         f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#         return precision, recall, f1

#     @staticmethod
#     def calculate_for_pair(prediction: str, reference: str) -> tuple:
#         """Calculate F1 metrics for a single prediction-reference pair"""
#         if not reference:
#             return None, None, None
#         if not prediction:
#             return 0.0, 0.0, 0.0
            
#         pred_tokens = normalize_answer(prediction).split()
#         ref_tokens = normalize_answer(reference).split()
#         return F1ScoreCalculator.calculate_precision_recall_f1(pred_tokens, ref_tokens)

#     @staticmethod
#     def calculate_for_dataset(predictions: List[str], references: List[List[str]]) -> tuple:
#         """Calculate F1 metrics for a dataset"""
#         assert len(predictions) == len(references)
        
#         precision_scores = []
#         recall_scores = []
#         f1_scores = []
        
#         for pred, ref_list in zip(predictions, references):
#             best_f1 = 0.0
#             best_precision = 0.0
#             best_recall = 0.0
            
#             for reference in ref_list:
#                 reference = reference.strip()
#                 if not reference:
#                     continue
                    
#                 precision, recall, f1 = F1ScoreCalculator.calculate_for_pair(pred, reference)
#                 if f1 is None:
#                     continue
                    
#                 if f1 > best_f1:
#                     best_f1 = f1
#                     best_precision = precision
#                     best_recall = recall
            
#             if best_f1 > 0:
#                 precision_scores.append(best_precision)
#                 recall_scores.append(best_recall)
#                 f1_scores.append(best_f1)
        
#         return (
#             np.mean(precision_scores) if precision_scores else 0.0,
#             np.mean(recall_scores) if recall_scores else 0.0,
#             np.mean(f1_scores) if f1_scores else 0.0
#         )

# # Evaluation utilities
# def load_ground_truth_data(file_path: str) -> List[List[str]]:
#     """Load ground truth answers from a JSON file"""
#     with open(file_path, "r") as f:
#         data = json.load(f)
    
#     results = []
#     for item in data:
#         if "answers" in item:
#             results.append(item["answers"])
#         elif "answer" in item:
#             results.append([item["answer"] if isinstance(item["answer"], str) else str(item["answer"])])
#         else:
#             raise ValueError("Missing answer field in ground truth data")
#     return results

# def load_predicted_answers(file_path: str) -> List[str]:
#     """Load predicted answers from a text file"""
#     with open(file_path, "r") as f:
#         return [line.strip() for line in f]

# def evaluate_f1_score(ground_truth_file: str, prediction_file: str, dataset_name: str = "default"):
#     """Evaluate F1 score between ground truth and predictions"""
#     ground_truth = load_ground_truth_data(ground_truth_file)
#     predictions = load_predicted_answers(prediction_file)
    
#     if "inscit" in ground_truth_file:
#         ground_truth = [
#             [ans for ans in answers if "Sorry. I cannot find the answer" not in ans]
#             for answers in ground_truth
#         ]
    
#     precision, recall, f1 = F1ScoreCalculator.calculate_for_dataset(predictions, ground_truth)
#     print(f"Dataset: {dataset_name} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# def evaluate_unanswerable_questions(ground_truth_file: str, prediction_file: str):
#     """Evaluate performance on answerable and unanswerable questions"""
#     ground_truth = load_ground_truth_data(ground_truth_file)
#     predictions = load_predicted_answers(prediction_file)
    
#     unanswerable_indices = []
#     answerable_indices = []
#     unanswerable_response = "Sorry. I cannot find the answer based on the context."
    
#     for i, answers in enumerate(ground_truth):
#         if any(unanswerable_response in ans for ans in answers):
#             unanswerable_indices.append(i)
#         else:
#             answerable_indices.append(i)
    
#     print(f"Unanswerable questions: {len(unanswerable_indices)}/{len(ground_truth)}")
#     print(f"Answerable questions: {len(answerable_indices)}/{len(ground_truth)}")
    
#     # Calculate unanswerable accuracy
#     unanswerable_correct = sum(
#         "sorry" in predictions[i].lower() and "cannot find" in predictions[i].lower()
#         for i in unanswerable_indices
#     )
#     unanswerable_acc = unanswerable_correct / len(unanswerable_indices) if unanswerable_indices else 0.0
    
#     # Calculate answerable accuracy
#     answerable_correct = sum(
#         not ("sorry" in predictions[i].lower() and "cannot find" in predictions[i].lower())
#         for i in answerable_indices
#     )
#     answerable_acc = answerable_correct / len(answerable_indices) if answerable_indices else 0.0
    
#     print(f"Unanswerable accuracy: {unanswerable_acc:.4f}")
#     print(f"Answerable accuracy: {answerable_acc:.4f}")

# # Financial QA evaluation
# def evaluate_financial_qa(ground_truth_file: str, prediction_file: str):
#     """Specialized evaluation for financial QA datasets"""
#     def is_float(value: str) -> bool:
#         try:
#             float(value)
#             return True
#         except ValueError:
#             return False

#     with open(ground_truth_file, "r") as f:
#         ground_truth = json.load(f)
    
#     correct_count = 0
#     predictions = load_predicted_answers(prediction_file)
    
#     for item, pred in zip(ground_truth, predictions):
#         gold_value = item.get('exe_answer', '')
#         gold_formula = item.get('answers', [''])[0]
#         question = item.get('messages', [{}])[-1].get('content', '')
        
#         # Clean prediction
#         cleaned_pred = pred.lower()
#         cleaned_pred = re.sub(r'[,$]|million|billion|\(\w+\)', '', cleaned_pred)
#         cleaned_pred = ' '.join(cleaned_pred.split())
        
#         # Check for matches
#         if any(match in cleaned_pred for match in [str(gold_value), gold_formula]):
#             correct_count += 1
#         elif is_float(gold_value) and any(
#             str(round(float(gold_value), precision)) in cleaned_pred
#             for precision in [2, 3]
#         ):
#             correct_count += 1
#         elif "percent" in question and any(
#             str(float(gold_value) * 100) in cleaned_pred or
#             str(round(float(gold_value) * 100, precision)) in cleaned_pred
#             for precision in [1, 2]
#         ):
#             correct_count += 1
#         elif str(gold_value).endswith(".0") and str(int(float(gold_value))) in cleaned_pred:
#             correct_count += 1
#         elif "decrease" in cleaned_pred and is_float(gold_value) and float(gold_value) < 0:
#             if str(-1 * float(gold_value)) in cleaned_pred:
#                 correct_count += 1
    
#     accuracy = correct_count / len(predictions)
#     print(f"Financial QA accuracy: {accuracy:.4f}")

# # Comprehensive evaluation
# def evaluate_all_metrics(ground_truth_file: str, prediction_file: str, model_outputs: Optional[Dict] = None) -> Dict:
#     """Evaluate multiple metrics for predictions"""
#     ground_truth = load_ground_truth_data(ground_truth_file)
#     predictions = load_predicted_answers(prediction_file)
    
#     # Calculate F1 metrics
#     precision, recall, f1 = F1ScoreCalculator.calculate_for_dataset(predictions, ground_truth)
    
#     # Prepare results
#     results = {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "num_examples": len(predictions)
#     }
    
#     # Add perplexity if model outputs are available
#     if model_outputs and 'logits' in model_outputs and 'labels' in model_outputs:
#         results["perplexity"] = PerplexityCalculator.compute(
#             model_outputs['logits'], model_outputs['labels']
#         )
    
#     print("\n" + "=" * 50)
#     print("Evaluation Results:")
#     print("=" * 50)
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     if "perplexity" in results:
#         print(f"Perplexity: {results['perplexity']:.4f}")
#     print("=" * 50)
    
#     return results

# # Perplexity calculation (simplified)
# class PerplexityCalculator:
#     @staticmethod
#     def compute(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
#         """Calculate perplexity from model outputs"""
#         # Implementation would go here
#         return 0.0  # Placeholder