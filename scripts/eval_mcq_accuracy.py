#!/usr/bin/env python3
"""
Evaluate MCQ (Multiple Choice Question) Accuracy with Confusion Matrix

Calculates accuracy for AgenticIQA-Eval dataset by comparing predicted answers
against ground truth. Includes:
  - Overall and per-category accuracy
  - Confusion matrix analysis
  - Per-option precision and recall
  - Most confused option pairs

Usage:
    python scripts/eval_mcq_accuracy.py --input results/output.jsonl --ground-truth data/ground_truth.jsonl --confusion
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    return records


def extract_final_answer(result: Dict) -> str:
    """Extract final answer from pipeline result."""
    summarizer_result = result.get('summarizer_result', {})
    
    if isinstance(summarizer_result, dict):
        return summarizer_result.get('final_answer', '').strip()
    
    # Handle string representation
    if isinstance(summarizer_result, str):
        return summarizer_result.strip()
    
    return ''


def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> Tuple[float, int, int]:
    """Calculate accuracy percentage and counts."""
    if not predictions or not ground_truth:
        return 0.0, 0, 0

    if len(predictions) != len(ground_truth):
        print(f"Warning: Prediction count ({len(predictions)}) != Ground truth count ({len(ground_truth)})")

    correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.upper() == gt.upper())
    total = len(predictions)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    return accuracy, correct, total


def generate_confusion_matrix(predictions: List[str], ground_truth: List[str], options: List[str] = ['A', 'B', 'C', 'D']) -> np.ndarray:
    """
    Generate confusion matrix for MCQ predictions.

    Args:
        predictions: List of predicted answers
        ground_truth: List of correct answers
        options: List of valid options (default: A, B, C, D)

    Returns:
        Confusion matrix as numpy array (rows=ground truth, cols=predictions)
    """
    n_options = len(options)
    option_to_idx = {opt.upper(): i for i, opt in enumerate(options)}

    confusion = np.zeros((n_options, n_options), dtype=int)

    for pred, gt in zip(predictions, ground_truth):
        pred_upper = pred.upper()
        gt_upper = gt.upper()

        if pred_upper in option_to_idx and gt_upper in option_to_idx:
            gt_idx = option_to_idx[gt_upper]
            pred_idx = option_to_idx[pred_upper]
            confusion[gt_idx, pred_idx] += 1

    return confusion


def calculate_precision_recall(confusion: np.ndarray, options: List[str] = ['A', 'B', 'C', 'D']) -> Dict:
    """
    Calculate per-option precision and recall from confusion matrix.

    Args:
        confusion: Confusion matrix
        options: List of option labels

    Returns:
        Dictionary with precision and recall per option
    """
    precision = {}
    recall = {}

    for i, opt in enumerate(options):
        # Precision = diagonal / column sum
        col_sum = confusion[:, i].sum()
        precision[opt] = float(confusion[i, i] / col_sum) if col_sum > 0 else 0.0

        # Recall = diagonal / row sum
        row_sum = confusion[i, :].sum()
        recall[opt] = float(confusion[i, i] / row_sum) if row_sum > 0 else 0.0

    return {'precision': precision, 'recall': recall}


def find_most_confused_pairs(confusion: np.ndarray, options: List[str] = ['A', 'B', 'C', 'D'], top_k: int = 5) -> List[Tuple]:
    """
    Identify most confused option pairs (ground truth -> predicted).

    Args:
        confusion: Confusion matrix
        options: List of option labels
        top_k: Number of top confused pairs to return

    Returns:
        List of (ground_truth, predicted, count) tuples
    """
    confused_pairs = []

    for i in range(len(options)):
        for j in range(len(options)):
            if i != j:  # Skip diagonal (correct predictions)
                count = int(confusion[i, j])
                if count > 0:
                    confused_pairs.append((options[i], options[j], count))

    # Sort by count descending
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return confused_pairs[:top_k]


def print_confusion_matrix(confusion: np.ndarray, options: List[str] = ['A', 'B', 'C', 'D']):
    """Print confusion matrix in ASCII table format."""
    print("\nConfusion Matrix (rows=ground truth, cols=predicted):")
    print("-" * 60)

    # Header
    print("GT\\Pred  ", end="")
    for opt in options:
        print(f"{opt:>6s}", end="")
    print()

    # Rows
    for i, opt in enumerate(options):
        print(f"  {opt:2s}    ", end="")
        for j in range(len(options)):
            print(f"{confusion[i, j]:6d}", end="")
        print()

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MCQ accuracy with confusion matrix')
    parser.add_argument('--input', '-i', type=Path, required=True, help='Input JSONL file with predictions')
    parser.add_argument('--ground-truth', '-g', type=Path, help='Ground truth JSONL file (optional if included in input)')
    parser.add_argument('--category-field', default='category', help='Field name for category (default: category)')
    parser.add_argument('--output', '-o', type=Path, help='Save results to JSON file')
    parser.add_argument('--confusion', action='store_true', help='Generate confusion matrix analysis')
    parser.add_argument('--options', nargs='+', default=['A', 'B', 'C', 'D'], help='MCQ options (default: A B C D)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Load predictions
    print(f"Loading predictions from {args.input}...")
    results = load_jsonl(args.input)
    print(f"Loaded {len(results)} results")
    
    # Build predictions dict
    predictions = {}
    categories = {}
    
    for result in results:
        sample_id = result.get('sample_id')
        if not sample_id:
            continue
        
        final_answer = extract_final_answer(result)
        predictions[sample_id] = final_answer
        
        # Extract category if available
        category = result.get(args.category_field, 'Unknown')
        categories[sample_id] = category
    
    # Load ground truth
    if args.ground_truth:
        print(f"Loading ground truth from {args.ground_truth}...")
        gt_records = load_jsonl(args.ground_truth)
        ground_truth = {r['sample_id']: r.get('correct_answer', r.get('answer', '')) for r in gt_records}
    else:
        # Try to extract from input file
        print("Extracting ground truth from input file...")
        ground_truth = {r['sample_id']: r.get('correct_answer', r.get('answer', '')) for r in results if 'correct_answer' in r or 'answer' in r}
    
    if not ground_truth:
        print("Error: No ground truth found. Provide --ground-truth file or include 'correct_answer' field in input.")
        return 1
    
    print(f"Loaded {len(ground_truth)} ground truth answers")
    
    # Match predictions with ground truth
    matched_predictions = []
    matched_ground_truth = []
    matched_categories = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            matched_predictions.append(predictions[sample_id])
            matched_ground_truth.append(ground_truth[sample_id])
            matched_categories.append(categories.get(sample_id, 'Unknown'))
    
    print(f"Matched {len(matched_predictions)} samples")
    
    # Calculate overall accuracy
    overall_acc, overall_correct, overall_total = calculate_accuracy(matched_predictions, matched_ground_truth)
    
    # Calculate per-category accuracy
    category_stats = defaultdict(lambda: {'predictions': [], 'ground_truth': []})
    
    for pred, gt, cat in zip(matched_predictions, matched_ground_truth, matched_categories):
        category_stats[cat]['predictions'].append(pred)
        category_stats[cat]['ground_truth'].append(gt)
    
    category_results = {}
    for category, data in category_stats.items():
        acc, correct, total = calculate_accuracy(data['predictions'], data['ground_truth'])
        category_results[category] = {
            'accuracy': acc,
            'correct': correct,
            'total': total
        }
    
    # Print results
    print("\n" + "=" * 70)
    print("MCQ Accuracy Evaluation Results")
    print("=" * 70)
    print(f"\nOverall Accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")
    
    if category_results:
        print("\nPer-Category Accuracy:")
        print("-" * 70)
        for category in sorted(category_results.keys()):
            stats = category_results[category]
            print(f"  {category:20s}: {stats['accuracy']:6.2f}% ({stats['correct']:3d}/{stats['total']:3d})")

    print("=" * 70 + "\n")

    # Generate confusion matrix if requested
    confusion_analysis = None
    if args.confusion:
        print("\n" + "=" * 70)
        print("Confusion Matrix Analysis")
        print("=" * 70)

        confusion = generate_confusion_matrix(matched_predictions, matched_ground_truth, args.options)
        print_confusion_matrix(confusion, args.options)

        # Calculate precision/recall
        metrics = calculate_precision_recall(confusion, args.options)
        print("\nPer-Option Metrics:")
        print("-" * 60)
        print(f"{'Option':>8s} {'Precision':>12s} {'Recall':>12s}")
        print("-" * 60)
        for opt in args.options:
            prec = metrics['precision'].get(opt, 0.0)
            rec = metrics['recall'].get(opt, 0.0)
            print(f"{opt:>8s} {prec:>12.2%} {rec:>12.2%}")

        # Find most confused pairs
        confused_pairs = find_most_confused_pairs(confusion, args.options)
        if confused_pairs:
            print("\nMost Confused Pairs (Ground Truth → Predicted):")
            print("-" * 60)
            for gt, pred, count in confused_pairs:
                print(f"  {gt} → {pred}: {count} errors")

        print("=" * 70 + "\n")

        # Store for JSON output
        confusion_analysis = {
            'confusion_matrix': confusion.tolist(),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'most_confused_pairs': [{'ground_truth': gt, 'predicted': pred, 'count': count}
                                     for gt, pred, count in confused_pairs]
        }

    # Save results if requested
    if args.output:
        results_data = {
            'overall': {
                'accuracy': overall_acc,
                'correct': overall_correct,
                'total': overall_total
            },
            'per_category': category_results
        }

        if confusion_analysis:
            results_data['confusion_analysis'] = confusion_analysis

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
