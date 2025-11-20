#!/usr/bin/env python3
"""
Evaluate Quality Score Correlation (SRCC/PLCC)

Calculates Spearman Rank Correlation Coefficient (SRCC) and Pearson Linear 
Correlation Coefficient (PLCC) for quality scoring tasks on datasets like 
TID2013, BID, AGIQA-3K.

Usage:
    python scripts/eval_correlation.py --input results/output.jsonl --ground-truth data/mos.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

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


def extract_quality_score(result: Dict) -> float:
    """
    Extract quality score from pipeline result.
    
    Tries multiple locations:
    1. summarizer_result.final_answer (if numeric)
    2. fused_score from score fusion
    3. tool scores average
    """
    summarizer_result = result.get('summarizer_result', {})
    
    # Try final_answer if it's numeric
    if isinstance(summarizer_result, dict):
        final_answer = summarizer_result.get('final_answer', '')
        
        # Try to parse as number
        try:
            score = float(final_answer)
            if 1.0 <= score <= 5.0:
                return score
        except (ValueError, TypeError):
            pass
        
        # Try letter grade mapping
        letter_to_score = {'A': 5.0, 'B': 4.0, 'C': 3.0, 'D': 2.0, 'E': 1.0}
        if final_answer.strip().upper() in letter_to_score:
            return letter_to_score[final_answer.strip().upper()]
    
    # Try fused_score
    if 'fused_score' in result:
        return float(result['fused_score'])
    
    # Try tool scores average
    executor_evidence = result.get('executor_evidence', {})
    if isinstance(executor_evidence, dict):
        quality_scores = executor_evidence.get('quality_scores', {})
        if quality_scores:
            scores = []
            for obj_scores in quality_scores.values():
                for tool_score in obj_scores.values():
                    if isinstance(tool_score, (list, tuple)) and len(tool_score) >= 2:
                        scores.append(float(tool_score[1]))  # normalized score
                    elif isinstance(tool_score, (int, float)):
                        scores.append(float(tool_score))
            if scores:
                return np.mean(scores)
    
    return None


def calculate_correlations(predictions: np.ndarray, ground_truth: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate SRCC and PLCC with p-values.
    
    Returns:
        (srcc, srcc_pvalue, plcc, plcc_pvalue)
    """
    if len(predictions) == 0 or len(ground_truth) == 0:
        return 0.0, 1.0, 0.0, 1.0
    
    # Spearman Rank Correlation
    srcc, srcc_pvalue = stats.spearmanr(predictions, ground_truth)
    
    # Pearson Linear Correlation
    plcc, plcc_pvalue = stats.pearsonr(predictions, ground_truth)
    
    return srcc, srcc_pvalue, plcc, plcc_pvalue


def main():
    parser = argparse.ArgumentParser(description='Evaluate quality score correlation (SRCC/PLCC)')
    parser.add_argument('--input', '-i', type=Path, required=True, help='Input JSONL file with predictions')
    parser.add_argument('--ground-truth', '-g', type=Path, help='Ground truth JSONL file with MOS scores')
    parser.add_argument('--mos-field', default='mos', help='Field name for MOS in ground truth (default: mos)')
    parser.add_argument('--output', '-o', type=Path, help='Save results to JSON file')
    
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
    for result in results:
        sample_id = result.get('sample_id')
        if not sample_id:
            continue
        
        score = extract_quality_score(result)
        if score is not None:
            predictions[sample_id] = score
    
    print(f"Extracted {len(predictions)} quality scores")
    
    # Load ground truth
    if args.ground_truth:
        print(f"Loading ground truth from {args.ground_truth}...")
        gt_records = load_jsonl(args.ground_truth)
        ground_truth = {r['sample_id']: float(r[args.mos_field]) for r in gt_records if args.mos_field in r}
    else:
        # Try to extract from input file
        print("Extracting ground truth from input file...")
        ground_truth = {}
        for r in results:
            sample_id = r.get('sample_id')
            if sample_id and args.mos_field in r:
                ground_truth[sample_id] = float(r[args.mos_field])
    
    if not ground_truth:
        print(f"Error: No ground truth found. Provide --ground-truth file or include '{args.mos_field}' field in input.")
        return 1
    
    print(f"Loaded {len(ground_truth)} ground truth MOS values")
    
    # Match predictions with ground truth
    matched_sample_ids = []
    matched_predictions = []
    matched_ground_truth = []
    
    for sample_id in predictions:
        if sample_id in ground_truth:
            matched_sample_ids.append(sample_id)
            matched_predictions.append(predictions[sample_id])
            matched_ground_truth.append(ground_truth[sample_id])
    
    if not matched_predictions:
        print("Error: No matched samples found between predictions and ground truth")
        return 1
    
    print(f"Matched {len(matched_predictions)} samples")
    
    # Calculate correlations
    pred_array = np.array(matched_predictions)
    gt_array = np.array(matched_ground_truth)
    
    srcc, srcc_pval, plcc, plcc_pval = calculate_correlations(pred_array, gt_array)
    
    # Print results
    print("\n" + "=" * 70)
    print("Quality Score Correlation Evaluation Results")
    print("=" * 70)
    print(f"\nNumber of samples: {len(matched_predictions)}")
    print(f"Prediction range: [{pred_array.min():.3f}, {pred_array.max():.3f}]")
    print(f"Ground truth range: [{gt_array.min():.3f}, {gt_array.max():.3f}]")
    print(f"\nSpearman Rank Correlation (SRCC): {srcc:.4f} (p={srcc_pval:.4e})")
    print(f"Pearson Linear Correlation (PLCC): {plcc:.4f} (p={plcc_pval:.4e})")
    print("=" * 70 + "\n")
    
    # Interpret results
    if srcc_pval < 0.05:
        print(f"✓ SRCC is statistically significant (p < 0.05)")
    else:
        print(f"✗ SRCC is not statistically significant (p >= 0.05)")
    
    if plcc_pval < 0.05:
        print(f"✓ PLCC is statistically significant (p < 0.05)")
    else:
        print(f"✗ PLCC is not statistically significant (p >= 0.05)")
    
    print()
    
    # Save results if requested
    if args.output:
        results_data = {
            'n_samples': len(matched_predictions),
            'prediction_stats': {
                'min': float(pred_array.min()),
                'max': float(pred_array.max()),
                'mean': float(pred_array.mean()),
                'std': float(pred_array.std())
            },
            'ground_truth_stats': {
                'min': float(gt_array.min()),
                'max': float(gt_array.max()),
                'mean': float(gt_array.mean()),
                'std': float(gt_array.std())
            },
            'correlations': {
                'srcc': float(srcc),
                'srcc_pvalue': float(srcc_pval),
                'plcc': float(plcc),
                'plcc_pvalue': float(plcc_pval)
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {args.output}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
