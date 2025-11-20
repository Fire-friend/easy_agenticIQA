#!/usr/bin/env python3
"""
Evaluate correlation metrics (SRCC, PLCC) with bootstrap confidence intervals.

Provides statistically rigorous evaluation with:
  - Bootstrap resampling for confidence intervals
  - P-values for significance testing
  - Parallel processing for performance
  - JSON output for report generation

Usage:
    python eval_with_ci.py \\
        --input outputs/tid2013_scores.jsonl \\
        --output results/tid2013_ci.json \\
        --n-bootstrap 1000 \\
        --confidence 0.95 \\
        --parallel
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_results(input_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load predictions and ground truth from pipeline results.

    Args:
        input_path: Path to JSONL results file

    Returns:
        (predictions, ground_truth, sample_ids)
    """
    predictions = []
    ground_truth = []
    sample_ids = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)

            # Extract predicted quality score
            pred = None
            if 'final_score' in result:
                pred = result['final_score']
            elif 'result' in result and isinstance(result['result'], dict):
                pred = result['result'].get('final_score')
            elif 'quality_score' in result:
                pred = result['quality_score']

            # Extract ground truth MOS
            gt = result.get('mos')
            if gt is None and 'metadata' in result:
                gt = result['metadata'].get('mos')

            # Extract sample_id
            sample_id = result.get('sample_id', f'sample_{len(sample_ids)}')

            if pred is not None and gt is not None:
                predictions.append(float(pred))
                ground_truth.append(float(gt))
                sample_ids.append(sample_id)

    logger.info(f"Loaded {len(predictions)} valid samples")
    return np.array(predictions), np.array(ground_truth), sample_ids


def bootstrap_correlation_worker(
    indices: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metric: str
) -> float:
    """
    Worker function for parallel bootstrap iteration.

    Args:
        indices: Bootstrap sample indices
        predictions: Full prediction array
        ground_truth: Full ground truth array
        metric: 'srcc' or 'plcc'

    Returns:
        Correlation coefficient for this bootstrap sample
    """
    pred_sample = predictions[indices]
    gt_sample = ground_truth[indices]

    if metric == 'srcc':
        corr, _ = spearmanr(pred_sample, gt_sample)
    else:  # plcc
        corr, _ = pearsonr(pred_sample, gt_sample)

    return corr


def bootstrap_correlation(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metric: str = 'srcc',
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    parallel: bool = False,
    seed: int = 42
) -> Dict:
    """
    Calculate correlation with bootstrap confidence intervals.

    Args:
        predictions: Predicted quality scores
        ground_truth: Ground truth MOS scores
        metric: 'srcc' (Spearman) or 'plcc' (Pearson)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (0.95 for 95% CI)
        parallel: Use multiprocessing
        seed: Random seed for reproducibility

    Returns:
        Dictionary with correlation, CI, and p-value
    """
    np.random.seed(seed)

    # Calculate original correlation
    if metric == 'srcc':
        corr, p_value = spearmanr(predictions, ground_truth)
    else:  # plcc
        corr, p_value = pearsonr(predictions, ground_truth)

    logger.info(f"Original {metric.upper()}: {corr:.4f} (p={p_value:.4e})")

    # Bootstrap resampling
    n_samples = len(predictions)
    bootstrap_corrs = []

    if parallel:
        # Generate all bootstrap indices upfront
        bootstrap_indices = [
            np.random.choice(n_samples, size=n_samples, replace=True)
            for _ in range(n_bootstrap)
        ]

        # Parallel processing
        n_workers = mp.cpu_count()
        logger.info(f"Running {n_bootstrap} bootstrap iterations with {n_workers} workers...")

        with mp.Pool(n_workers) as pool:
            worker = partial(
                bootstrap_correlation_worker,
                predictions=predictions,
                ground_truth=ground_truth,
                metric=metric
            )
            bootstrap_corrs = pool.map(worker, bootstrap_indices)

    else:
        # Sequential processing
        logger.info(f"Running {n_bootstrap} bootstrap iterations sequentially...")
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_sample = predictions[indices]
            gt_sample = ground_truth[indices]

            if metric == 'srcc':
                boot_corr, _ = spearmanr(pred_sample, gt_sample)
            else:
                boot_corr, _ = pearsonr(pred_sample, gt_sample)

            bootstrap_corrs.append(boot_corr)

            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{n_bootstrap}")

    bootstrap_corrs = np.array(bootstrap_corrs)

    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_corrs, lower_percentile)
    ci_upper = np.percentile(bootstrap_corrs, upper_percentile)

    logger.info(f"Bootstrap {int(confidence*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        f'{metric}': float(corr),
        f'{metric}_ci_lower': float(ci_lower),
        f'{metric}_ci_upper': float(ci_upper),
        f'{metric}_pvalue': float(p_value),
        'n_bootstrap': n_bootstrap,
        'confidence': confidence
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate correlations with bootstrap CI")
    parser.add_argument('--input', type=Path, required=True, help="Input JSONL results file")
    parser.add_argument('--output', type=Path, required=True, help="Output JSON results file")
    parser.add_argument('--metrics', nargs='+', choices=['srcc', 'plcc', 'both'], default=['both'],
                        help="Metrics to compute")
    parser.add_argument('--n-bootstrap', type=int, default=1000, help="Number of bootstrap iterations")
    parser.add_argument('--confidence', type=float, default=0.95, help="Confidence level")
    parser.add_argument('--parallel', action='store_true', help="Use parallel processing")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.input}")
    predictions, ground_truth, sample_ids = load_results(args.input)

    if len(predictions) == 0:
        logger.error("No valid samples found in input file")
        exit(1)

    # Determine which metrics to compute
    metrics_to_compute = []
    if 'both' in args.metrics:
        metrics_to_compute = ['srcc', 'plcc']
    else:
        metrics_to_compute = args.metrics

    # Compute metrics
    results = {
        'n_samples': len(predictions),
        'dataset': args.input.stem
    }

    for metric in metrics_to_compute:
        logger.info(f"\n{'='*60}")
        logger.info(f"Computing {metric.upper()} with bootstrap CI")
        logger.info(f"{'='*60}")

        metric_results = bootstrap_correlation(
            predictions=predictions,
            ground_truth=ground_truth,
            metric=metric,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            parallel=args.parallel,
            seed=args.seed
        )

        results.update(metric_results)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Results saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Dataset: {results['dataset']}")
    print(f"Samples: {results['n_samples']}")

    if 'srcc' in results:
        print(f"\nSRCC: {results['srcc']:.4f}")
        print(f"  95% CI: [{results['srcc_ci_lower']:.4f}, {results['srcc_ci_upper']:.4f}]")
        print(f"  P-value: {results['srcc_pvalue']:.4e}")

    if 'plcc' in results:
        print(f"\nPLCC: {results['plcc']:.4f}")
        print(f"  95% CI: [{results['plcc_ci_lower']:.4f}, {results['plcc_ci_upper']:.4f}]")
        print(f"  P-value: {results['plcc_pvalue']:.4e}")

    print("=" * 60)


if __name__ == '__main__':
    main()
