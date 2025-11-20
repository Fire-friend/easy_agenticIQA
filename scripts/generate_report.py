#!/usr/bin/env python3
"""
Generate comprehensive evaluation report for AgenticIQA.

Aggregates results from multiple evaluation runs and generates a structured
Markdown report with:
  - Environment and configuration info
  - Dataset statistics
  - MCQ accuracy results
  - Correlation metrics with confidence intervals
  - Cost and performance analysis

Usage:
    python scripts/generate_report.py \\
        --output-dir outputs \\
        --report reports/reproduction_report.md
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def collect_results(output_dir: Path) -> Dict[str, Any]:
    """
    Collect all evaluation results from output directory.

    Args:
        output_dir: Directory containing evaluation outputs

    Returns:
        Dictionary with aggregated results
    """
    results = {
        'mcq': {},
        'correlation': {},
        'raw_outputs': []
    }

    # Collect MCQ results
    mcq_files = list(output_dir.glob('*_mcq_*.json'))
    for mcq_file in mcq_files:
        dataset_name = mcq_file.stem.replace('_mcq', '').replace('_results', '')
        with open(mcq_file, 'r') as f:
            results['mcq'][dataset_name] = json.load(f)

    # Collect correlation results
    corr_files = list(output_dir.glob('*_ci.json')) + list(output_dir.glob('*_correlation.json'))
    for corr_file in corr_files:
        dataset_name = corr_file.stem.replace('_ci', '').replace('_correlation', '')
        with open(corr_file, 'r') as f:
            results['correlation'][dataset_name] = json.load(f)

    # Collect raw outputs
    output_files = list(output_dir.glob('*.jsonl'))
    for output_file in output_files:
        results['raw_outputs'].append(str(output_file))

    logger.info(f"Collected {len(results['mcq'])} MCQ results")
    logger.info(f"Collected {len(results['correlation'])} correlation results")
    logger.info(f"Found {len(results['raw_outputs'])} raw output files")

    return results


def format_environment_section() -> str:
    """Generate environment and configuration section."""
    import platform
    import sys

    section = []
    section.append("## 1. Environment & Configuration\n")
    section.append(f"- **Report Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    section.append(f"- **Python Version**: {sys.version.split()[0]}")
    section.append(f"- **Platform**: {platform.platform()}")
    section.append(f"- **System**: {platform.system()} {platform.release()}")
    section.append("\n")

    return "\n".join(section)


def format_mcq_results_section(mcq_results: Dict) -> str:
    """Format MCQ accuracy results as Markdown table."""
    section = []
    section.append("## 2. AgenticIQA-Eval MCQ Results\n")

    if not mcq_results:
        section.append("_No MCQ results found._\n")
        return "\n".join(section)

    # Overall results table
    section.append("### Overall Accuracy\n")
    section.append("| Dataset | Accuracy | Correct | Total |")
    section.append("|---------|----------|---------|-------|")

    for dataset, data in mcq_results.items():
        overall = data.get('overall', {})
        acc = overall.get('accuracy', 0)
        correct = overall.get('correct', 0)
        total = overall.get('total', 0)
        section.append(f"| {dataset} | {acc:.2f}% | {correct} | {total} |")

    section.append("\n")

    # Per-category breakdown (if available)
    for dataset, data in mcq_results.items():
        per_cat = data.get('per_category', {})
        if per_cat:
            section.append(f"### {dataset} - Per-Category Breakdown\n")
            section.append("| Category | Accuracy | Correct | Total |")
            section.append("|----------|----------|---------|-------|")

            for category, stats in sorted(per_cat.items()):
                acc = stats.get('accuracy', 0)
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                section.append(f"| {category} | {acc:.2f}% | {correct} | {total} |")

            section.append("\n")

    return "\n".join(section)


def format_correlation_results_section(corr_results: Dict) -> str:
    """Format correlation metrics as Markdown table."""
    section = []
    section.append("## 3. Correlation Metrics (SRCC/PLCC)\n")

    if not corr_results:
        section.append("_No correlation results found._\n")
        return "\n".join(section)

    section.append("| Dataset | Metric | Value | 95% CI | P-value | Samples |")
    section.append("|---------|--------|-------|--------|---------|---------|")

    for dataset, data in corr_results.items():
        n_samples = data.get('n_samples', 0)

        # SRCC row
        if 'srcc' in data:
            srcc = data['srcc']
            ci_lower = data.get('srcc_ci_lower', 0)
            ci_upper = data.get('srcc_ci_upper', 0)
            pvalue = data.get('srcc_pvalue', 0)
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            pval_str = f"{pvalue:.2e}" if pvalue < 0.001 else f"{pvalue:.4f}"
            section.append(f"| {dataset} | SRCC | {srcc:.4f} | {ci_str} | {pval_str} | {n_samples} |")

        # PLCC row
        if 'plcc' in data:
            plcc = data['plcc']
            ci_lower = data.get('plcc_ci_lower', 0)
            ci_upper = data.get('plcc_ci_upper', 0)
            pvalue = data.get('plcc_pvalue', 0)
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            pval_str = f"{pvalue:.2e}" if pvalue < 0.001 else f"{pvalue:.4f}"
            section.append(f"| {dataset} | PLCC | {plcc:.4f} | {ci_str} | {pval_str} | {n_samples} |")

    section.append("\n")

    return "\n".join(section)


def format_summary_section(results: Dict) -> str:
    """Generate summary and conclusion section."""
    section = []
    section.append("## 4. Summary\n")

    # Count total datasets evaluated
    mcq_count = len(results['mcq'])
    corr_count = len(results['correlation'])

    section.append(f"- **MCQ Datasets Evaluated**: {mcq_count}")
    section.append(f"- **Correlation Datasets Evaluated**: {corr_count}")
    section.append(f"- **Total Output Files**: {len(results['raw_outputs'])}")

    section.append("\n")
    section.append("### Key Findings\n")
    section.append("_Add analysis and interpretation of results here._\n")

    return "\n".join(section)


def generate_report(output_dir: Path, report_path: Path):
    """
    Generate comprehensive evaluation report.

    Args:
        output_dir: Directory with evaluation outputs
        report_path: Path to save Markdown report
    """
    logger.info("Collecting evaluation results...")
    results = collect_results(output_dir)

    logger.info("Generating report sections...")
    report = []

    # Title
    report.append("# AgenticIQA Evaluation Report\n")
    report.append(f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_\n")
    report.append("---\n")

    # Sections
    report.append(format_environment_section())
    report.append(format_mcq_results_section(results['mcq']))
    report.append(format_correlation_results_section(results['correlation']))
    report.append(format_summary_section(results))

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = "\n".join(report)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    logger.info(f"✓ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                        help="Directory containing evaluation outputs")
    parser.add_argument('--report', type=Path, default=Path('reports/evaluation_report.md'),
                        help="Output report path")

    args = parser.parse_args()

    if not args.output_dir.exists():
        logger.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)

    generate_report(args.output_dir, args.report)
    logger.info("Report generation complete!")


if __name__ == '__main__':
    main()
