#!/usr/bin/env python3
"""
Generate JSONL manifests for AgenticIQA-Eval MCQ dataset.

Splits into four task-specific manifests:
  - planner.jsonl
  - executor_distortion.jsonl
  - executor_tool.jsonl
  - summarizer.jsonl

Usage:
    python generate_agenticiqa_eval_manifest.py \\
        --data-dir data/raw/agenticiqa_eval \\
        --output-dir data/processed/agenticiqa_eval
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_mcq_questions(mcq_file: Path) -> List[Dict]:
    """
    Load MCQ questions from JSON or JSONL file.

    Expected format (JSON):
    [
      {
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "answer": "A",
        "task_type": "planner",
        "reference_mode": "NR",
        ...
      }
    ]

    Or JSONL (one question per line).

    Args:
        mcq_file: Path to MCQ questions file

    Returns:
        List of question dictionaries
    """
    questions = []

    if mcq_file.suffix == '.json':
        with open(mcq_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    else:  # Assume JSONL
        with open(mcq_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

    logger.info(f"Loaded {len(questions)} MCQ questions from {mcq_file}")
    return questions


def generate_manifests(
    data_dir: Path,
    output_dir: Path,
    split: str = 'test'
) -> Dict[str, int]:
    """
    Generate task-specific JSONL manifests for AgenticIQA-Eval.

    Args:
        data_dir: Root directory containing MCQ questions
        output_dir: Output directory for manifests
        split: Dataset split

    Returns:
        Dictionary mapping task_type -> sample count
    """
    # Locate MCQ questions file
    mcq_file = data_dir / 'questions.json'
    if not mcq_file.exists():
        mcq_file = data_dir / 'questions.jsonl'

    if not mcq_file.exists():
        logger.error(f"MCQ questions file not found in {data_dir}")
        return {}

    # Load questions
    questions = load_mcq_questions(mcq_file)

    # Group by task type
    task_groups = {
        'planner': [],
        'executor_distortion': [],
        'executor_tool': [],
        'summarizer': []
    }

    for q in questions:
        task_type = q.get('task_type', 'unknown')
        if task_type in task_groups:
            task_groups[task_type].append(q)
        else:
            logger.warning(f"Unknown task_type: {task_type}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write task-specific manifests
    counts = {}

    for task_type, task_questions in task_groups.items():
        output_file = output_dir / f"{task_type}.jsonl"
        sample_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, q in enumerate(task_questions, 1):
                # Create standardized entry
                entry = {
                    'sample_id': f"agenticiqa_eval_{task_type}_{idx:04d}",
                    'dataset': 'agenticiqa-eval',
                    'question': q.get('question', ''),
                    'options': q.get('options', {}),
                    'answer': q.get('answer', ''),
                    'task_type': task_type,
                    'reference_mode': q.get('reference_mode', 'NR'),
                    'split': split
                }

                # Optional fields
                if 'image_path' in q:
                    entry['image_path'] = q['image_path']
                if 'reference_path' in q:
                    entry['reference_path'] = q['reference_path']

                # Metadata
                metadata = {}
                if 'question_type' in q:
                    metadata['question_type'] = q['question_type']
                if 'difficulty' in q:
                    metadata['difficulty'] = q['difficulty']
                if 'distortion_type' in q:
                    metadata['distortion_type'] = q['distortion_type']

                if metadata:
                    entry['metadata'] = metadata

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                sample_count += 1

        counts[task_type] = sample_count
        logger.info(f"✓ {task_type}: {sample_count} questions → {output_file}")

    return counts


def main():
    parser = argparse.ArgumentParser(description="Generate AgenticIQA-Eval manifests")
    parser.add_argument('--data-dir', type=Path, required=True, help="AgenticIQA-Eval dataset root")
    parser.add_argument('--output-dir', type=Path, required=True, help="Output directory for manifests")
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')

    args = parser.parse_args()

    counts = generate_manifests(args.data_dir, args.output_dir, args.split)

    if counts:
        total = sum(counts.values())
        logger.info(f"\nSuccess: Generated {len(counts)} manifests with {total} total questions")
        for task_type, count in counts.items():
            logger.info(f"  - {task_type}: {count}")
    else:
        logger.error("Failed to generate manifests")
        exit(1)


if __name__ == '__main__':
    main()
