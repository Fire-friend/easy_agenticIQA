#!/usr/bin/env python3
"""
Generate JSONL manifest for BID No-Reference IQA dataset.

BID Structure:
  bid/
    DatabaseGradedImages/
      img001.bmp
      ...
    AllImages_release.txt or mos.csv

Usage:
    python generate_bid_manifest.py \\
        --data-dir data/raw/bid \\
        --output data/processed/bid/manifest.jsonl \\
        --split test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_mos_file(mos_file: Path) -> Dict[str, float]:
    """
    Parse MOS file for BID dataset.

    Format can vary, but typically:
      img001.bmp 3.45
      OR
      001,3.45

    Args:
        mos_file: Path to MOS annotations file

    Returns:
        Dictionary mapping filename -> MOS score
    """
    mos_dict = {}

    with open(mos_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try space-separated: "img001.bmp 3.45"
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[0]
                try:
                    mos = float(parts[1])
                    mos_dict[filename] = mos
                    continue
                except ValueError:
                    pass

            # Try comma-separated: "001,3.45"
            parts = line.split(',')
            if len(parts) >= 2:
                image_id = parts[0].strip()
                try:
                    mos = float(parts[1].strip())
                    # Assume filename format: img<id>.bmp
                    filename = f"img{image_id.zfill(3)}.bmp"
                    mos_dict[filename] = mos
                    continue
                except ValueError:
                    pass

            logger.warning(f"Could not parse line {line_num}: {line}")

    logger.info(f"Parsed {len(mos_dict)} MOS scores from {mos_file}")
    return mos_dict


def generate_manifest(
    data_dir: Path,
    output: Path,
    split: str = 'test',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    resume: bool = False
) -> int:
    """
    Generate JSONL manifest for BID dataset.

    Args:
        data_dir: Root directory of BID dataset
        output: Output JSONL manifest path
        split: Dataset split ('train', 'val', 'test')
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        resume: Skip already processed samples

    Returns:
        Number of samples written
    """
    # Locate image directory and MOS file
    image_dir = data_dir / 'DatabaseGradedImages'
    if not image_dir.exists():
        image_dir = data_dir / 'images'

    if not image_dir.exists():
        logger.error(f"Image directory not found in {data_dir}")
        return 0

    # Try multiple MOS file names
    mos_file = None
    for candidate in ['AllImages_release.txt', 'mos.csv', 'mos.txt', 'annotations.txt']:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            mos_file = candidate_path
            break

    if not mos_file:
        logger.error(f"MOS file not found in {data_dir}")
        return 0

    # Parse MOS scores
    mos_dict = parse_mos_file(mos_file)

    # Get list of images
    image_files = sorted(image_dir.glob('*.bmp')) + sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png'))
    logger.info(f"Found {len(image_files)} images")

    # Load existing sample IDs if resuming
    processed_ids = set()
    if resume and output.exists():
        with open(output, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(entry['sample_id'])
        logger.info(f"Resuming: Skipping {len(processed_ids)} already-processed samples")

    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    # Split samples deterministically
    total_samples = len(image_files)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Open output file
    mode = 'a' if resume else 'w'
    sample_count = 0

    with open(output, mode, encoding='utf-8') as f:
        for idx, image_path in enumerate(image_files, 1):
            filename = image_path.name

            # Determine split
            if idx <= train_end:
                current_split = 'train'
            elif idx <= val_end:
                current_split = 'val'
            else:
                current_split = 'test'

            # Skip if not matching requested split
            if split != 'all' and current_split != split:
                continue

            # Check if already processed
            sample_id = f"bid_{idx:04d}"
            if sample_id in processed_ids:
                continue

            # Get MOS score
            mos = mos_dict.get(filename)
            if mos is None:
                logger.warning(f"No MOS score for {filename}, skipping")
                continue

            # Create manifest entry
            entry = {
                'sample_id': sample_id,
                'dataset': 'bid',
                'image_path': str(image_path.relative_to(data_dir.parent)),
                'mos': round(mos, 3),
                'split': current_split,
                'metadata': {
                    'image_id': filename
                }
            }

            # Write to output
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            sample_count += 1

            if sample_count % 100 == 0:
                logger.info(f"Processed {sample_count} samples...")

    logger.info(f"✓ Generated manifest with {sample_count} samples: {output}")
    return sample_count


def main():
    parser = argparse.ArgumentParser(description="Generate BID manifest")
    parser.add_argument('--data-dir', type=Path, required=True, help="BID dataset root directory")
    parser.add_argument('--output', type=Path, required=True, help="Output JSONL manifest path")
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all', help="Dataset split")
    parser.add_argument('--train-ratio', type=float, default=0.7, help="Training set ratio")
    parser.add_argument('--val-ratio', type=float, default=0.15, help="Validation set ratio")
    parser.add_argument('--resume', action='store_true', help="Resume from existing manifest")

    args = parser.parse_args()

    count = generate_manifest(
        data_dir=args.data_dir,
        output=args.output,
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        resume=args.resume
    )

    if count > 0:
        logger.info(f"Success: {count} samples written to {args.output}")
    else:
        logger.error("Failed to generate manifest")
        exit(1)


if __name__ == '__main__':
    main()
