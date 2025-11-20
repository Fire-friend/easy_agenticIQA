#!/usr/bin/env python3
"""
Generate JSONL manifest for TID2013 Full-Reference IQA dataset.

TID2013 Structure:
  tid2013/
    distorted_images/
      I01_01_1.bmp  (reference I01, distortion type 01, level 1)
      ...
    reference_images/
      I01.BMP
      ...
    mos_with_names.txt or mos.csv

Usage:
    python generate_tid2013_manifest.py \\
        --data-dir data/raw/tid2013 \\
        --output data/processed/tid2013/manifest.jsonl \\
        --split test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# TID2013 distortion types (24 types)
DISTORTION_TYPES = {
    1: "Additive Gaussian noise",
    2: "Additive noise in color components",
    3: "Spatially correlated noise",
    4: "Masked noise",
    5: "High frequency noise",
    6: "Impulse noise",
    7: "Quantization noise",
    8: "Gaussian blur",
    9: "Image denoising",
    10: "JPEG compression",
    11: "JPEG2000 compression",
    12: "JPEG transmission errors",
    13: "JPEG2000 transmission errors",
    14: "Non eccentricity pattern noise",
    15: "Local block-wise distortions",
    16: "Mean shift",
    17: "Contrast change",
    18: "Change of color saturation",
    19: "Multiplicative Gaussian noise",
    20: "Comfort noise",
    21: "Lossy compression of noisy images",
    22: "Image color quantization with dither",
    23: "Chromatic aberrations",
    24: "Sparse sampling and reconstruction"
}


def parse_mos_file(mos_file: Path) -> Dict[str, float]:
    """
    Parse MOS file and return mapping of image filename -> MOS score.

    Args:
        mos_file: Path to mos_with_names.txt or mos.csv

    Returns:
        Dictionary mapping filename (e.g., "I01_01_1.bmp") to MOS score
    """
    mos_dict = {}

    with open(mos_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try space-separated format: "I01_01_1.bmp 5.432"
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[0]
                try:
                    mos = float(parts[1])
                    mos_dict[filename] = mos
                except ValueError:
                    logger.warning(f"Could not parse MOS from line: {line}")
                    continue

    logger.info(f"Parsed {len(mos_dict)} MOS scores from {mos_file}")
    return mos_dict


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    Parse TID2013 filename to extract reference ID, distortion type, and level.

    Format: I<ref>_<dist>_<level>.bmp
    Example: I01_01_1.bmp -> ref=I01, distortion_type=1, level=1

    Args:
        filename: Distorted image filename

    Returns:
        Dictionary with parsed components or None if invalid format
    """
    # Pattern: I<ref_num>_<distortion_type>_<level>.bmp
    pattern = r'^I(\d{2})_(\d{2})_(\d)\.(?:bmp|BMP)$'
    match = re.match(pattern, filename)

    if not match:
        return None

    ref_num = match.group(1)
    distortion_type_id = int(match.group(2))
    level = int(match.group(3))

    return {
        'reference_id': f'I{ref_num}',
        'distortion_type_id': distortion_type_id,
        'distortion_type': DISTORTION_TYPES.get(distortion_type_id, f"Unknown_{distortion_type_id}"),
        'level': level
    }


def generate_manifest(
    data_dir: Path,
    output: Path,
    split: str = 'test',
    resume: bool = False
) -> int:
    """
    Generate JSONL manifest for TID2013 dataset.

    Args:
        data_dir: Root directory of TID2013 dataset
        output: Output JSONL manifest path
        split: Dataset split ('train', 'val', 'test')
        resume: Skip already processed samples

    Returns:
        Number of samples written
    """
    # Locate directories and MOS file
    distorted_dir = data_dir / 'distorted_images'
    reference_dir = data_dir / 'reference_images'

    # Try multiple MOS file names
    mos_file = None
    for candidate in ['mos_with_names.txt', 'mos.csv', 'mos.txt']:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            mos_file = candidate_path
            break

    if not distorted_dir.exists():
        logger.error(f"Distorted images directory not found: {distorted_dir}")
        return 0

    if not reference_dir.exists():
        logger.error(f"Reference images directory not found: {reference_dir}")
        return 0

    if not mos_file:
        logger.error(f"MOS file not found in {data_dir}")
        return 0

    # Parse MOS scores
    mos_dict = parse_mos_file(mos_file)

    # Get list of distorted images
    distorted_images = sorted(distorted_dir.glob('*.bmp')) + sorted(distorted_dir.glob('*.BMP'))
    logger.info(f"Found {len(distorted_images)} distorted images")

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

    # Open output file in append mode if resuming
    mode = 'a' if resume else 'w'
    sample_count = 0

    with open(output, mode, encoding='utf-8') as f:
        for idx, distorted_path in enumerate(distorted_images, 1):
            filename = distorted_path.name

            # Parse filename
            parsed = parse_filename(filename)
            if not parsed:
                logger.warning(f"Skipping invalid filename format: {filename}")
                continue

            # Check if already processed
            sample_id = f"tid2013_{idx:04d}"
            if sample_id in processed_ids:
                continue

            # Get MOS score
            mos = mos_dict.get(filename)
            if mos is None:
                logger.warning(f"No MOS score for {filename}, skipping")
                continue

            # Locate reference image
            reference_filename = f"{parsed['reference_id']}.BMP"
            reference_path = reference_dir / reference_filename

            # Try lowercase if uppercase doesn't exist
            if not reference_path.exists():
                reference_path = reference_dir / f"{parsed['reference_id']}.bmp"

            if not reference_path.exists():
                logger.warning(f"Reference image not found: {reference_filename}, skipping {filename}")
                continue

            # Create manifest entry
            entry = {
                'sample_id': sample_id,
                'dataset': 'tid2013',
                'distorted_path': str(distorted_path.relative_to(data_dir.parent)),
                'reference_path': str(reference_path.relative_to(data_dir.parent)),
                'mos': round(mos, 3),
                'split': split,
                'metadata': {
                    'distortion_type': parsed['distortion_type'],
                    'level': parsed['level'],
                    'reference_id': parsed['reference_id']
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
    parser = argparse.ArgumentParser(description="Generate TID2013 manifest")
    parser.add_argument('--data-dir', type=Path, required=True, help="TID2013 dataset root directory")
    parser.add_argument('--output', type=Path, required=True, help="Output JSONL manifest path")
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help="Dataset split")
    parser.add_argument('--resume', action='store_true', help="Resume from existing manifest")

    args = parser.parse_args()

    count = generate_manifest(
        data_dir=args.data_dir,
        output=args.output,
        split=args.split,
        resume=args.resume
    )

    if count > 0:
        logger.info(f"Success: {count} samples written to {args.output}")
    else:
        logger.error("Failed to generate manifest")
        exit(1)


if __name__ == '__main__':
    main()
