#!/usr/bin/env python3
"""
Generate JSONL manifest for AGIQA-3K AI-Generated Image Quality Assessment dataset.

Usage:
    python generate_agiqa3k_manifest.py \\
        --data-dir data/raw/agiqa-3k \\
        --output data/processed/agiqa3k/manifest.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_metadata_file(metadata_file: Path) -> Dict[str, Dict]:
    """
    Parse AGIQA-3K metadata JSON file.

    Expected format:
    {
      "img001.png": {
        "mos": 4.2,
        "generator": "Stable Diffusion",
        "prompt": "A beautiful sunset...",
        ...
      }
    }

    Args:
        metadata_file: Path to metadata JSON

    Returns:
        Dictionary mapping filename -> metadata dict
    """
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata for {len(metadata)} images")
    return metadata


def generate_manifest(
    data_dir: Path,
    output: Path,
    split: str = 'test',
    resume: bool = False
) -> int:
    """
    Generate JSONL manifest for AGIQA-3K dataset.

    Args:
        data_dir: Root directory of AGIQA-3K dataset
        output: Output JSONL manifest path
        split: Dataset split
        resume: Skip already processed samples

    Returns:
        Number of samples written
    """
    # Locate images and metadata
    image_dir = data_dir / 'images'
    metadata_file = data_dir / 'metadata.json'

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return 0

    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return 0

    # Parse metadata
    metadata_dict = parse_metadata_file(metadata_file)

    # Get image files
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
    logger.info(f"Found {len(image_files)} images")

    # Load existing samples if resuming
    processed_ids = set()
    if resume and output.exists():
        with open(output, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                processed_ids.add(entry['sample_id'])
        logger.info(f"Resuming: Skipping {len(processed_ids)} samples")

    # Create output directory
    output.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if resume else 'w'
    sample_count = 0

    with open(output, mode, encoding='utf-8') as f:
        for idx, image_path in enumerate(image_files, 1):
            filename = image_path.name

            sample_id = f"agiqa3k_{idx:04d}"
            if sample_id in processed_ids:
                continue

            # Get metadata
            meta = metadata_dict.get(filename, {})
            mos = meta.get('mos')

            if mos is None:
                logger.warning(f"No MOS for {filename}, skipping")
                continue

            # Create manifest entry
            entry = {
                'sample_id': sample_id,
                'dataset': 'agiqa-3k',
                'image_path': str(image_path.relative_to(data_dir.parent)),
                'mos': round(float(mos), 3),
                'split': split,
                'metadata': {
                    'generator_model': meta.get('generator', 'unknown'),
                    'prompt': meta.get('prompt', ''),
                    'seed': meta.get('seed'),
                    'sampling_steps': meta.get('steps'),
                    'guidance_scale': meta.get('guidance_scale')
                }
            }

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            sample_count += 1

            if sample_count % 500 == 0:
                logger.info(f"Processed {sample_count} samples...")

    logger.info(f"✓ Generated manifest with {sample_count} samples: {output}")
    return sample_count


def main():
    parser = argparse.ArgumentParser(description="Generate AGIQA-3K manifest")
    parser.add_argument('--data-dir', type=Path, required=True, help="AGIQA-3K dataset root")
    parser.add_argument('--output', type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--resume', action='store_true', help="Resume from existing manifest")

    args = parser.parse_args()

    count = generate_manifest(args.data_dir, args.output, args.split, args.resume)

    if count > 0:
        logger.info(f"Success: {count} samples written")
    else:
        logger.error("Failed to generate manifest")
        exit(1)


if __name__ == '__main__':
    main()
