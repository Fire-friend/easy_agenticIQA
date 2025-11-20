#!/usr/bin/env python3
"""
Validate JSONL manifest against JSON Schema.

Checks:
  - Schema compliance for each line
  - File path existence (optional)
  - MOS value ranges
  - Duplicate sample_ids

Usage:
    python validate_manifest.py \\
        --manifest data/processed/tid2013/manifest.jsonl \\
        --schema data/schemas/tid2013_schema.json \\
        --check-paths \\
        --strict
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("ERROR: jsonschema package not installed. Run: pip install jsonschema")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_schema(schema_path: Path) -> Dict:
    """Load JSON Schema from file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema


def validate_manifest(
    manifest_path: Path,
    schema: Dict,
    check_paths: bool = False,
    strict: bool = False,
    base_dir: Path = None
) -> Tuple[int, List[Dict]]:
    """
    Validate JSONL manifest against schema.

    Args:
        manifest_path: Path to JSONL manifest
        schema: JSON Schema dictionary
        check_paths: Verify file paths exist
        strict: Stop at first error
        base_dir: Base directory for resolving relative paths

    Returns:
        (valid_count, error_list)
    """
    if base_dir is None:
        base_dir = manifest_path.parent.parent

    errors = []
    valid_count = 0
    seen_ids = set()

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                error = {
                    'line': line_num,
                    'type': 'json_parse_error',
                    'message': str(e)
                }
                errors.append(error)
                logger.error(f"Line {line_num}: JSON parse error - {e}")

                if strict:
                    break
                continue

            # Check for duplicate sample_id
            sample_id = entry.get('sample_id')
            if sample_id:
                if sample_id in seen_ids:
                    error = {
                        'line': line_num,
                        'type': 'duplicate_id',
                        'sample_id': sample_id
                    }
                    errors.append(error)
                    logger.error(f"Line {line_num}: Duplicate sample_id '{sample_id}'")

                    if strict:
                        break
                    continue

                seen_ids.add(sample_id)

            # Validate against schema
            try:
                validate(instance=entry, schema=schema)
            except ValidationError as e:
                error = {
                    'line': line_num,
                    'type': 'schema_validation',
                    'sample_id': sample_id,
                    'message': e.message,
                    'path': '.'.join(str(p) for p in e.path) if e.path else 'root'
                }
                errors.append(error)
                logger.error(f"Line {line_num}: Schema validation failed - {e.message}")

                if strict:
                    break
                continue

            # Check file paths exist
            if check_paths:
                path_fields = []

                if 'distorted_path' in entry:
                    path_fields.append(('distorted_path', entry['distorted_path']))
                if 'reference_path' in entry:
                    path_fields.append(('reference_path', entry['reference_path']))
                if 'image_path' in entry:
                    path_fields.append(('image_path', entry['image_path']))

                for field_name, path_str in path_fields:
                    path = Path(path_str)

                    # Try absolute path first
                    if not path.is_absolute():
                        path = base_dir / path

                    if not path.exists():
                        error = {
                            'line': line_num,
                            'type': 'missing_file',
                            'sample_id': sample_id,
                            'field': field_name,
                            'path': str(path)
                        }
                        errors.append(error)
                        logger.warning(f"Line {line_num}: File not found - {field_name}: {path}")

                        if strict:
                            break

            valid_count += 1

    return valid_count, errors


def main():
    parser = argparse.ArgumentParser(description="Validate manifest against JSON Schema")
    parser.add_argument('--manifest', type=Path, required=True, help="JSONL manifest file")
    parser.add_argument('--schema', type=Path, required=True, help="JSON Schema file")
    parser.add_argument('--check-paths', action='store_true', help="Verify file paths exist")
    parser.add_argument('--strict', action='store_true', help="Stop at first error")
    parser.add_argument('--base-dir', type=Path, help="Base directory for resolving relative paths")

    args = parser.parse_args()

    if not args.manifest.exists():
        logger.error(f"Manifest file not found: {args.manifest}")
        sys.exit(1)

    if not args.schema.exists():
        logger.error(f"Schema file not found: {args.schema}")
        sys.exit(1)

    # Load schema
    logger.info(f"Loading schema: {args.schema}")
    schema = load_schema(args.schema)

    # Validate manifest
    logger.info(f"Validating manifest: {args.manifest}")
    valid_count, errors = validate_manifest(
        manifest_path=args.manifest,
        schema=schema,
        check_paths=args.check_paths,
        strict=args.strict,
        base_dir=args.base_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"Valid entries: {valid_count}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nError breakdown:")
        error_types = {}
        for error in errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in error_types.items():
            print(f"  - {error_type}: {count}")

    print("=" * 60)

    if errors:
        logger.error(f"Validation failed with {len(errors)} error(s)")
        sys.exit(1)
    else:
        logger.info("✓ Validation passed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
