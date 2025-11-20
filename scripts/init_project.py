#!/usr/bin/env python3
"""
AgenticIQA Project Initialization Script
Creates all required directories for the project.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory and parent directories
    env_path = Path.cwd() / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
    else:
        # Try to find .env in script's parent directory (project root)
        root_env = Path(__file__).parent.parent / '.env'
        if root_env.exists():
            load_dotenv(root_env)
            print(f"Loaded environment from: {root_env}")
except ImportError:
    print("Warning: 'python-dotenv' not installed. .env file will not be loaded.")
    print("Install it with: pip install python-dotenv")


def get_project_root() -> Path:
    """Get project root directory from environment or current directory."""
    root = os.environ.get('AGENTIC_ROOT')
    if root:
        return Path(root)
    else:
        # Assume script is in scripts/ subdirectory
        return Path(__file__).parent.parent


def get_directory_config() -> List[Tuple[str, str]]:
    """
    Get list of directories to create with descriptions.

    Returns:
        List of (path, description) tuples
    """
    root = get_project_root()
    data_root = Path(os.environ.get('AGENTIC_DATA_ROOT', root / 'data'))
    tool_home = Path(os.environ.get('AGENTIC_TOOL_HOME', root / 'iqa_tools'))
    log_root = Path(os.environ.get('AGENTIC_LOG_ROOT', root / 'logs'))

    directories = [
        # Scripts
        (root / 'scripts', 'Utility and validation scripts'),

        # Source code
        (root / 'src', 'Source code root'),
        (root / 'src' / 'agentic', 'Core LangGraph pipeline'),
        (root / 'src' / 'agentic' / 'nodes', 'Agent node implementations'),
        (root / 'src' / 'utils', 'Shared utility functions'),

        # IQA tools
        (tool_home / 'weights', 'IQA model checkpoints'),
        (tool_home / 'metadata', 'Tool metadata JSON files'),

        # Data directories
        (data_root / 'raw', 'Original datasets'),
        (data_root / 'raw' / 'agenticiqa_eval', 'AgenticIQA-Eval MCQ dataset'),
        (data_root / 'raw' / 'tid2013', 'TID2013 dataset'),
        (data_root / 'raw' / 'bid', 'BID dataset'),
        (data_root / 'raw' / 'agiqa-3k', 'AGIQA-3K dataset'),
        (data_root / 'processed', 'Processed manifests'),
        (data_root / 'cache', 'Intermediate results cache'),

        # Logs
        (log_root, 'Execution logs and traces'),

        # Configs
        (root / 'configs', 'Configuration files'),
    ]

    return directories


def create_directory(path: Path, description: str) -> Tuple[bool, str]:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Directory path to create
        description: Human-readable description

    Returns:
        (success, status_message) tuple
    """
    try:
        if path.exists():
            if path.is_dir():
                return True, f"Already exists: {path}"
            else:
                return False, f"Path exists but is not a directory: {path}"

        # Create directory with parents
        path.mkdir(parents=True, exist_ok=True, mode=0o755)

        # Verify it's writable
        test_file = path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            return False, f"Directory created but not writable: {path} ({e})"

        return True, f"Created: {path}"

    except PermissionError:
        return False, f"Permission denied: {path}"
    except Exception as e:
        return False, f"Error creating {path}: {e}"


def create_init_files() -> None:
    """Create __init__.py files for Python packages."""
    root = get_project_root()

    init_files = [
        root / 'src' / '__init__.py',
        root / 'src' / 'agentic' / '__init__.py',
        root / 'src' / 'agentic' / 'nodes' / '__init__.py',
        root / 'src' / 'utils' / '__init__.py',
    ]

    for init_file in init_files:
        if not init_file.exists():
            try:
                init_file.touch()
                print(f"  Created: {init_file}")
            except Exception as e:
                print(f"  Warning: Could not create {init_file}: {e}")


def create_gitkeep_files() -> None:
    """Create .gitkeep files in empty directories that should be tracked."""
    root = get_project_root()
    data_root = Path(os.environ.get('AGENTIC_DATA_ROOT', root / 'data'))
    tool_home = Path(os.environ.get('AGENTIC_TOOL_HOME', root / 'iqa_tools'))

    gitkeep_dirs = [
        data_root / 'raw',
        data_root / 'processed',
        tool_home / 'weights',
        tool_home / 'metadata',
    ]

    for directory in gitkeep_dirs:
        if directory.exists():
            gitkeep = directory / '.gitkeep'
            if not gitkeep.exists():
                try:
                    gitkeep.touch()
                    print(f"  Created: {gitkeep}")
                except Exception as e:
                    print(f"  Warning: Could not create {gitkeep}: {e}")


def main():
    """Main initialization entry point."""
    print("="*70)
    print("AgenticIQA Project Initialization")
    print("="*70)

    root = get_project_root()
    print(f"\nProject root: {root}")

    # Display environment variables
    print("\nEnvironment variables:")
    env_vars = ['AGENTIC_ROOT', 'AGENTIC_DATA_ROOT', 'AGENTIC_TOOL_HOME', 'AGENTIC_LOG_ROOT']
    for var in env_vars:
        value = os.environ.get(var, '(not set)')
        print(f"  {var}: {value}")

    # Get directories to create
    directories = get_directory_config()

    print(f"\nCreating {len(directories)} directories...")
    print()

    created = 0
    existed = 0
    failed = 0
    errors = []

    for path, description in directories:
        success, message = create_directory(path, description)

        if success:
            if "Already exists" in message:
                print(f"  ✓ {message}")
                existed += 1
            else:
                print(f"  ✓ {message}")
                created += 1
        else:
            print(f"  ✗ {message}")
            errors.append(message)
            failed += 1

    # Create __init__.py files
    print("\nCreating Python package __init__.py files...")
    create_init_files()

    # Create .gitkeep files
    print("\nCreating .gitkeep files for empty tracked directories...")
    create_gitkeep_files()

    # Summary
    print("\n" + "="*70)
    print("Initialization Summary")
    print("="*70)
    print(f"  Created: {created} directories")
    print(f"  Already existed: {existed} directories")
    print(f"  Failed: {failed} directories")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  • {error}")
        print("\nRemediation:")
        print("  1. Check directory permissions")
        print("  2. Verify environment variables are set correctly")
        print("  3. Run with appropriate privileges if needed")
        sys.exit(1)
    else:
        print("\n✓ Project initialization complete!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Validate environment: python scripts/check_env.py")
        print("  3. Configure model backends in configs/model_backends.yaml")
        sys.exit(0)


if __name__ == '__main__':
    main()
