#!/usr/bin/env python3
"""
AgenticIQA Environment Validation Script
Validates Python environment, dependencies, GPU availability, and configuration.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' package not installed. Output will be plain text.")


class EnvironmentValidator:
    """Validates AgenticIQA environment setup."""

    def __init__(self, check_api: bool = False):
        self.check_api = check_api
        self.errors = []
        self.warnings = []
        self.info = []

        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

    def print(self, message: str, style: str = ""):
        """Print message with optional styling."""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)

    def check_python_version(self) -> bool:
        """Check Python version is 3.10.x."""
        self.print("\n[bold]1. Python Environment[/bold]", style="cyan")

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        self.print(f"  Python version: {version_str}")

        if version.major == 3 and version.minor == 10:
            self.print("  ✓ Python 3.10.x detected", style="green")
            return True
        else:
            self.errors.append(f"Python 3.10.x required, found {version_str}")
            self.print(f"  ✗ Python 3.10.x required, found {version_str}", style="red")
            return False

    def check_virtual_env(self) -> bool:
        """Check if running in virtual environment."""
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        conda_env = os.environ.get('CONDA_DEFAULT_ENV')

        if conda_env:
            self.print(f"  ✓ Running in conda environment: {conda_env}", style="green")
            return True
        elif in_venv:
            self.print("  ✓ Running in virtual environment", style="green")
            return True
        else:
            self.warnings.append("Not running in a virtual environment")
            self.print("  ⚠ Not in virtual environment (recommended)", style="yellow")
            return True

    def check_gpu_cuda(self) -> bool:
        """Check GPU and CUDA availability."""
        self.print("\n[bold]2. GPU and CUDA[/bold]", style="cyan")

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            self.print(f"  CUDA available: {cuda_available}")

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()

                self.print(f"  ✓ GPU detected: {gpu_name}", style="green")
                self.print(f"  CUDA version: {cuda_version}")
                self.print(f"  GPU count: {gpu_count}")

                # Check GPU memory
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    memory_gb = total_memory / (1024**3)
                    self.print(f"  GPU memory: {memory_gb:.1f} GB")

                    if memory_gb < 24:
                        self.warnings.append(f"GPU memory ({memory_gb:.1f}GB) < 24GB recommended for local VLM")
                        self.print(f"  ⚠ GPU memory < 24GB (local VLM may be slow)", style="yellow")
                except Exception as e:
                    self.warnings.append(f"Could not check GPU memory: {e}")
            else:
                self.warnings.append("No GPU detected - API-only mode")
                self.print("  ⚠ No GPU detected (API-only mode)", style="yellow")

            return True
        except ImportError:
            self.errors.append("PyTorch not installed")
            self.print("  ✗ PyTorch not installed", style="red")
            return False
        except Exception as e:
            self.warnings.append(f"GPU check failed: {e}")
            self.print(f"  ⚠ GPU check failed: {e}", style="yellow")
            return True

    def check_packages(self) -> bool:
        """Check required package installations and versions."""
        self.print("\n[bold]3. Package Dependencies[/bold]", style="cyan")

        # Core packages with version requirements
        core_packages = {
            'torch': '2.3.0',
            'transformers': '4.42.0',
            'pydantic': '2.7.0',
        }

        # Required packages (any version)
        required_packages = [
            'langgraph',
            'langchain',
            'pyiqa',
            'PIL',  # pillow
            'cv2',  # opencv-python
            'numpy',
            'scipy',
        ]

        # Optional API clients
        optional_packages = [
            'openai',
            'anthropic',
            'google.genai',  # Google's new recommended library
            'qwen_vl_utils',
        ]

        all_ok = True

        # Check core packages with versions
        for package, min_version in core_packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                self.print(f"  ✓ {package}: {version}", style="green")

                if version != 'unknown' and version < min_version:
                    self.warnings.append(f"{package} version {version} < {min_version} recommended")
                    self.print(f"    ⚠ Version {min_version}+ recommended", style="yellow")
            except ImportError:
                self.errors.append(f"Required package missing: {package}")
                self.print(f"  ✗ {package}: NOT FOUND", style="red")
                all_ok = False

        # Check required packages
        for package in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'installed')
                self.print(f"  ✓ {package}: {version}", style="green")
            except ImportError:
                self.errors.append(f"Required package missing: {package}")
                self.print(f"  ✗ {package}: NOT FOUND", style="red")
                all_ok = False

        # Check optional packages
        self.print("\n  [bold]Optional Packages:[/bold]")
        for package in optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'installed')
                self.print(f"  ✓ {package}: {version}", style="dim green")
            except ImportError:
                self.print(f"  - {package}: not installed", style="dim")

        return all_ok

    def check_environment_variables(self) -> bool:
        """Check required environment variables."""
        self.print("\n[bold]4. Environment Variables[/bold]", style="cyan")

        required_vars = [
            'AGENTIC_ROOT',
            'AGENTIC_DATA_ROOT',
            'AGENTIC_TOOL_HOME',
            'AGENTIC_LOG_ROOT',
        ]

        optional_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'GOOGLE_API_KEY',
            'OPENAI_BASE_URL',
            'ANTHROPIC_BASE_URL',
            'GOOGLE_API_BASE_URL',
        ]

        all_ok = True

        # Check required variables
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                path = Path(value)
                if path.exists():
                    self.print(f"  ✓ {var}: {value}", style="green")
                else:
                    self.warnings.append(f"{var} path does not exist: {value}")
                    self.print(f"  ⚠ {var}: {value} (path not found)", style="yellow")
            else:
                self.errors.append(f"Required environment variable not set: {var}")
                self.print(f"  ✗ {var}: NOT SET", style="red")
                all_ok = False

        # Check optional variables
        self.print("\n  [bold]Optional Variables:[/bold]")
        for var in optional_vars:
            value = os.environ.get(var)
            if value:
                # Don't display API keys
                if 'KEY' in var:
                    self.print(f"  ✓ {var}: ***configured***", style="dim green")
                else:
                    self.print(f"  ✓ {var}: {value}", style="dim green")
            else:
                self.print(f"  - {var}: not set", style="dim")

        return all_ok

    def check_api_connectivity(self) -> bool:
        """Check API connectivity (optional)."""
        if not self.check_api:
            return True

        self.print("\n[bold]5. API Connectivity[/bold]", style="cyan")
        self.print("  (Skipped - use --check-api to test)")

        # TODO: Implement actual API connectivity tests
        # This would make actual API calls to verify keys work

        return True

    def generate_report(self) -> int:
        """Generate final validation report and return exit code."""
        self.print("\n" + "="*60)

        if self.errors:
            self.print("\n[bold red]✗ VALIDATION FAILED[/bold red]")
            self.print(f"\n{len(self.errors)} error(s) found:")
            for error in self.errors:
                self.print(f"  • {error}", style="red")

            if self.warnings:
                self.print(f"\n{len(self.warnings)} warning(s):")
                for warning in self.warnings:
                    self.print(f"  • {warning}", style="yellow")

            self.print("\n[bold]Remediation Steps:[/bold]")
            self.print("1. Install missing packages: pip install -r requirements.txt")
            self.print("2. Set required environment variables in your shell profile")
            self.print("3. Run this script again to verify fixes")

            return 1

        elif self.warnings:
            self.print("\n[bold yellow]✓ VALIDATION PASSED WITH WARNINGS[/bold yellow]")
            self.print(f"\n{len(self.warnings)} warning(s):")
            for warning in self.warnings:
                self.print(f"  • {warning}", style="yellow")

            self.print("\n[dim]Environment is functional but some optimizations recommended.[/dim]")
            return 0

        else:
            self.print("\n[bold green]✓ VALIDATION SUCCESSFUL[/bold green]")
            self.print("\n[dim]All checks passed. Environment is ready for AgenticIQA.[/dim]")
            return 0


def main():
    """Main validation entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate AgenticIQA environment setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--check-api',
        action='store_true',
        help='Test API connectivity (makes actual API calls)',
    )

    args = parser.parse_args()

    validator = EnvironmentValidator(check_api=args.check_api)

    if RICH_AVAILABLE:
        validator.console.print(
            Panel.fit(
                "[bold cyan]AgenticIQA Environment Validator[/bold cyan]\n"
                "[dim]Checking Python, GPU, packages, and configuration...[/dim]",
                border_style="cyan",
            )
        )
    else:
        print("="*60)
        print("AgenticIQA Environment Validator")
        print("="*60)

    # Run all checks
    validator.check_python_version()
    validator.check_virtual_env()
    validator.check_gpu_cuda()
    validator.check_packages()
    validator.check_environment_variables()
    validator.check_api_connectivity()

    # Generate report and exit
    exit_code = validator.generate_report()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
