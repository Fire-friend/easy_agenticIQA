#!/usr/bin/env python3
"""
Test environment variable loading for API server.
"""

import os
import sys
from pathlib import Path

# IMPORTANT: Set AGENTIC_ROOT before any project imports
project_root = Path(__file__).parent.parent
if not os.getenv('AGENTIC_ROOT'):
    os.environ['AGENTIC_ROOT'] = str(project_root)

# Load .env file if it exists (before any project imports)
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=False)
        print(f"✓ Loaded environment from {env_path}")
    else:
        print(f"✗ .env file not found at {env_path}")
except ImportError:
    print("✗ python-dotenv not installed")

# Set other required environment variables if not set
if not os.getenv('AGENTIC_LOG_ROOT'):
    os.environ['AGENTIC_LOG_ROOT'] = str(project_root / 'logs')

if not os.getenv('AGENTIC_DATA_ROOT'):
    os.environ['AGENTIC_DATA_ROOT'] = str(project_root / 'data')

if not os.getenv('AGENTIC_TOOL_HOME'):
    os.environ['AGENTIC_TOOL_HOME'] = str(project_root / 'iqa_tools')

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Now import project modules
from src.utils.config import load_model_backends

print("\n" + "="*60)
print("Environment Variable Check")
print("="*60)

# Check project paths
print("\nProject Paths:")
print(f"  AGENTIC_ROOT: {os.getenv('AGENTIC_ROOT')}")
print(f"  AGENTIC_LOG_ROOT: {os.getenv('AGENTIC_LOG_ROOT')}")
print(f"  AGENTIC_DATA_ROOT: {os.getenv('AGENTIC_DATA_ROOT')}")
print(f"  AGENTIC_TOOL_HOME: {os.getenv('AGENTIC_TOOL_HOME')}")

# Check API keys
print("\nAPI Keys:")
if os.getenv('OPENAI_API_KEY'):
    key = os.getenv('OPENAI_API_KEY')
    print(f"  ✓ OPENAI_API_KEY: {key[:10]}...{key[-4:]}")
else:
    print("  ✗ OPENAI_API_KEY: Not set")

if os.getenv('OPENAI_BASE_URL'):
    print(f"  ✓ OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
else:
    print("  ✗ OPENAI_BASE_URL: Not set")

if os.getenv('ANTHROPIC_API_KEY'):
    key = os.getenv('ANTHROPIC_API_KEY')
    print(f"  ✓ ANTHROPIC_API_KEY: {key[:10]}...{key[-4:]}")
else:
    print("  - ANTHROPIC_API_KEY: Not set (optional)")

if os.getenv('GOOGLE_API_KEY'):
    key = os.getenv('GOOGLE_API_KEY')
    print(f"  ✓ GOOGLE_API_KEY: {key[:10]}...{key[-4:]}")
else:
    print("  - GOOGLE_API_KEY: Not set (optional)")

# Check model backends configuration
print("\nModel Backends Configuration:")
try:
    backends = load_model_backends()
    print(f"  ✓ Planner: {backends.planner.backend} (temp={backends.planner.temperature})")
    print(f"  ✓ Executor: {backends.executor.backend} (temp={backends.executor.temperature})")
    print(f"  ✓ Summarizer: {backends.summarizer.backend} (temp={backends.summarizer.temperature})")
except Exception as e:
    print(f"  ✗ Failed to load model backends: {e}")

print("\n" + "="*60)
print("Environment check complete!")
print("="*60)
