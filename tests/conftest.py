"""
Pytest configuration file.
Adds the project root to sys.path so tests can import from src.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
