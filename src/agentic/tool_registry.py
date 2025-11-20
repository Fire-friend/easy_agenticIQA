"""
Tool Registry for IQA-PyTorch integration.
Manages IQA tool metadata, execution, and score normalization.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""
    pass


class ToolRegistry:
    """
    Registry for managing IQA tools and their execution.

    Handles:
    - Tool metadata loading from JSON
    - Tool execution via IQA-PyTorch
    - Score normalization using logistic function
    - Caching of tool outputs
    """

    def __init__(self, metadata_path: Optional[Path] = None, cache_size: int = 1000):
        """
        Initialize Tool Registry.

        Args:
            metadata_path: Path to tools.json metadata file
            cache_size: Maximum number of cached results (LRU)
        """
        if metadata_path is None:
            # Default to iqa_tools/metadata/tools.json
            import os
            root = Path(os.environ.get('AGENTIC_ROOT', Path.cwd()))
            metadata_path = root / 'iqa_tools' / 'metadata' / 'tools.json'

        self.metadata_path = metadata_path
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.cache: OrderedDict[str, Tuple[float, float]] = OrderedDict()
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load tool metadata from JSON file."""
        if not self.metadata_path.exists():
            logger.warning(f"Tool metadata file not found: {self.metadata_path}")
            logger.warning("Using empty tool registry. Add tools.json to enable tool execution.")
            return

        try:
            with open(self.metadata_path, 'r') as f:
                self.tools = json.load(f)

            logger.info(f"Loaded {len(self.tools)} tools from {self.metadata_path}")

            # Validate tool metadata
            for tool_name, metadata in self.tools.items():
                if 'type' not in metadata:
                    raise ValueError(f"Tool {tool_name} missing 'type' field")
                if metadata['type'] not in ['FR', 'NR']:
                    raise ValueError(f"Tool {tool_name} has invalid type: {metadata['type']}")
                if 'strengths' not in metadata:
                    logger.warning(f"Tool {tool_name} missing 'strengths' field")
                    metadata['strengths'] = []
                if 'logistic_params' not in metadata:
                    logger.warning(f"Tool {tool_name} missing logistic_params, will use defaults")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool metadata file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load tool metadata: {e}")

    def get_tools_for_distortion(self, distortion: str, reference_available: bool = False) -> List[str]:
        """
        Get list of tools that can handle a specific distortion.

        Args:
            distortion: Distortion type (e.g., "Blurs", "Noise")
            reference_available: Whether reference image is available (prioritize FR tools)

        Returns:
            List of tool names sorted by suitability (FR first if reference available)
        """
        suitable_tools = []

        for tool_name, metadata in self.tools.items():
            # Check if tool handles this distortion
            if distortion in metadata.get('strengths', []):
                suitable_tools.append(tool_name)

        # Sort by type if reference available (FR tools first)
        if reference_available:
            suitable_tools.sort(key=lambda t: (self.tools[t]['type'] != 'FR', t))
        else:
            suitable_tools.sort(key=lambda t: (self.tools[t]['type'] == 'FR', t))

        return suitable_tools

    def get_tools_by_type(self, tool_type: str) -> List[str]:
        """
        Get all tools of a specific type.

        Args:
            tool_type: "FR" or "NR"

        Returns:
            List of tool names matching the type
        """
        return [
            tool_name for tool_name, metadata in self.tools.items()
            if metadata['type'] == tool_type
        ]

    def is_tool_available(self, tool_name: str) -> bool:
        """
        Check if a tool is available in the registry.

        Args:
            tool_name: Tool identifier

        Returns:
            True if tool exists in metadata
        """
        return tool_name in self.tools

    def _compute_image_hash(self, image_path: str) -> str:
        """Compute SHA256 hash of image file."""
        with open(image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _get_cache_key(self, tool_name: str, image_path: str, reference_path: Optional[str] = None) -> str:
        """Generate cache key for tool execution."""
        image_hash = self._compute_image_hash(image_path)
        ref_hash = self._compute_image_hash(reference_path) if reference_path else ""
        return f"{tool_name}:{image_hash}:{ref_hash}"

    def _check_cache(self, cache_key: str) -> Optional[Tuple[float, float]]:
        """Check cache for tool output. Returns (raw_score, normalized_score) if found."""
        if cache_key in self.cache:
            self.cache_hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        self.cache_misses += 1
        return None

    def _store_cache(self, cache_key: str, raw_score: float, normalized_score: float) -> None:
        """Store tool output in cache with LRU eviction."""
        self.cache[cache_key] = (raw_score, normalized_score)
        self.cache.move_to_end(cache_key)

        # Evict oldest if cache too large
        if len(self.cache) > self.cache_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            logger.debug(f"Cache evicted: {oldest_key}")

    def normalize_score(self, tool_name: str, raw_score: float) -> float:
        """
        Normalize tool output to [1, 5] scale using logistic function.

        Formula: f(x) = (β1 - β2) / (1 + exp(-(x - β3)/|β4|)) + β2

        Args:
            tool_name: Tool identifier
            raw_score: Raw tool output

        Returns:
            Normalized score in [1, 5] range
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        metadata = self.tools[tool_name]
        params = metadata.get('logistic_params')

        # Use defaults if parameters not provided
        if params is None:
            # Default: higher raw score = better quality
            params = {'beta1': 5.0, 'beta2': 1.0, 'beta3': 0.5, 'beta4': 0.1}
            logger.warning(f"Using default logistic params for {tool_name}")

        beta1 = params.get('beta1', 5.0)
        beta2 = params.get('beta2', 1.0)
        beta3 = params.get('beta3', 0.5)
        beta4 = params.get('beta4', 0.1)

        # Apply logistic function
        try:
            normalized = (beta1 - beta2) / (1 + np.exp(-(raw_score - beta3) / abs(beta4))) + beta2
        except (OverflowError, FloatingPointError):
            # Handle extreme values
            if raw_score > beta3:
                normalized = beta1
            else:
                normalized = beta2

        # Clip to [1, 5] range
        normalized = float(np.clip(normalized, 1.0, 5.0))

        if not (1.0 <= normalized <= 5.0):
            logger.warning(f"Normalized score {normalized} outside [1, 5], clipping")
            normalized = max(1.0, min(5.0, normalized))

        return normalized

    def execute_tool(
        self,
        tool_name: str,
        image_path: str,
        reference_path: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Execute an IQA tool and return raw and normalized scores.

        Args:
            tool_name: IQA tool identifier
            image_path: Path to test image
            reference_path: Optional path to reference image (for FR tools)

        Returns:
            Tuple of (raw_score, normalized_score)

        Raises:
            ToolExecutionError: If tool execution fails
        """
        if tool_name not in self.tools:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")

        # Check cache
        cache_key = self._get_cache_key(tool_name, image_path, reference_path)
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {tool_name} on {Path(image_path).name}")
            return cached_result

        # Execute tool
        metadata = self.tools[tool_name]
        tool_type = metadata['type']

        try:
            # Import IQA-PyTorch
            import pyiqa

            # Create metric instance
            device = 'cuda' if self._cuda_available() else 'cpu'
            metric = pyiqa.create_metric(tool_name.lower(), device=device)

            # Load images
            if tool_type == 'FR':
                if reference_path is None:
                    raise ToolExecutionError(f"Tool {tool_name} requires reference image")

                # Full-Reference scoring
                raw_score = float(metric(image_path, reference_path).item())
            else:
                # No-Reference scoring
                raw_score = float(metric(image_path).item())

            # Handle NaN/Inf
            if not np.isfinite(raw_score):
                raise ToolExecutionError(f"Tool {tool_name} returned non-finite score: {raw_score}")

            # Normalize score
            normalized_score = self.normalize_score(tool_name, raw_score)

            # Store in cache
            self._store_cache(cache_key, raw_score, normalized_score)

            logger.debug(f"Tool {tool_name}: raw={raw_score:.4f}, normalized={normalized_score:.2f}")
            return raw_score, normalized_score

        except ImportError as e:
            raise ToolExecutionError(f"IQA-PyTorch not installed: {e}")
        except Exception as e:
            raise ToolExecutionError(f"Tool {tool_name} execution failed: {e}")

    @staticmethod
    def _cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate
        }
