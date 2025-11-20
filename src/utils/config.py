"""
Configuration loading and validation utilities for AgenticIQA.
Handles YAML parsing, environment variable interpolation, and schema validation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# Load .env file if it exists (do this at module import time)
try:
    from dotenv import load_dotenv
    # Look for .env file in current directory
    env_path = Path.cwd() / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        # Try to find .env in project root (assuming src/utils/config.py structure)
        root_env = Path(__file__).parent.parent.parent / '.env'
        if root_env.exists():
            load_dotenv(root_env, override=False)
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


class ModelBackendConfig(BaseModel):
    """Configuration for a VLM backend."""
    backend: str = Field(..., description="Model backend identifier (e.g., 'openai.gpt-4o')")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    base_url: Optional[str] = Field(None, description="Custom API endpoint URL")

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate backend format."""
        valid_prefixes = ['openai.', 'anthropic.', 'google.', 'qwen2.5-vl', 'local.']
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(f"Backend must start with one of {valid_prefixes}")
        return v


class ModelBackendsConfig(BaseModel):
    """Configuration for all module backends."""
    planner: ModelBackendConfig
    executor: ModelBackendConfig
    summarizer: ModelBackendConfig


class PipelineConfig(BaseModel):
    """Pipeline orchestration configuration."""
    max_replan: int = Field(default=2, ge=0, description="Maximum replanning iterations")
    cache_dir: str = Field(..., description="Directory for caching intermediate results")
    log_path: str = Field(..., description="Path to pipeline log file")
    enable_tracing: bool = Field(default=False, description="Enable LangGraph tracing")


class AgenticIQAConfig(BaseModel):
    """Complete AgenticIQA configuration."""
    pipeline: PipelineConfig


def interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate environment variables in configuration values.
    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.

    Args:
        value: Configuration value (can be str, dict, list, or primitive)

    Returns:
        Value with environment variables expanded

    Raises:
        ValueError: If referenced environment variable is not set and no default provided
    """
    if isinstance(value, str):
        # Find all ${VAR_NAME} or ${VAR_NAME:-default} patterns
        # Pattern captures: (var_name, default_value)
        # ${VAR} -> var_name="VAR", default_value=None (no :- present)
        # ${VAR:-default} -> var_name="VAR", default_value="default"
        # ${VAR:-} -> var_name="VAR", default_value="" (empty default)
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'

        # Use finditer to distinguish between no default and empty default
        for match in re.finditer(pattern, value):
            full_match = match.group(0)  # e.g., "${VAR:-default}" or "${VAR}"
            var_name = match.group(1)     # e.g., "VAR"
            # Check if group 2 was captured (i.e., :- was present)
            has_default = match.lastindex >= 2
            default_value = match.group(2) if has_default else None

            env_value = os.environ.get(var_name)
            if env_value is None:
                if has_default:
                    # Use default value if :- was present (even if empty)
                    value = value.replace(full_match, default_value)
                else:
                    # No default provided, raise error
                    raise ValueError(f"Environment variable ${{{var_name}}} is not set")
            else:
                # Use env value
                value = value.replace(full_match, env_value)

        return value

    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]

    else:
        return value


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML syntax is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")


def load_model_backends(config_path: Optional[Path] = None) -> ModelBackendsConfig:
    """
    Load and validate model backend configuration.

    Args:
        config_path: Path to model_backends.yaml (defaults to configs/model_backends.yaml)

    Returns:
        Validated ModelBackendsConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        # Default to configs/model_backends.yaml
        root = Path(os.environ.get('AGENTIC_ROOT', Path.cwd()))
        config_path = root / 'configs' / 'model_backends.yaml'

    # Load YAML
    config = load_yaml_config(config_path)

    # Interpolate environment variables
    config = interpolate_env_vars(config)

    # Validate with Pydantic
    try:
        return ModelBackendsConfig(**config)
    except Exception as e:
        raise ValueError(f"Invalid model backends configuration: {e}")


def load_pipeline_config(config_path: Optional[Path] = None) -> AgenticIQAConfig:
    """
    Load and validate pipeline configuration.

    Args:
        config_path: Path to pipeline.yaml (defaults to configs/pipeline.yaml)

    Returns:
        Validated AgenticIQAConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        # Default to configs/pipeline.yaml
        root = Path(os.environ.get('AGENTIC_ROOT', Path.cwd()))
        config_path = root / 'configs' / 'pipeline.yaml'

    # Load YAML
    config = load_yaml_config(config_path)

    # Interpolate environment variables
    config = interpolate_env_vars(config)

    # Apply defaults for missing values
    if 'pipeline' not in config:
        config['pipeline'] = {}

    pipeline = config['pipeline']
    if 'max_replan' not in pipeline:
        pipeline['max_replan'] = 2
    if 'cache_dir' not in pipeline:
        log_root = os.environ.get('AGENTIC_LOG_ROOT', './logs')
        pipeline['cache_dir'] = f"{log_root}/cache"
    if 'log_path' not in pipeline:
        log_root = os.environ.get('AGENTIC_LOG_ROOT', './logs')
        pipeline['log_path'] = f"{log_root}/pipeline.log"
    if 'enable_tracing' not in pipeline:
        pipeline['enable_tracing'] = False

    # Interpolate again after applying defaults
    config = interpolate_env_vars(config)

    # Validate with Pydantic
    try:
        return AgenticIQAConfig(**config)
    except Exception as e:
        raise ValueError(f"Invalid pipeline configuration: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Generic function to load any YAML configuration file.

    Args:
        config_path: Path to YAML configuration file (string or Path)

    Returns:
        Configuration dictionary with environment variables interpolated

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML syntax is invalid
    """
    path = Path(config_path)

    # Load YAML
    config = load_yaml_config(path)

    # Interpolate environment variables
    config = interpolate_env_vars(config)

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries with later overriding earlier.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration dictionary
    """
    merged = {}

    for config in configs:
        if config:
            _deep_merge(merged, config)

    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """
    Deep merge override dict into base dict in-place.

    Args:
        base: Base dictionary to merge into
        override: Override dictionary with new values
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# Example usage
if __name__ == '__main__':
    # Test configuration loading
    try:
        backends = load_model_backends()
        print("Model Backends Configuration:")
        print(f"  Planner: {backends.planner.backend} (temp={backends.planner.temperature})")
        print(f"  Executor: {backends.executor.backend} (temp={backends.executor.temperature})")
        print(f"  Summarizer: {backends.summarizer.backend} (temp={backends.summarizer.temperature})")
    except Exception as e:
        print(f"Error loading model backends: {e}")

    try:
        pipeline = load_pipeline_config()
        print("\nPipeline Configuration:")
        print(f"  Max replan: {pipeline.pipeline.max_replan}")
        print(f"  Cache dir: {pipeline.pipeline.cache_dir}")
        print(f"  Log path: {pipeline.pipeline.log_path}")
        print(f"  Tracing: {pipeline.pipeline.enable_tracing}")
    except Exception as e:
        print(f"Error loading pipeline config: {e}")
