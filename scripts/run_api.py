#!/usr/bin/env python3
"""
Startup script for AgenticIQA FastAPI server.

Usage:
    python scripts/run_api.py                    # Default: 0.0.0.0:8000
    python scripts/run_api.py --port 9000        # Custom port
    python scripts/run_api.py --reload           # Development mode with hot reload
    python scripts/run_api.py --help             # Show help
"""

import os
import sys
from pathlib import Path
import argparse
import logging

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
        print(f"Loaded environment from {env_path}")
except ImportError:
    pass  # python-dotenv not installed

# Set other required environment variables if not set
if not os.getenv('AGENTIC_LOG_ROOT'):
    os.environ['AGENTIC_LOG_ROOT'] = str(project_root / 'logs')

if not os.getenv('AGENTIC_DATA_ROOT'):
    os.environ['AGENTIC_DATA_ROOT'] = str(project_root / 'data')

if not os.getenv('AGENTIC_TOOL_HOME'):
    os.environ['AGENTIC_TOOL_HOME'] = str(project_root / 'iqa_tools')

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AgenticIQA FastAPI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind (default: from config or 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: from config or 8000)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level (default: info)"
    )

    return parser.parse_args()


def load_api_config():
    """Load API configuration from config file."""
    try:
        from src.utils.config import load_config
        config = load_config("configs/api.yaml")
        return config.get("api", {})
    except Exception as e:
        logger.warning(f"Failed to load API config: {e}")
        return {}


def main():
    """Main entry point."""
    args = parse_args()

    logger.info(f"AGENTIC_ROOT={os.getenv('AGENTIC_ROOT')}")
    logger.info(f"AGENTIC_LOG_ROOT={os.getenv('AGENTIC_LOG_ROOT')}")

    # Check for required API keys
    api_keys_found = []
    if os.getenv('OPENAI_API_KEY'):
        api_keys_found.append('OPENAI_API_KEY')
    if os.getenv('ANTHROPIC_API_KEY'):
        api_keys_found.append('ANTHROPIC_API_KEY')
    if os.getenv('GOOGLE_API_KEY'):
        api_keys_found.append('GOOGLE_API_KEY')

    if api_keys_found:
        logger.info(f"Found API keys: {', '.join(api_keys_found)}")
    else:
        logger.warning("No API keys found in environment. Make sure to set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

    # Load configuration
    api_config = load_api_config()

    # Determine host and port (CLI args override config)
    host = args.host or api_config.get("host", "0.0.0.0")
    port = args.port or api_config.get("port", 8000)
    reload = args.reload or api_config.get("reload", False)

    logger.info("=" * 60)
    logger.info("AgenticIQA API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("=" * 60)
    logger.info(f"API Documentation: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    logger.info(f"Health Check: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/health")
    logger.info("=" * 60)

    # Import and run uvicorn
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Please run: pip install uvicorn[standard]")
        sys.exit(1)

    # Run server
    try:
        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=reload,
            workers=args.workers if not reload else 1,  # Workers not supported with reload
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down API server...")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
