"""
FastAPI application for AgenticIQA REST API.
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import __version__
from src.api.schemas import (
    AssessPathRequest,
    AssessResponse,
    HealthResponse,
)
from src.api.pipeline_wrapper import PipelineWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global instance (initialized in lifespan)
pipeline_wrapper: Optional[PipelineWrapper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global pipeline_wrapper

    # Startup
    logger.info("Starting AgenticIQA API server...")

    # Initialize pipeline wrapper
    pipeline_wrapper = PipelineWrapper()

    logger.info("API server initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="AgenticIQA API",
    description="REST API for Image Quality Assessment using Agentic Framework",
    version=__version__,
    lifespan=lifespan
)


# Configure CORS
def setup_cors(app: FastAPI, config: dict):
    """Setup CORS middleware from configuration."""
    cors_config = config.get("api", {}).get("cors", {})
    if cors_config.get("enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("allow_origins", ["*"]),
            allow_credentials=cors_config.get("allow_credentials", False),
            allow_methods=cors_config.get("allow_methods", ["GET", "POST"]),
            allow_headers=cors_config.get("allow_headers", ["*"]),
        )
        logger.info("CORS middleware configured")


# Load config and setup CORS
try:
    from src.utils.config import load_config
    config = load_config("configs/api.yaml")
    setup_cors(app, config)
except Exception as e:
    logger.warning(f"Failed to setup CORS from config: {e}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error_type": "internal_server_error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        }
    )


# API Endpoints

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and version information.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now().isoformat()
    )


@app.post("/assess", response_model=AssessResponse, tags=["Assessment"])
async def assess_image_path(request: AssessPathRequest):
    """
    Assess image quality via file path.

    Accepts image file paths (local or accessible to server) and returns quality assessment.
    Supports both No-Reference (single image) and Full-Reference (with reference image) modes.
    """
    # Execute pipeline directly with provided paths
    result = await pipeline_wrapper.assess_image(
        query=request.query,
        image_path=request.image_path,
        reference_path=request.reference_path,
        max_replan_iterations=request.max_replan_iterations
    )

    return result


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AgenticIQA API",
        "version": __version__,
        "description": "REST API for Image Quality Assessment",
        "docs_url": "/docs",
        "health_check": "/health"
    }
