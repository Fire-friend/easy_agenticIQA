#!/usr/bin/env python3
"""
Retry Logic with Exponential Backoff

Provides decorators and utilities for retrying failed API calls with
exponential backoff, rate limit handling, and model fallback.

Usage:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def api_call():
        # Your API call here
        pass
"""

import time
import logging
from functools import wraps
from typing import Callable, Optional, Tuple, Any, List
import random


logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base class for errors that should trigger retry."""
    pass


class RateLimitError(RetryableError):
    """API rate limit exceeded (429)."""
    pass


class TransientError(RetryableError):
    """Transient error (network, timeout, 5xx)."""
    pass


class NonRetryableError(Exception):
    """Base class for errors that should NOT trigger retry."""
    pass


class AuthenticationError(NonRetryableError):
    """Authentication failed (401, 403)."""
    pass


class InvalidRequestError(NonRetryableError):
    """Invalid request (400, 4xx except 429)."""
    pass


def classify_error(exception: Exception) -> Exception:
    """
    Classify exception into retryable/non-retryable categories.

    Args:
        exception: Original exception

    Returns:
        Classified exception (may be wrapped)
    """
    error_msg = str(exception).lower()

    # Check for rate limit errors
    if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
        return RateLimitError(f"Rate limit exceeded: {exception}")

    # Check for authentication errors
    if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
        return AuthenticationError(f"Authentication failed: {exception}")

    # Check for invalid request errors
    if "400" in error_msg or "invalid" in error_msg:
        return InvalidRequestError(f"Invalid request: {exception}")

    # Check for transient errors
    if any(code in error_msg for code in ["500", "502", "503", "504"]):
        return TransientError(f"Server error: {exception}")

    if "timeout" in error_msg or "connection" in error_msg:
        return TransientError(f"Network error: {exception}")

    # Default to transient (retry)
    return TransientError(f"Transient error: {exception}")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential multiplier for delay
        jitter: Add random jitter to delay

    Returns:
        Decorated function

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_api():
            return requests.get("https://api.example.com")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    classified_error = classify_error(e)
                    last_exception = classified_error

                    # Fast-fail for non-retryable errors
                    if isinstance(classified_error, NonRetryableError):
                        logger.error(f"Non-retryable error in {func.__name__}: {classified_error}")
                        raise classified_error

                    # Last attempt - raise error
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded in {func.__name__}")
                        raise classified_error

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    # Add jitter to avoid thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    # Special handling for rate limits - use longer delay
                    if isinstance(classified_error, RateLimitError):
                        delay = max(delay, 5.0)  # At least 5 seconds for rate limits

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed in {func.__name__}: {classified_error}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper

    return decorator


class ModelFallbackChain:
    """
    Model fallback mechanism for graceful degradation.

    Tries primary model first, falls back to alternatives on failure.
    """

    def __init__(self, models: List[str], retry_per_model: int = 2):
        """
        Initialize fallback chain.

        Args:
            models: List of models in priority order (e.g., ["gpt-4o", "gpt-4o-mini", "claude-3-haiku"])
            retry_per_model: Number of retries per model before falling back
        """
        self.models = models
        self.retry_per_model = retry_per_model
        self.fallback_history = []

    def execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """
        Execute function with model fallback.

        Args:
            func: Function to call (should accept 'model' kwarg)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            (result, model_used): Function result and model that succeeded

        Raises:
            Exception: If all models fail
        """
        last_exception = None

        for model in self.models:
            logger.info(f"Trying model: {model}")

            # Try with retries
            for attempt in range(self.retry_per_model):
                try:
                    kwargs['model'] = model
                    result = func(*args, **kwargs)

                    # Record successful model
                    self.fallback_history.append({
                        "model": model,
                        "success": True,
                        "attempt": attempt + 1
                    })

                    if model != self.models[0]:
                        logger.warning(f"Fallback succeeded with model: {model}")

                    return result, model

                except Exception as e:
                    classified_error = classify_error(e)
                    last_exception = classified_error

                    # Fast-fail for non-retryable errors (except auth - might work with different model)
                    if isinstance(classified_error, InvalidRequestError):
                        logger.error(f"Invalid request with {model}: {classified_error}")
                        break  # Try next model

                    logger.warning(f"Attempt {attempt + 1}/{self.retry_per_model} failed with {model}: {classified_error}")

                    if attempt < self.retry_per_model - 1:
                        delay = 1.0 * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)

            # Record failed model
            self.fallback_history.append({
                "model": model,
                "success": False,
                "error": str(last_exception)
            })

            logger.error(f"Model {model} failed after {self.retry_per_model} attempts")

        # All models failed
        logger.error(f"All models failed. Tried: {', '.join(self.models)}")
        raise Exception(f"All models failed. Last error: {last_exception}")

    def get_fallback_history(self) -> List[dict]:
        """Get history of fallback attempts."""
        return self.fallback_history

    def reset_history(self):
        """Clear fallback history."""
        self.fallback_history = []


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to function execution.

    Args:
        timeout_seconds: Maximum execution time

    Note:
        Uses signal module on Unix systems. May not work on Windows.
    """
    import signal

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} exceeded timeout of {timeout_seconds}s")

            # Set alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)

            return result

        return wrapper

    return decorator
