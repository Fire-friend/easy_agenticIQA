#!/usr/bin/env python3
"""
Structured Execution Logger for AgenticIQA Pipeline

Logs execution metrics in JSON Lines format for analysis and debugging.
Tracks timing, token usage, costs, replanning statistics, and errors.

Usage:
    logger = ExecutionLogger(log_path="logs/execution.jsonl", level="INFO")
    logger.log_sample_start(sample_id="sample_001", query="...")
    logger.log_stage(sample_id="sample_001", stage="planner", duration_ms=1234, tokens=500)
    logger.log_sample_end(sample_id="sample_001", status="success", total_duration_ms=5678)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
import threading


class LogLevel(str, Enum):
    """Logging levels for execution logger."""
    INFO = "INFO"       # Summary per sample
    DEBUG = "DEBUG"     # Include prompts and responses
    TRACE = "TRACE"     # Full state dumps


class ExecutionLogger:
    """
    Structured logger for pipeline execution metrics.

    Writes JSON Lines format with configurable logging levels.
    Thread-safe for concurrent logging.
    """

    def __init__(
        self,
        log_path: Path,
        level: LogLevel = LogLevel.INFO,
        enable_rotation: bool = False,
        max_size_mb: int = 100
    ):
        """
        Initialize execution logger.

        Args:
            log_path: Path to JSON Lines log file
            level: Logging level (INFO/DEBUG/TRACE)
            enable_rotation: Enable log rotation by size
            max_size_mb: Maximum log file size before rotation
        """
        self.log_path = Path(log_path)
        self.level = LogLevel(level)
        self.enable_rotation = enable_rotation
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()

        # Create log directory
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Python logger for console output
        self.console_logger = logging.getLogger("ExecutionLogger")

    def _should_log_level(self, required_level: LogLevel) -> bool:
        """Check if current log level allows this entry."""
        level_hierarchy = {
            LogLevel.INFO: 0,
            LogLevel.DEBUG: 1,
            LogLevel.TRACE: 2
        }
        return level_hierarchy[self.level] >= level_hierarchy[required_level]

    def _rotate_if_needed(self):
        """Rotate log file if size exceeds limit."""
        if not self.enable_rotation:
            return

        if not self.log_path.exists():
            return

        file_size = self.log_path.stat().st_size
        if file_size >= self.max_size_bytes:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            rotated_path = self.log_path.parent / f"{self.log_path.stem}_{timestamp}.jsonl"
            self.log_path.rename(rotated_path)
            self.console_logger.info(f"Rotated log to {rotated_path}")

    def _write_log(self, entry: Dict[str, Any]):
        """Write log entry to file (thread-safe)."""
        with self._lock:
            self._rotate_if_needed()

            # Add timestamp if not present
            if "timestamp" not in entry:
                entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

            # Write to file
            try:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f.flush()
            except Exception as e:
                self.console_logger.error(f"Failed to write log entry: {e}")

    def log_sample_start(self, sample_id: str, query: str, image_path: str, reference_path: Optional[str] = None):
        """Log sample processing start."""
        if not self._should_log_level(LogLevel.INFO):
            return

        entry = {
            "event": "sample_start",
            "sample_id": sample_id,
            "query": query if self._should_log_level(LogLevel.DEBUG) else query[:100] + "...",
            "image_path": image_path,
            "reference_path": reference_path
        }
        self._write_log(entry)

    def log_stage(
        self,
        sample_id: str,
        stage: str,
        duration_ms: float,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        backend: Optional[str] = None,
        error: Optional[str] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None
    ):
        """
        Log individual stage execution.

        Args:
            sample_id: Sample identifier
            stage: Stage name (planner/executor/summarizer)
            duration_ms: Execution duration in milliseconds
            tokens: Token count (if available)
            cost: Estimated cost (if available)
            backend: Model backend used
            error: Error message (if failed)
            prompt: Full prompt (DEBUG level)
            response: Full response (DEBUG level)
        """
        if not self._should_log_level(LogLevel.INFO):
            return

        entry = {
            "event": "stage_execution",
            "sample_id": sample_id,
            "stage": stage,
            "duration_ms": round(duration_ms, 2),
            "backend": backend,
            "status": "error" if error else "success"
        }

        if tokens is not None:
            entry["tokens"] = tokens

        if cost is not None:
            entry["cost_usd"] = round(cost, 6)

        if error:
            entry["error"] = error

        # Add prompt/response for DEBUG level
        if self._should_log_level(LogLevel.DEBUG):
            if prompt:
                entry["prompt"] = prompt
            if response:
                entry["response"] = response

        self._write_log(entry)

    def log_replan(self, sample_id: str, iteration: int, reason: str):
        """Log replanning event."""
        if not self._should_log_level(LogLevel.INFO):
            return

        entry = {
            "event": "replan",
            "sample_id": sample_id,
            "iteration": iteration,
            "reason": reason
        }
        self._write_log(entry)

    def log_tool_execution(
        self,
        sample_id: str,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
        result: Optional[Any] = None
    ):
        """Log IQA tool execution."""
        if not self._should_log_level(LogLevel.DEBUG):
            return

        entry = {
            "event": "tool_execution",
            "sample_id": sample_id,
            "tool_name": tool_name,
            "duration_ms": round(duration_ms, 2),
            "status": "success" if success else "error"
        }

        if error:
            entry["error"] = error

        if result and self._should_log_level(LogLevel.TRACE):
            entry["result"] = result

        self._write_log(entry)

    def log_sample_end(
        self,
        sample_id: str,
        status: str,
        total_duration_ms: float,
        replan_count: int = 0,
        total_tokens: Optional[int] = None,
        total_cost: Optional[float] = None,
        error: Optional[str] = None,
        final_state: Optional[Dict] = None
    ):
        """
        Log sample processing completion.

        Args:
            sample_id: Sample identifier
            status: Final status (success/error)
            total_duration_ms: Total execution time
            replan_count: Number of replanning iterations
            total_tokens: Total tokens used
            total_cost: Total estimated cost
            error: Error message (if failed)
            final_state: Complete final state (TRACE level)
        """
        if not self._should_log_level(LogLevel.INFO):
            return

        entry = {
            "event": "sample_end",
            "sample_id": sample_id,
            "status": status,
            "total_duration_ms": round(total_duration_ms, 2),
            "replan_count": replan_count
        }

        if total_tokens is not None:
            entry["total_tokens"] = total_tokens

        if total_cost is not None:
            entry["total_cost_usd"] = round(total_cost, 6)

        if error:
            entry["error"] = error

        # Add full state dump for TRACE level
        if final_state and self._should_log_level(LogLevel.TRACE):
            entry["final_state"] = final_state

        self._write_log(entry)

    def log_batch_summary(
        self,
        total_samples: int,
        successful: int,
        failed: int,
        total_duration_sec: float,
        total_tokens: Optional[int] = None,
        total_cost: Optional[float] = None
    ):
        """Log batch processing summary."""
        entry = {
            "event": "batch_summary",
            "total_samples": total_samples,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / total_samples * 100, 2) if total_samples > 0 else 0,
            "total_duration_sec": round(total_duration_sec, 2),
            "avg_duration_sec": round(total_duration_sec / total_samples, 2) if total_samples > 0 else 0
        }

        if total_tokens is not None:
            entry["total_tokens"] = total_tokens
            entry["avg_tokens"] = round(total_tokens / total_samples, 0) if total_samples > 0 else 0

        if total_cost is not None:
            entry["total_cost_usd"] = round(total_cost, 4)
            entry["avg_cost_usd"] = round(total_cost / total_samples, 6) if total_samples > 0 else 0

        self._write_log(entry)


class CostEstimator:
    """
    Cost estimation for API calls.

    Uses configurable rate cards for different models.
    """

    # Default rate card (USD per 1M tokens)
    DEFAULT_RATES = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "qwen2.5-vl": {"input": 0.00, "output": 0.00}  # Local model
    }

    def __init__(self, custom_rates: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize cost estimator.

        Args:
            custom_rates: Custom rate card (overrides defaults)
        """
        self.rates = {**self.DEFAULT_RATES}
        if custom_rates:
            self.rates.update(custom_rates)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for API call.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        model_key = model.lower()
        for key in self.rates:
            if key in model_key:
                rates = self.rates[key]
                input_cost = (input_tokens / 1_000_000) * rates["input"]
                output_cost = (output_tokens / 1_000_000) * rates["output"]
                return input_cost + output_cost

        # Unknown model - return 0
        return 0.0
