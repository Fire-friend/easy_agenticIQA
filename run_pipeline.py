#!/usr/bin/env python3
"""
AgenticIQA Batch Inference Pipeline

Processes datasets in JSONL format through the complete Planner→Executor→Summarizer workflow
with resume capability, progress tracking, structured logging, and error handling.

Usage:
    python run_pipeline.py --input data/processed/dataset.jsonl --output results/output.jsonl

For more options:
    python run_pipeline.py --help
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Set, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler

from src.agentic.graph import run_pipeline
from src.utils.config import load_model_backends
from src.utils.execution_logger import ExecutionLogger, LogLevel, CostEstimator

# Initialize Typer app and Rich console
app = typer.Typer(
    name="run_pipeline",
    help="AgenticIQA Batch Inference Pipeline",
    add_completion=False
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> list:
    """Load JSONL file and return list of samples."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON on line {line_num}: {e}")
                continue
    return samples


def apply_config_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply configuration overrides using dot notation.

    Args:
        config: Configuration dictionary
        overrides: List of override strings (e.g., ["planner.backend=gpt-4o-mini", "executor.temperature=0.5"])

    Returns:
        Updated configuration dictionary
    """
    for override in overrides:
        if '=' not in override:
            logger.warning(f"Invalid override format (expected KEY=VALUE): {override}")
            continue

        key_path, value = override.split('=', 1)
        keys = key_path.split('.')

        # Navigate to the nested location
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value (try to parse as number/bool if possible)
        final_key = keys[-1]
        try:
            # Try parsing as number
            if '.' in value:
                current[final_key] = float(value)
            else:
                current[final_key] = int(value)
        except ValueError:
            # Try parsing as boolean
            if value.lower() in ('true', 'false'):
                current[final_key] = value.lower() == 'true'
            else:
                # Keep as string
                current[final_key] = value

        logger.info(f"Override applied: {key_path} = {current[final_key]}")

    return config


def get_processed_sample_ids(output_path: Path) -> Set[str]:
    """Extract sample_ids from existing output file for resume capability."""
    processed_ids = set()
    if not output_path.exists():
        return processed_ids
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    if 'sample_id' in result:
                        processed_ids.add(result['sample_id'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Could not read existing output file: {e}")
    
    return processed_ids


def write_result(output_path: Path, result: Dict[str, Any]):
    """Write result to output file."""
    # Serialize to JSON
    json_line = json.dumps(result, ensure_ascii=False)

    try:
        # Append to main file
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
            f.flush()  # Ensure written to disk
    except Exception as e:
        logger.error(f"Failed to write result: {e}")
        raise


def serialize_state_for_output(state: Dict[str, Any], sample: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
    """
    Convert pipeline state to JSON-serializable output format.
    
    Handles Pydantic models by converting to dicts and removes non-serializable objects.
    """
    output = {
        "sample_id": sample.get("sample_id", "unknown"),
        "query": state.get("query"),
        "image_path": state.get("image_path"),
        "reference_path": state.get("reference_path"),
    }
    
    # Serialize plan (with JSON-compatible datetime serialization)
    plan = state.get("plan")
    if plan:
        if hasattr(plan, 'model_dump'):
            output["plan"] = plan.model_dump(mode='json')
        elif isinstance(plan, dict):
            output["plan"] = plan
        else:
            output["plan"] = str(plan)

    # Serialize executor evidence (with JSON-compatible datetime serialization)
    evidence = state.get("executor_evidence")
    if evidence:
        if hasattr(evidence, 'model_dump'):
            output["executor_evidence"] = evidence.model_dump(mode='json')
        elif isinstance(evidence, dict):
            output["executor_evidence"] = evidence
        else:
            output["executor_evidence"] = str(evidence)

    # Serialize summarizer result (with JSON-compatible datetime serialization)
    result = state.get("summarizer_result")
    if result:
        if hasattr(result, 'model_dump'):
            output["summarizer_result"] = result.model_dump(mode='json')
        elif isinstance(result, dict):
            output["summarizer_result"] = result
        else:
            output["summarizer_result"] = str(result)
    
    # Add metadata
    output["metadata"] = {
        "execution_time_ms": round(execution_time * 1000, 2),
        "replan_count": state.get("iteration_count", 0),
        "final_status": "error" if state.get("error") else "success",
        "error_details": state.get("error"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    
    return output


@app.command()
def main(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input JSONL file with samples"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output JSONL file for results"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to pipeline config YAML"),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing output (skip processed samples)"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="Limit number of samples to process"),
    max_replan_iterations: int = typer.Option(2, "--max-replan", help="Maximum replanning iterations per sample"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging (DEBUG level)"),
    execution_log: Optional[Path] = typer.Option(None, "--execution-log", "-e", help="Path to execution metrics log (JSONL format)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Execution log level: INFO/DEBUG/TRACE"),
    backend_override: Optional[List[str]] = typer.Option(None, "--backend-override", "-b", help="Override backend config (e.g., planner.backend=gpt-4o-mini)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and data without executing pipeline"),
):
    """
    Run AgenticIQA batch inference pipeline on a dataset.
    
    Processes samples from INPUT_PATH through Planner→Executor→Summarizer workflow
    and writes results to OUTPUT_PATH in JSONL format.
    
    Example:
        python run_pipeline.py -i data/processed/test.jsonl -o results/output.jsonl --resume
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate input file
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        raise typer.Exit(code=1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        if config_path:
            logger.info(f"Loading configuration from {config_path}")
            # TODO: Support custom config path
            # For now, use default config loading
        logger.info("Loading default configuration")
        config = load_model_backends()

        # Apply backend overrides
        if backend_override:
            logger.info(f"Applying {len(backend_override)} backend override(s)")
            config = apply_config_overrides(config, backend_override)

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(code=1)

    # Initialize execution logger if requested
    exec_logger = None
    cost_estimator = None
    if execution_log:
        try:
            exec_logger = ExecutionLogger(
                log_path=execution_log,
                level=LogLevel(log_level.upper()),
                enable_rotation=True,
                max_size_mb=100
            )
            cost_estimator = CostEstimator()
            logger.info(f"Execution logging enabled: {execution_log} (level: {log_level})")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize execution logger: {e}[/yellow]")
            exec_logger = None
    
    # Load input samples
    console.print(f"\n[bold blue]Loading samples from {input_path}...[/bold blue]")
    samples = load_jsonl(input_path)
    
    if not samples:
        console.print("[red]No samples found in input file[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[green]Loaded {len(samples)} samples[/green]")
    
    # Handle resume mode
    processed_ids = set()
    if resume:
        processed_ids = get_processed_sample_ids(output_path)
        if processed_ids:
            console.print(f"[yellow]Resume mode: Skipping {len(processed_ids)} already-processed samples[/yellow]")
    
    # Filter samples
    samples_to_process = [s for s in samples if s.get('sample_id') not in processed_ids]
    
    if max_samples:
        samples_to_process = samples_to_process[:max_samples]
        console.print(f"[yellow]Limiting to first {max_samples} samples[/yellow]")
    
    if not samples_to_process:
        console.print("[green]All samples already processed![/green]")
        return

    # Dry-run mode: Validate and print execution plan
    if dry_run:
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]DRY RUN MODE - Execution Plan[/bold cyan]")
        console.print("=" * 60)
        console.print(f"\n[bold]Input:[/bold]")
        console.print(f"  File: {input_path}")
        console.print(f"  Total samples loaded: {len(samples)}")
        console.print(f"  Samples to process: {len(samples_to_process)}")

        if resume and processed_ids:
            console.print(f"  Skipped (already processed): {len(processed_ids)}")

        console.print(f"\n[bold]Output:[/bold]")
        console.print(f"  File: {output_path}")
        console.print(f"  Directory exists: {output_path.parent.exists()}")
        console.print(f"  File writable: {output_path.parent.is_dir() if output_path.parent.exists() else False}")

        console.print(f"\n[bold]Configuration:[/bold]")
        console.print(f"  Config file: {config_path if config_path else 'Default (configs/model_backends.yaml)'}")
        console.print(f"  Max replan iterations: {max_replan_iterations}")
        if backend_override:
            console.print(f"  Backend overrides: {len(backend_override)}")
            for override in backend_override:
                console.print(f"    - {override}")

        if execution_log:
            console.print(f"\n[bold]Logging:[/bold]")
            console.print(f"  Execution log: {execution_log}")
            console.print(f"  Log level: {log_level}")

        # Estimate cost if available
        if cost_estimator and config:
            console.print(f"\n[bold]Cost Estimation:[/bold]")
            console.print(f"  [yellow]Note: Actual costs may vary based on prompt sizes and model responses[/yellow]")
            # TODO: Add more detailed cost estimation based on average token counts

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ Validation complete. Ready to run.[/bold green]")
        console.print("[cyan]Remove --dry-run flag to execute the pipeline.[/cyan]")
        console.print("=" * 60 + "\n")
        return

    console.print(f"[bold]Processing {len(samples_to_process)} samples...[/bold]\n")

    # Statistics
    stats = {
        "total": len(samples_to_process),
        "success": 0,
        "errors": 0,
        "start_time": time.time()
    }
    
    # Process samples with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing samples...", total=len(samples_to_process))
        
        for idx, sample in enumerate(samples_to_process, 1):
            sample_id = sample.get('sample_id', f'sample_{idx}')
            query = sample.get('query', '')
            image_path = sample.get('image_path', '')
            reference_path = sample.get('reference_path')

            progress.update(task, description=f"[cyan]Processing {sample_id}...")
            logger.info(f"[{idx}/{len(samples_to_process)}] Processing sample: {sample_id}")

            # Log sample start
            if exec_logger:
                exec_logger.log_sample_start(sample_id, query, image_path, reference_path)

            start_time = time.time()

            try:
                # Run pipeline
                final_state = run_pipeline(
                    query=query,
                    image_path=image_path,
                    reference_path=reference_path,
                    max_replan_iterations=max_replan_iterations
                )

                execution_time = time.time() - start_time

                # Serialize and write result
                output = serialize_state_for_output(final_state, sample, execution_time)
                write_result(output_path, output)

                stats["success"] += 1
                logger.info(f"✓ Completed {sample_id} in {execution_time:.2f}s")

                # Log sample end (success)
                if exec_logger:
                    replan_count = final_state.get("iteration_count", 0)
                    exec_logger.log_sample_end(
                        sample_id=sample_id,
                        status="success",
                        total_duration_ms=execution_time * 1000,
                        replan_count=replan_count
                    )
                
            except Exception as e:
                execution_time = time.time() - start_time
                stats["errors"] += 1
                logger.error(f"✗ Failed {sample_id}: {e}")

                # Log sample end (error)
                if exec_logger:
                    exec_logger.log_sample_end(
                        sample_id=sample_id,
                        status="error",
                        total_duration_ms=execution_time * 1000,
                        error=str(e)
                    )

                # Write partial result with error
                error_output = {
                    "sample_id": sample_id,
                    "query": query,
                    "image_path": image_path,
                    "reference_path": reference_path,
                    "metadata": {
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "final_status": "error",
                        "error_details": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                    }
                }
                try:
                    write_result(output_path, error_output)
                except Exception as write_error:
                    logger.error(f"Could not write error result: {write_error}")
            
            progress.update(task, advance=1)
    
    # Print summary
    total_time = time.time() - stats["start_time"]

    # Log batch summary
    if exec_logger:
        exec_logger.log_batch_summary(
            total_samples=stats["total"],
            successful=stats["success"],
            failed=stats["errors"],
            total_duration_sec=total_time
        )

    console.print("\n" + "=" * 60)
    console.print("[bold green]Batch Processing Complete![/bold green]")
    console.print(f"  Total samples: {stats['total']}")
    console.print(f"  [green]Successful: {stats['success']}[/green]")
    console.print(f"  [red]Errors: {stats['errors']}[/red]")
    console.print(f"  Total time: {total_time:.2f}s")
    console.print(f"  Average time: {total_time/stats['total']:.2f}s/sample")
    console.print(f"\nResults saved to: {output_path}")
    if execution_log:
        console.print(f"Execution log: {execution_log}")
    console.print("=" * 60 + "\n")

    if stats["errors"] > 0:
        console.print(f"[yellow]Warning: {stats['errors']} samples failed. Check logs for details.[/yellow]")


if __name__ == "__main__":
    app()
