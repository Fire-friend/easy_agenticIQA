# AgenticIQA

**AgenticIQA** is a modular agentic framework for Image Quality Assessment (IQA) that integrates vision-language models (VLMs) with traditional IQA tools. The system uses a **Planner-Executor-Summarizer** architecture to provide interpretable, query-aware quality assessment with human-aligned explanations.

## Overview

AgenticIQA decomposes image quality assessment into three specialized agent roles:

1. **Planner** - Analyzes user queries and images, generates structured task plans
2. **Executor** - Executes plans through four subtasks: distortion detection, analysis, tool selection, and execution
3. **Summarizer** - Fuses evidence with visual understanding to generate final answers and quality reasoning

### Key Features

- **Query-Aware Assessment**: Understands and responds to specific user questions about image quality
- **Interpretable Reasoning**: Provides human-readable explanations for quality assessments
- **VLM + IQA Fusion**: Combines vision-language models (GPT-4o, Claude, Qwen2.5-VL) with traditional IQA metrics
- **Flexible Architecture**: Supports Full-Reference (FR) and No-Reference (NR) quality assessment
- **Multiple Task Types**: Handles scoring (1-5 scale), explanation generation, and multiple-choice questions

### Architecture

```
User Input (query + image + optional reference)
  ↓
Planner → Task Plan JSON (query_type, scope, distortion_source,
                          distortions, reference_mode, control flags)
  ↓
Executor → Structured Evidence
    ├─ Distortion Detection → distortion_set
    ├─ Distortion Analysis → distortion_analysis
    ├─ Tool Selection → selected_tools
    └─ Tool Execution → quality_scores (normalized 1-5)
  ↓
Summarizer → Final Answer + Quality Reasoning (with optional replan request)
  ↓
Output (score/explanation/MCQ answer)
```

The system uses **LangGraph** to orchestrate the agent workflow with replanning capability (max 2 iterations by default).

---

## Quick Start

Get up and running in under 10 minutes:

```bash
# 1. Create conda environment
conda create -n agenticIQA python=3.10 -y
conda activate agenticIQA

# 2. Install core dependencies
pip install torch==2.3.0 torchvision==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Install IQA-PyTorch
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch && pip install -e . && cd ..

# 4. Set environment variables
export AGENTIC_ROOT=$(pwd)
export AGENTIC_DATA_ROOT=${AGENTIC_ROOT}/data
export AGENTIC_TOOL_HOME=${AGENTIC_ROOT}/iqa_tools
export AGENTIC_LOG_ROOT=${AGENTIC_ROOT}/logs

# 5. Configure API key (choose one)
export OPENAI_API_KEY=<your_openai_api_key>
# OR
export ANTHROPIC_API_KEY=<your_anthropic_api_key>

# 6. Verify environment
python scripts/check_env.py

# 7. Run a simple example (requires prepared data)
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/agenticiqa_eval/sample.jsonl \
  --output outputs/results.jsonl
```

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Installation

### Prerequisites

- **Python 3.10**
- **CUDA 12.1+** (for GPU acceleration)
- **Conda** (recommended for environment management)
- **Git**

### Step 1: Create Conda Environment

```bash
conda create -n agenticIQA python=3.10 -y
conda activate agenticIQA
```

### Step 2: Install PyTorch

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.3.0 torchvision==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Core Dependencies

```bash
# Install transformers and acceleration libraries
pip install transformers==4.42.0 accelerate==0.31.0 bitsandbytes==0.43.1

# Install VLM API clients
pip install openai==1.35.7 anthropic==0.30.0 google-genai

# Install image processing libraries
pip install opencv-python pillow numpy scipy scikit-image einops tqdm pyyaml

# Install framework and utilities
pip install pydantic==2.7.1 typer rich loguru

# Install LangGraph for orchestration
pip install langgraph langchain langchain-openai langchain-anthropic
```

Or install all at once using the requirements file:

```bash
pip install -r requirements.txt
```

### Step 4: Install IQA-PyTorch

```bash
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch
pip install -e .
cd ..
```

This library provides traditional IQA metrics (TOPIQ, QAlign, LPIPS, DISTS, BRISQUE, NIQE, etc.).

### Step 5: Configure Environment Variables

#### Required Variables

```bash
export AGENTIC_ROOT=/path/to/agenticIQA
export AGENTIC_DATA_ROOT=${AGENTIC_ROOT}/data
export AGENTIC_TOOL_HOME=${AGENTIC_ROOT}/iqa_tools
export AGENTIC_LOG_ROOT=${AGENTIC_ROOT}/logs
```

#### API Keys (choose based on your VLM backend)

```bash
# For OpenAI models (GPT-4o, GPT-4o-mini)
export OPENAI_API_KEY=<your_openai_api_key>

# For Anthropic models (Claude 3.5)
export ANTHROPIC_API_KEY=<your_anthropic_api_key>

# For Google models (Gemini)
export GOOGLE_API_KEY=<your_google_api_key>
```

#### Optional: Custom API Endpoints

If using custom or proxy endpoints:

```bash
export OPENAI_BASE_URL=https://your-endpoint/v1
export ANTHROPIC_BASE_URL=https://your-endpoint
export GOOGLE_API_BASE_URL=https://your-endpoint
```

### Step 6: Verify Installation

Run the environment validation script to ensure everything is correctly set up:

```bash
python scripts/check_env.py
```

This script checks:
- Python version
- Required packages and versions
- Environment variables
- GPU availability
- API key configuration
- IQA-PyTorch installation

---

## Project Structure

```
agenticIQA/
├── configs/                       # Configuration files
│   ├── model_backends.yaml        # VLM backend settings (planner, executor, summarizer)
│   ├── pipeline.yaml              # Pipeline orchestration settings
│   └── graph_settings.yaml        # Optional LangGraph parameters
│
├── src/                           # Source code
│   ├── agentic/                   # Core LangGraph-based pipeline
│   │   ├── graph.py               # LangGraph StateGraph definition
│   │   ├── nodes/                 # Agent node implementations
│   │   │   ├── planner.py         # Planner agent node
│   │   │   ├── executor.py        # Executor agent node
│   │   │   └── summarizer.py      # Summarizer agent node
│   │   └── tool_registry.py       # IQA tool registration and management
│   │
│   └── utils/                     # Common utilities, logging wrappers
│
├── iqa_tools/                     # IQA tool storage
│   ├── weights/                   # Third-party IQA model weights (auto-downloaded)
│   └── metadata/                  # Tool metadata and logistic parameters
│
├── data/                          # Dataset directory
│   ├── raw/                       # Original evaluation images/questions
│   │   ├── agenticiqa_eval/       # AgenticIQA-Eval dataset (MCQ tasks)
│   │   ├── tid2013/               # TID2013 dataset (scoring evaluation)
│   │   ├── bid/                   # BID dataset (scoring evaluation)
│   │   └── agiqa-3k/              # AGIQA-3K dataset (scoring evaluation)
│   ├── processed/                 # Processed manifests (JSONL format)
│   └── cache/                     # Intermediate results, prompt cache
│
├── scripts/                       # Utility scripts
│   ├── check_env.py               # Environment validation script
│   ├── launch_graph.py            # Launch LangGraph agent (interactive/batch)
│   ├── eval_agenticqa_eval.py     # MCQ accuracy evaluation
│   ├── eval_srocc_plcc.py         # SRCC/PLCC correlation calculation
│   └── generate_report.py         # Comprehensive evaluation report
│
├── logs/                          # Execution logs (auto-generated)
│   ├── cache/                     # Tool execution cache
│   ├── checkpoints/               # Pipeline checkpoints
│   └── intermediate/              # Intermediate results
│
├── outputs/                       # Pipeline output directory
│
├── docs/                          # Detailed documentation (Chinese)
│   ├── 00_overview.md             # System overview
│   ├── 01_environment_setup.md    # Environment setup guide
│   ├── 02_module_planner.md       # Planner module details
│   ├── 03_module_executor.md      # Executor module details
│   ├── 04_module_summarizer.md    # Summarizer module details
│   ├── 05_inference_pipeline.md   # Pipeline integration guide
│   └── 06_evaluation_protocol.md  # Evaluation procedures
│
├── tests/                         # Test suite
│
├── run_pipeline.py                # Main pipeline runner script
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment specification
├── paper.pdf                      # Original AgenticIQA research paper
├── CLAUDE.md                      # AI assistant guidance
└── README.md                      # This file
```

### Key Files

- **`run_pipeline.py`**: Main entry point for running the AgenticIQA pipeline on datasets
- **`configs/model_backends.yaml`**: Configure which VLM models to use for each agent (planner, executor, summarizer)
- **`configs/pipeline.yaml`**: Control pipeline behavior (replanning, timeouts, caching, checkpointing)
- **`scripts/check_env.py`**: Validate environment setup before running experiments
- **`requirements.txt`**: All Python package dependencies with versions
- **`environment.yml`**: Conda environment specification
- **`paper.pdf`**: Original research paper describing the AgenticIQA methodology

---

## Configuration

### Model Backend Configuration

Edit `configs/model_backends.yaml` to configure VLM backends for each module:

```yaml
# Planner module configuration
planner:
  backend: openai.gpt-4o          # Options: openai.gpt-4o, anthropic.claude-3.5-sonnet, qwen2.5-vl-local
  temperature: 0.0                 # Temperature for sampling (0.0 = deterministic)
  max_tokens: 2048
  top_p: 0.1

# Executor module configuration
executor:
  backend: openai.gpt-4o          # Can use different model than planner
  temperature: 0.0
  max_tokens: 4096                # Executor needs more tokens for detailed analysis
  top_p: 0.1

# Summarizer module configuration
summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 2048
  top_p: 0.1

# API Base URLs (optional)
api_endpoints:
  openai_base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
  anthropic_base_url: ${ANTHROPIC_BASE_URL:-https://api.anthropic.com}
  google_api_base_url: ${GOOGLE_API_BASE_URL:-}
```

#### Supported VLM Backends

- **`openai.gpt-4o`**: OpenAI GPT-4o (primary VLM, best performance)
- **`openai.gpt-4o-mini`**: Cheaper alternative for development/debugging
- **`anthropic.claude-3.5-sonnet`**: Anthropic Claude 3.5 Sonnet
- **`qwen2.5-vl-local`**: Local Qwen2.5-VL inference (requires GPU)
- **`google.gemini-pro-vision`**: Google Gemini Pro Vision

#### Temperature Settings

- **`temperature: 0.0`** (recommended): Deterministic outputs for reproducible results
- **`temperature: 0.3-0.7`**: More creative/diverse outputs (less reproducible)

### Pipeline Configuration

Edit `configs/pipeline.yaml` to control pipeline behavior:

```yaml
pipeline:
  max_replan: 2                    # Maximum replanning iterations (if evidence insufficient)

  timeout:
    planner: 60                    # Timeout per module in seconds
    executor: 300
    summarizer: 60

  cache_dir: ${AGENTIC_LOG_ROOT:-logs}/cache
  enable_cache: true               # Cache tool execution results by image hash

  log_path: ${AGENTIC_LOG_ROOT:-logs}/pipeline.log
  log_level: INFO                  # Options: DEBUG, INFO, WARNING, ERROR

  checkpoint:
    enable: true                   # Enable checkpointing for resume capability
    save_dir: ${AGENTIC_LOG_ROOT:-logs}/checkpoints
    save_interval: 10              # Save checkpoint every N samples

# Executor subtask settings
executor:
  tool_selection:
    max_tools: 5                   # Maximum number of IQA tools to select
    metadata_path: ${AGENTIC_TOOL_HOME:-iqa_tools}/metadata/tool_metadata.json

  tool_execution:
    score_normalization: true      # Normalize tool scores to 1-5 scale
    parallel_execution: false      # Run tools in parallel (experimental)
    timeout: 120                   # Timeout per tool in seconds

# Summarizer settings
summarizer:
  fusion:
    weight_tool_scores: 0.6        # Weight for IQA tool scores in final score
    weight_vlm_probability: 0.4    # Weight for VLM probability distribution

  replan_trigger:
    enable: true                   # Allow replanning if evidence insufficient
    insufficient_evidence_threshold: 0.3
```

#### Key Configuration Parameters

- **`max_replan`**: Controls how many times the Summarizer can request the Planner to regenerate the plan if evidence is insufficient (default: 2, prevents infinite loops)
- **`enable_cache`**: Cache tool execution results by image hash to avoid redundant computation
- **`enable_checkpoint`**: Save pipeline state periodically for resume capability on long runs
- **`weight_tool_scores` / `weight_vlm_probability`**: Controls fusion of IQA tool scores with VLM reasoning in final quality score

---

## Usage

### Basic Pipeline Run

Run the AgenticIQA pipeline on a dataset:

```bash
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/agenticiqa_eval/manifest.jsonl \
  --output outputs/results.jsonl
```

### Input Format

Input files should be in JSONL (JSON Lines) format, with one sample per line:

```jsonl
{"query": "What is the quality score of this image?", "image_path": "data/raw/image001.jpg", "reference_path": null, "ground_truth": 3.5}
{"query": "Does this image have blur artifacts?", "image_path": "data/raw/image002.jpg", "reference_path": null, "ground_truth": "Yes"}
```

**Fields:**
- `query` (required): User question about image quality
- `image_path` (required): Path to distorted/test image
- `reference_path` (optional): Path to reference image (for Full-Reference assessment, null for No-Reference)
- `ground_truth` (optional): Ground truth answer for evaluation

### Output Format

Output files are in JSONL format with pipeline results:

```jsonl
{"query": "What is the quality score?", "image_path": "...", "plan": {...}, "evidence": {...}, "result": {"answer": "3.2", "reasoning": "...", "confidence": 0.85}, "execution_time": 12.3}
```

**Output Fields:**
- `plan`: Planner's task plan (JSON structure)
- `evidence`: Executor's gathered evidence (distortions, tool scores)
- `result`: Summarizer's final answer with reasoning
- `execution_time`: Total processing time in seconds

### Running on Different Datasets

#### AgenticIQA-Eval (MCQ Tasks)

```bash
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/agenticiqa_eval/planner.jsonl \
  --output outputs/agenticiqa_eval_planner_results.jsonl
```

#### TID2013 (Scoring Evaluation)

```bash
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/tid2013/manifest.jsonl \
  --output outputs/tid2013_scores.jsonl
```

#### BID Dataset

```bash
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/bid/manifest.jsonl \
  --output outputs/bid_scores.jsonl
```

#### AGIQA-3K Dataset

```bash
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/agiqa_3k/manifest.jsonl \
  --output outputs/agiqa_3k_scores.jsonl
```

### Interactive Mode (LangGraph)

Use `launch_graph.py` for interactive exploration:

```bash
# Interactive mode (single query)
python scripts/launch_graph.py \
  --config configs/graph_settings.yaml \
  --mode interactive

# Batch mode (process multiple samples)
python scripts/launch_graph.py \
  --config configs/graph_settings.yaml \
  --mode batch \
  --input data/processed/agenticiqa_eval/sample.jsonl
```

---

## Evaluation

### MCQ Accuracy (AgenticIQA-Eval)

Evaluate multiple-choice question accuracy on the AgenticIQA-Eval dataset:

```bash
python scripts/eval_agenticqa_eval.py --input outputs/agenticiqa_eval_results.jsonl
```

**Output:**
- Overall accuracy
- Per-module accuracy (Planner, Executor, Summarizer)
- Confusion matrix

### SRCC/PLCC Correlation (Scoring Datasets)

Calculate Spearman (SRCC) and Pearson (PLCC) correlation for scoring tasks:

```bash
python scripts/eval_srocc_plcc.py --input outputs/tid2013_scores.jsonl
```

**Metrics:**
- **SRCC (Spearman Rank Correlation Coefficient)**: Measures monotonic relationship between predicted and ground truth scores
- **PLCC (Pearson Linear Correlation Coefficient)**: Measures linear relationship between predicted and ground truth scores
- **KRCC (Kendall Rank Correlation Coefficient)**: Another rank correlation metric

### Comprehensive Report

Generate a comprehensive evaluation report across all datasets:

```bash
python scripts/generate_report.py --output reports/evaluation_report.md
```

This aggregates results from all experiments and produces a markdown report with:
- MCQ accuracy by module
- SRCC/PLCC by dataset
- Execution time statistics
- Error analysis

---

## Development

### Environment Validation

Before running experiments, validate your environment setup:

```bash
python scripts/check_env.py
```

This checks:
- Python version (3.10 required)
- Package versions (PyTorch, transformers, etc.)
- Environment variables (AGENTIC_ROOT, API keys)
- GPU availability and CUDA version
- IQA-PyTorch installation
- API connectivity (optional)

### Cost Management

AgenticIQA uses commercial VLM APIs which can incur costs. Here are strategies to reduce costs:

#### 1. Use Cheaper Models for Development

Edit `configs/model_backends.yaml` to use `openai.gpt-4o-mini` during development:

```yaml
planner:
  backend: openai.gpt-4o-mini     # ~10x cheaper than gpt-4o
  temperature: 0.0
```

#### 2. Enable Caching

Ensure caching is enabled in `configs/pipeline.yaml`:

```yaml
pipeline:
  enable_cache: true               # Cache tool execution results
  cache_dir: ${AGENTIC_LOG_ROOT:-logs}/cache
```

This caches tool execution results by image hash, avoiding redundant computation.

#### 3. Batch Processing

Use checkpointing for long runs to avoid re-processing on failures:

```yaml
pipeline:
  checkpoint:
    enable: true
    save_interval: 10              # Save every 10 samples
```

#### 4. Use Local Models

For zero API cost, use local Qwen2.5-VL inference:

```yaml
planner:
  backend: qwen2.5-vl-local
```

**Note:** Requires a CUDA-capable GPU with sufficient memory (~24GB recommended).

### Logging and Debugging

#### Log Files

- **Pipeline log**: `logs/pipeline.log` - Overall pipeline execution
- **LangGraph log**: `logs/langgraph.log` - Agent state transitions
- **Tool execution log**: `logs/tool_execution.log` - IQA tool runs

#### Debug Mode

Enable debug mode in `configs/pipeline.yaml`:

```yaml
langgraph:
  debug_mode: true                 # Enable verbose logging
  save_graph_visualization: true   # Save graph execution visualization
```

#### Intermediate Results

Intermediate results are saved if enabled:

```yaml
data:
  save_intermediate_results: true
  intermediate_dir: ${AGENTIC_LOG_ROOT:-logs}/intermediate
```

This saves:
- Planner output (task plans)
- Executor output (evidence)
- Summarizer input/output

---

## Troubleshooting

### Missing Dependencies

**Problem:** `ImportError: No module named 'transformers'`

**Solution:**
1. Verify conda environment is activated: `conda activate agenticIQA`
2. Install dependencies: `pip install -r requirements.txt`
3. Run environment validation: `python scripts/check_env.py`

---

### API Rate Limiting

**Problem:** `RateLimitError: Too many requests`

**Solution:**
1. **Enable caching** to reduce redundant API calls:
   ```yaml
   pipeline:
     enable_cache: true
   ```

2. **Use batch processing with checkpointing** to resume on failure:
   ```yaml
   pipeline:
     checkpoint:
       enable: true
       save_interval: 10
   ```

3. **Reduce concurrent requests** by processing samples sequentially

4. **Wait and retry** - rate limits typically reset after a short period

---

### CUDA / GPU Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
1. **Reduce batch size** (if using local models):
   ```yaml
   resources:
     batch:
       batch_size: 1
   ```

2. **Use CPU-only mode** for small-scale testing:
   ```yaml
   resources:
     gpu:
       enable: false
   ```

3. **Clear GPU cache** periodically:
   ```yaml
   resources:
     memory:
       clear_cache_interval: 50
   ```

4. **Use API-based models** (GPT-4o, Claude) instead of local inference

**Problem:** `RuntimeError: CUDA not available`

**Solution:**
1. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reinstall PyTorch with CUDA support: `pip install torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121`
3. Check NVIDIA driver: `nvidia-smi`

---

### IQA Tool Weight Download Issues

**Problem:** Tool weights fail to download automatically

**Solution:**
1. **Manual download:** Download weights from IQA-PyTorch repository and place in `iqa_tools/weights/`
2. **Check network connection:** Ensure access to model hosting platforms (HuggingFace, etc.)
3. **Verify disk space:** Weight files can be large (several GB total)
4. **Use alternative tools:** Configure alternative IQA metrics if specific tools fail:
   ```yaml
   executor:
     tool_selection:
       alternatives: [qalign, brisque, niqe, lpips]
   ```

---

### Environment Variable Not Set

**Problem:** `KeyError: 'AGENTIC_ROOT'` or similar

**Solution:**
1. Set required environment variables:
   ```bash
   export AGENTIC_ROOT=$(pwd)
   export AGENTIC_DATA_ROOT=${AGENTIC_ROOT}/data
   export AGENTIC_TOOL_HOME=${AGENTIC_ROOT}/iqa_tools
   export AGENTIC_LOG_ROOT=${AGENTIC_ROOT}/logs
   ```

2. Add to `.bashrc` or `.zshrc` for persistence:
   ```bash
   echo 'export AGENTIC_ROOT=/path/to/agenticIQA' >> ~/.bashrc
   source ~/.bashrc
   ```

3. Verify: `python scripts/check_env.py`

---

### FAQ

**Q: Can I use multiple VLM backends simultaneously?**

A: Yes! You can configure different backends for each module. For example:
```yaml
planner:
  backend: openai.gpt-4o
executor:
  backend: qwen2.5-vl-local        # Local model to save costs
summarizer:
  backend: anthropic.claude-3.5-sonnet
```

**Q: What's the difference between Full-Reference (FR) and No-Reference (NR) assessment?**

A:
- **Full-Reference (FR)**: Compares distorted image against a pristine reference image (e.g., LPIPS, DISTS)
- **No-Reference (NR)**: Assesses quality without a reference (e.g., BRISQUE, NIQE)

AgenticIQA automatically detects the mode based on whether `reference_path` is provided.

**Q: How long does it take to process one image?**

A: Processing time varies by configuration:
- **API-based VLM (GPT-4o)**: 10-30 seconds per image
- **Local VLM (Qwen2.5-VL)**: 5-15 seconds per image (with GPU)
- **With replanning**: Up to 2x longer (if evidence insufficient)

**Q: Can I add custom IQA tools?**

A: Yes! Register new tools in `src/agentic/tool_registry.py` and add metadata to `iqa_tools/metadata/tool_metadata.json`. See the IQA-PyTorch documentation for supported metrics.

**Q: What datasets are supported?**

A: Currently supported:
- **AgenticIQA-Eval**: MCQ tasks for Planner/Executor/Summarizer evaluation
- **TID2013**: 25 distortion types, 3000 images
- **BID**: Authentic distortions, 23,200 images
- **AGIQA-3K**: AI-generated images, 3,000 images

You can add custom datasets by preparing JSONL manifests in the same format.

**Q: Is training/fine-tuning required?**

A: No! AgenticIQA is an **inference-only** framework. It uses:
- Pretrained VLM APIs (GPT-4o, Claude) or local models (Qwen2.5-VL)
- Pretrained IQA metrics from IQA-PyTorch

No training or fine-tuning is involved.

---

## Documentation

### Detailed Documentation (Chinese)

Comprehensive implementation documentation is available in the `docs/` directory:

- **[docs/00_overview.md](docs/00_overview.md)** - System overview and design philosophy
- **[docs/01_environment_setup.md](docs/01_environment_setup.md)** - Detailed environment setup guide
- **[docs/02_module_planner.md](docs/02_module_planner.md)** - Planner module implementation details
- **[docs/03_module_executor.md](docs/03_module_executor.md)** - Executor module and subtasks
- **[docs/04_module_summarizer.md](docs/04_module_summarizer.md)** - Summarizer module and fusion logic
- **[docs/05_inference_pipeline.md](docs/05_inference_pipeline.md)** - LangGraph pipeline integration
- **[docs/06_evaluation_protocol.md](docs/06_evaluation_protocol.md)** - Evaluation procedures and metrics

### Additional Resources

- **[CLAUDE.md](CLAUDE.md)** - Guidance for AI assistants working with this codebase
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - High-level implementation summary
- **[paper.pdf](paper.pdf)** - Original AgenticIQA research paper

### External Links

- **[IQA-PyTorch Repository](https://github.com/chaofengc/IQA-PyTorch)** - Traditional IQA metrics library
- **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)** - Agent orchestration framework

---

## License

Please refer to the license information in the original research paper and repository.

---

## Citation

If you use AgenticIQA in your research, please cite:

```bibtex
@article{agenticIQA2024,
  title={AgenticIQA: Agentic Framework for Image Quality Assessment},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2024}
}
```

---

## Contact

For questions, issues, or contributions, please refer to the project repository or contact the authors.
