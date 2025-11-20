# Project Context

## Purpose
**AgenticIQA** is a modular agentic framework for Image Quality Assessment (IQA) that integrates vision-language models (VLMs) with traditional IQA tools. The system uses a **Planner-Executor-Summarizer** architecture to provide interpretable, query-aware quality assessment with human-aligned explanations.

### Core Goals
- Provide query-aware image quality assessment using agentic reasoning
- Decompose IQA into specialized agent roles (Planner, Executor, Summarizer)
- Integrate VLM understanding with traditional IQA metrics
- Generate human-interpretable quality explanations
- Support both Full-Reference (FR) and No-Reference (NR) quality assessment
- Handle multiple task types: scoring, explanation generation, multiple-choice questions

## Tech Stack

### Core Framework
- **Python 3.10** - Primary language
- **PyTorch 2.3.0** + torchvision 0.18.0 - Deep learning framework
- **LangGraph** - Agent workflow orchestration (StateGraph for Planner→Executor→Summarizer)
- **LangChain** - LLM integration utilities

### Vision-Language Models (VLMs)
- **GPT-4o** (OpenAI) - Primary VLM backend
- **Claude 3.5** (Anthropic) - Alternative VLM
- **Qwen2.5-VL** - Open-source alternative (local or API)
- **Google Gemini** - Alternative VLM
- Transformers 4.42.0, Accelerate 0.31.0, BitsAndBytes 0.43.1 for local inference

### IQA Tools
- **IQA-PyTorch** - Traditional IQA metrics library (TOPIQ, QAlign, LPIPS, DISTS, BRISQUE, NIQE, etc.)
- Integration via tool registry system

### Utilities & Infrastructure
- **Pydantic 2.7.1** - Data validation and state modeling
- **Typer** - CLI interfaces
- **Rich** - Terminal formatting
- **Loguru** - Logging
- **PyYAML** - Configuration management
- **OpenCV, Pillow** - Image processing
- **NumPy, SciPy, scikit-image** - Scientific computing

## Project Conventions

### Code Style
- **Minimize file creation**: Avoid creating complex project structures; categorize new functions/classes into existing code structure
- **Avoid unnecessary optionality**: Remove unnecessary execution paths; keep logic clear and concise
- **Reuse over redundancy**: Reuse existing code rather than creating redundant implementations
- **No proactive fallbacks**: Don't create fallback mechanisms unless explicitly required
- **NEVER create unnecessary files**: Only create files when absolutely necessary
- **ALWAYS prefer editing existing files** over creating new ones
- **NO documentation files unless requested**: Never proactively create *.md or README files
- **Chinese comments allowed**: Code comments can be in Chinese

### Core Coding Values
- **Read documentation carefully** - Be ashamed of guessing APIs; be proud of reading docs
- **Seek clarification** - Be ashamed of vague execution; be proud of confirmation
- **Validate with users** - Be ashamed of armchair theorizing; be proud of validation
- **Reuse existing APIs** - Be ashamed of inventing new APIs; be proud of reusing
- **Proactive testing** - Be ashamed of skipping validation; be proud of testing
- **Follow conventions** - Be ashamed of breaking architecture; be proud of standards
- **Honest uncertainty** - Be ashamed of pretending; be proud of "I don't know"
- **Careful refactoring** - Be ashamed of blind edits; be proud of careful changes

### Architecture Patterns

#### Three-Agent Pipeline (LangGraph Orchestration)
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

#### Key Architectural Decisions
- **LangGraph StateGraph**: Manages state transitions and orchestration
- **Replanning Loop**: Summarizer can trigger replanning if evidence insufficient (max 2 iterations)
- **Shared VLM Backend**: All agents use same VLM but different prompts
- **Tool Registry Pattern**: IQA tools registered and managed centrally
- **State Persistence**: All intermediate results maintained in LangGraph state
- **Score Normalization**: All tool outputs normalized to 1-5 scale
- **Weighted Fusion**: Combines tool scores with VLM probability distribution

#### Directory Structure
```
agenticIQA/
├── configs/                    # YAML configurations
│   ├── model_backends.yaml     # VLM endpoints & settings
│   ├── pipeline.yaml           # Pipeline orchestration
│   └── graph_settings.yaml     # LangGraph parameters
├── src/agentic/                # Core LangGraph pipeline
│   ├── graph.py                # StateGraph definition
│   ├── nodes/                  # Agent node implementations
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── summarizer.py
│   └── tool_registry.py        # IQA tool management
├── iqa_tools/                  # IQA model weights & metadata
├── data/                       # Datasets & manifests
├── scripts/                    # Evaluation & utilities
└── logs/                       # Execution logs (auto-generated)
```

### Testing Strategy

#### Evaluation Metrics
- **MCQ Accuracy**: Multiple-choice question correctness on AgenticIQA-Eval dataset
- **SRCC (Spearman)**: Rank correlation for scoring tasks
- **PLCC (Pearson)**: Linear correlation for scoring tasks

#### Evaluation Datasets
- **AgenticIQA-Eval**: MCQ tasks for Planner/Executor/Summarizer modules
- **TID2013, BID, AGIQA-3K**: SRCC/PLCC scoring evaluation

#### Testing Approach
- **Inference-only**: No training or fine-tuning involved
- **Batch processing**: Support checkpointing for resume capability
- **Result caching**: Cache tool execution by image hash
- **Environment validation**: `scripts/check_env.py` for dependency verification

#### Evaluation Commands
```bash
# MCQ accuracy evaluation
python scripts/eval_agenticqa_eval.py --input outputs/results.jsonl

# SRCC/PLCC calculation
python scripts/eval_srocc_plcc.py --input outputs/scores.jsonl

# Comprehensive report
python scripts/generate_report.py --output reports/report.md
```

### Git Workflow
- **Exclude memory bank files**: NEVER commit CLAUDE.md or CLAUDE-*.md files
- **Never delete memory bank files**: These files are critical project documentation
- Standard git workflow for code changes
- Keep memory bank files in sync with code changes manually (separate from commits)

## Domain Context

### Image Quality Assessment (IQA)
- **Full-Reference (FR)**: Compare distorted image against pristine reference
- **No-Reference (NR)**: Assess quality without reference image
- **Common Distortions**: Blur, noise, compression artifacts, over/underexposure, color shifts

### Vision-Language Models (VLMs)
- Multimodal models that understand both images and text
- Used for visual reasoning, distortion detection, and quality assessment
- Examples: GPT-4o, Claude 3.5, Qwen2.5-VL

### Traditional IQA Metrics
- **TOPIQ**: Transformer-based perceptual quality metric
- **QAlign**: Quality-aware alignment metric
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **DISTS**: Deep Image Structure and Texture Similarity
- **BRISQUE/NIQE**: Statistical NR-IQA metrics

### Agentic Reasoning
- Decompose complex tasks into specialized agent roles
- Each agent has specific responsibilities and expertise
- Agents communicate through structured state/messages
- Enables interpretable decision-making process

## Important Constraints

### Technical Constraints
- **Inference-only reproduction**: No training or fine-tuning involved
- **Pretrained models only**: Uses existing VLM APIs and IQA model weights
- **GPU requirements**: Local VLM inference requires CUDA-capable GPU
- **API rate limits**: Commercial VLM APIs have rate limiting
- **Score normalization**: All outputs must be normalized to 1-5 scale
- **Replanning limit**: Maximum 2 replanning iterations to prevent infinite loops

### Cost Management
- Cache tool execution results by image hash
- Optional API request/response caching
- Use cheaper models (GPT-4o-mini) for development/debugging
- Batch processing to minimize API calls

### Environment Requirements
- Python 3.10 with conda environment
- CUDA 12.1+ for GPU acceleration
- Environment variables: `AGENTIC_ROOT`, `AGENTIC_DATA_ROOT`, `AGENTIC_TOOL_HOME`, `AGENTIC_LOG_ROOT`
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (as needed)

## External Dependencies

### APIs & Services
- **OpenAI API** (GPT-4o, GPT-4o-mini) - Primary VLM backend
  - Configurable via `OPENAI_BASE_URL` for custom endpoints
- **Anthropic API** (Claude 3.5) - Alternative VLM backend
  - Configurable via `ANTHROPIC_BASE_URL`
- **Google AI API** (Gemini) - Alternative VLM backend
  - Configurable via `GOOGLE_API_BASE_URL`

### External Libraries
- **IQA-PyTorch** - Traditional IQA tools and metrics
  - GitHub: https://github.com/chaofengc/IQA-PyTorch
  - Install: `git clone` + `pip install -e .`
  - Requires model weights downloaded to `iqa_tools/weights/`

### Model Weights
- IQA model checkpoints stored in `iqa_tools/weights/`
- Tool metadata and logistic parameters in `iqa_tools/metadata/`
- Weights auto-downloaded on first use (IQA-PyTorch handles this)

### Alternative Models (Fallbacks)
- **VLM alternatives**: Claude 3.5, Qwen2.5-VL, GPT-4o-mini, MiniCPM-V
- **IQA tool fallbacks**: QAlign, BRISQUE, NIQE, LPIPS if primary tools unavailable
