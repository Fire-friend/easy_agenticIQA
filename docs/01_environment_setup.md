# 环境与依赖配置

## 1. 系统要求
- **操作系统**：推荐 Ubuntu 22.04（其他 Linux/macOS 亦可，需注意 CUDA/驱动兼容）
- **硬件**：  
  - GPU：NVIDIA RTX 3090 或同级别，显存 ≥24GB（可使用 API 时降级为 CPU，速度较慢）  
  - CPU：8 核以上  
  - 内存：≥32GB  
  - 磁盘：≥200GB（存储数据集、日志、缓存、tool 权重）
- **网络**：可访问模型 API、GitHub、数据集下载站点

## 2. Python 与虚拟环境
```bash
# 已存在环境（如 agenticiqa）时
conda activate agenticIQA

# 尚未创建时
conda create -n agenticIQA python=3.10 -y
conda activate agenticIQA

# 若使用 pipenv/venv，可自行替换
```

## 3. 基础依赖
```bash
pip install torch==2.3.0 torchvision==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.42.0 accelerate==0.31.0 bitsandbytes==0.43.1
pip install "openai>=2.0.0,<3.0.0" "anthropic>=0.72.0,<1.0.0" google-genai "httpx>=0.28.1,<1.0.0"  # 更新至兼容版本
pip install qwen-vl-utils==0.0.8  # 若本地加载 Qwen2.5-VL
pip install opencv-python pillow numpy scipy scikit-image einops tqdm pyyaml
pip install pydantic==2.7.1 typer rich loguru
pip install langgraph "langchain-core>=1.0.0,<2.0.0" "langchain-openai>=1.0.0,<2.0.0" "langchain-anthropic>=1.0.0,<2.0.0"  # LangGraph agent 框架
```

> **提示**：确保在已激活的 `agenticIQA` 环境中执行上述安装命令；根据实际 GPU 驱动选择对应的 CUDA 版 torch，若仅调用 API 可安装 CPU 版。

> **LangGraph 使用建议**：LangGraph 通过图结构定义 agent 流程，建议在 `src/agentic/graph.py` 中维护节点与边，并使用 `langgraph.cli` 帮助调试；示例参考 `docs/05_inference_pipeline.md`。

## 4. IQA tool 与资源
1. 克隆并安装 IQA-PyTorch（或直接使用其 pip 包）：
   ```bash
   git clone https://github.com/chaofengc/IQA-PyTorch.git
   cd IQA-PyTorch
   pip install -e .
   ```
2. 下载所需 tool 权重（TOPIQ、QAlign、AHIQ、LPIPS、DISTS、WaDIQaM 等），统一放置于：
   ```
   ${AGENTIC_TOOL_HOME}/weights/<tool_name>/
   ```
3. 准备 tool metadata JSON（样例在 `03_module_executor.md` 提供），用于 Tool Selection。

## 5. Agent 框架（LangGraph）
- 使用 LangGraph 定义 `Planner → Executor → Summarizer` 的编排图，推荐在 `src/agentic/graph.py` 中集中维护节点与边。
- 可通过 `langgraph dev` 启动本地可视化调试，或使用 `langgraph deploy`（若接入 LangSmith/LangServe）发布服务。
- Graph 节点内可直接复用现有模块：例如 Planner 节点调用 `planner.run(plan_prompt)`，Executor 节点封装 tool 调用逻辑，Summarizer 节点汇总上下文。
- 若需要追踪对话状态，可启用 LangGraph Memory（`StateGraph`），并在配置中绑定 Redis/Postgres backend。

## 6. 模型接入方式
- **云端 API**：配置 OpenAI、Anthropic、Google API Key，设置环境变量：
```bash
export OPENAI_API_KEY=<key>
export ANTHROPIC_API_KEY=<key>
export GOOGLE_API_KEY=<key>
# 如需通过代理或私有部署访问，允许覆盖 Base URL
export OPENAI_BASE_URL=https://your-openai-endpoint/v1        # 可选
export ANTHROPIC_BASE_URL=https://your-anthropic-endpoint     # 可选
export GOOGLE_API_BASE_URL=https://your-google-endpoint       # 可选
```
- **本地大模型**：下载 Qwen2.5-VL/Qwen2.5-VL*、MiniCPM-V 等权重，使用 `transformers.AutoModelForCausalLM` 或官方推理 SDK 加载。
- **配置文件模板**（`configs/model_backends.yaml`）：
  ```yaml
  planner:
    backend: openai.gpt-4o
    temperature: 0.0
  executor:
    backend: qwen2.5-vl-local
    temperature: 0.0
  summarizer:
    backend: openai.gpt-4o
    temperature: 0.0
  ```
  > **Base URL 提示**：若后端支持自定义 endpoint，可在配置中引用 `OPENAI_BASE_URL` 等环境变量，或在 `backend` 部分改写为自建服务的注册名。

## 7. 环境变量与目录约定
| 变量 | 作用 | 示例 |
| --- | --- | --- |
| `AGENTIC_ROOT` | 项目根目录 | `/data/wujiawei/Agent/agenticIQA` |
| `AGENTIC_DATA_ROOT` | 数据主目录 | `${AGENTIC_ROOT}/data` |
| `AGENTIC_TOOL_HOME` | IQA tool 权重与 metadata | `${AGENTIC_ROOT}/iqa_tools` |
| `AGENTIC_LOG_ROOT` | 日志与缓存 | `${AGENTIC_ROOT}/logs` |
| `VLM_BACKEND` | 默认大模型后端 | `openai.gpt-4o` |
| `OPENAI_BASE_URL` | OpenAI API Base URL（覆盖默认 `api.openai.com`） | `https://your-openai-endpoint/v1` |
| `ANTHROPIC_BASE_URL` | Anthropic API Base URL | `https://your-anthropic-endpoint` |
| `GOOGLE_API_BASE_URL` | Google API Base URL | `https://your-google-endpoint` |
| `LANGGRAPH_STORAGE` | LangGraph 状态存储（可选） | `redis://localhost:6379/0` |

## 8. 自检脚本
创建 `scripts/check_env.py`，检查 GPU、依赖版本与模型连通性：
```python
import os
import torch
import importlib

REQ_PKGS = ["torch", "transformers", "opencv", "iqa_pytorch", "langgraph"]

def main():
    print("Python:", os.sys.version)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    for pkg in REQ_PKGS:
        try:
            module = importlib.import_module(pkg)
            print(f"{pkg} version:", getattr(module, "__version__", "unknown"))
        except ImportError:
            print(f"[ERROR] Missing package: {pkg}")
    print("AGENTIC_TOOL_HOME:", os.getenv("AGENTIC_TOOL_HOME", "NOT SET"))
    # 可选：执行一次 Planner Prompt 并捕获返回

if __name__ == "__main__":
    main()
```
运行：
```bash
python scripts/check_env.py
```
确认输出无错误后，再进入后续阶段。

## 9. 注意事项
- 若调用 API 受限，建议先行缓存 Prompt 与返回结果，并在评测阶段重放缓存。
- tool 执行可能需要 CUDA/cuDNN 支持，请确保驱动与 PyTorch 版本匹配。
- 建议使用 `requirements.txt` 或 `environment.yml` 固化依赖，便于团队协作与 CI 执行。
