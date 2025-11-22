# AgenticIQA

> **⚠️ 非官方复现** | This is an **unofficial reproduction** of the AgenticIQA paper.

AgenticIQA 是一个图像质量评估框架，结合视觉语言模型(VLM)和传统IQA工具，提供可解释的质量评估。

## 目录

- [快速开始](#快速开始)
- [运行方式](#运行方式)
  - [命令行批量处理](#1-命令行批量处理)
  - [REST API 服务](#2-rest-api-服务)
- [API 使用示例](#api-使用示例)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 安装依赖

```bash
# 创建 conda 环境
conda create -n agenticIQA python=3.10 -y
conda activate agenticIQA

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt

# 安装 IQA-PyTorch (提供传统IQA指标)
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch && pip install -e . && cd ..
```

### 2. 配置环境变量

**复制示例配置文件：**

```bash
cp .env.example .env
```

**编辑 `.env` 文件，填入你的配置：**

```bash
# 项目路径（通常不需要修改）
AGENTIC_ROOT=/your/path/to/agenticIQA
AGENTIC_DATA_ROOT=${AGENTIC_ROOT}/data
AGENTIC_TOOL_HOME=${AGENTIC_ROOT}/iqa_tools
AGENTIC_LOG_ROOT=${AGENTIC_ROOT}/logs

# API 密钥（至少配置一个）
OPENAI_API_KEY=sk-xxxx           # GPT-4o
ANTHROPIC_API_KEY=sk-ant-xxxx    # Claude 3.5
GOOGLE_API_KEY=xxxx              # Gemini
```

> 💡 **提示**：系统会自动从 `.env` 文件加载环境变量，无需手动 export。

### 3. 验证安装

```bash
# 检查环境配置
python scripts/check_env.py

# 验证 IQA 工具（首次运行会自动下载模型权重）
python scripts/verify_iqa_tools.py --type NR --skip-slow
```

---

## 运行方式

### 1. 命令行批量处理

适用于批量处理数据集，支持断点续传。

**准备输入数据** (`data/input.jsonl`)：

```jsonl
{"sample_id": "img001", "query": "这张图片的质量如何？", "image_path": "/path/to/image1.jpg"}
{"sample_id": "img002", "query": "图像有什么失真？", "image_path": "/path/to/image2.jpg"}
{"sample_id": "img003", "query": "与参考图相比质量如何？", "image_path": "/path/to/test.jpg", "reference_path": "/path/to/ref.jpg"}
```

**运行：**

```bash
# 基本用法
python run_pipeline.py \
  --input data/input.jsonl \
  --output results/output.jsonl

# 完整参数
python run_pipeline.py \
  --input data/input.jsonl \
  --output results/output.jsonl \
  --resume \                        # 断点续传
  --max-samples 100 \               # 限制处理数量
  --max-replan 2 \                  # 最大重规划次数
  --verbose                         # 详细日志

# 预览执行计划（不实际运行）
python run_pipeline.py \
  --input data/input.jsonl \
  --output results/output.jsonl \
  --dry-run
```

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--input, -i` | 输入 JSONL 文件路径（必需） |
| `--output, -o` | 输出 JSONL 文件路径（必需） |
| `--resume` | 跳过已处理的样本，从中断处继续 |
| `--max-samples, -n` | 限制处理的样本数量 |
| `--max-replan` | 最大重规划迭代次数（默认 2） |
| `--dry-run` | 验证配置，显示执行计划，不实际运行 |
| `--verbose, -v` | 启用详细日志 |

---

### 2. REST API 服务

适用于集成到其他系统或提供 Web 服务。

**启动服务：**

```bash
# 默认启动 (0.0.0.0:8000)
python scripts/run_api.py

# 指定端口
python scripts/run_api.py --port 9000

# 开发模式（热重载）
python scripts/run_api.py --reload

# 生产模式（多 worker）
python scripts/run_api.py --workers 4
```

**服务启动后：**

- API 文档：http://localhost:8000/docs (Swagger UI)
- 健康检查：http://localhost:8000/health

---

## API 使用示例

### 健康检查

```bash
curl http://localhost:8000/health
```

### 图像质量评估（通过文件路径）

```bash
# 无参考评估 (No-Reference)
curl -X POST http://localhost:8000/assess-path \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这张图片的质量如何？",
    "image_path": "/path/to/image.jpg"
  }'

# 有参考评估 (Full-Reference)
curl -X POST http://localhost:8000/assess-path \
  -H "Content-Type: application/json" \
  -d '{
    "query": "与参考图相比，图像质量如何？",
    "image_path": "/path/to/distorted.jpg",
    "reference_path": "/path/to/reference.jpg"
  }'
```

### 图像质量评估（通过文件上传）

```bash
# 无参考评估
curl -X POST http://localhost:8000/assess \
  -F "query=这张图片的质量如何？" \
  -F "image=@/path/to/image.jpg"

# 有参考评估
curl -X POST http://localhost:8000/assess \
  -F "query=与参考图相比质量如何？" \
  -F "image=@/path/to/distorted.jpg" \
  -F "reference=@/path/to/reference.jpg"
```

### Python 调用示例

```python
import requests

# 通过文件路径评估
response = requests.post(
    "http://localhost:8000/assess-path",
    json={
        "query": "这张图片的质量如何？",
        "image_path": "/path/to/image.jpg",
        "max_replan_iterations": 2
    }
)
result = response.json()
print(f"质量评分: {result['quality_score']}")
print(f"评估结论: {result['final_answer']}")
print(f"分析理由: {result['quality_reasoning']}")

# 通过文件上传评估
with open("/path/to/image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/assess",
        files={"image": f},
        data={"query": "图像有什么失真问题？"}
    )
```

### 响应格式

```json
{
  "final_answer": "图像质量良好，总体评分 4.2/5。",
  "quality_score": 4.2,
  "quality_reasoning": "图像清晰度较高，色彩还原准确...",
  "detected_distortions": ["轻微噪点", "边缘略有模糊"],
  "execution_metadata": {
    "execution_time_seconds": 3.5,
    "replan_count": 0,
    "tools_used": ["BRISQUE", "NIQE", "QAlign"]
  }
}
```

---

## 配置说明

### VLM 模型配置

编辑 `configs/model_backends.yaml`：

```yaml
planner:
  backend: openai.gpt-4o        # 可选: anthropic.claude-3.5-sonnet, google.gemini-pro
  temperature: 0.0

executor:
  backend: openai.gpt-4o
  temperature: 0.0

summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
```

### 运行时覆盖配置

```bash
# 临时使用更便宜的模型
python run_pipeline.py \
  --input data/test.jsonl \
  --output results/test.jsonl \
  --backend-override planner.backend=openai.gpt-4o-mini
```

---

## 常见问题

### API 密钥未配置

```
Error: No API keys found in environment
```

**解决**：检查 `.env` 文件是否正确配置了至少一个 API 密钥。

### IQA 工具权重下载失败

```
Error: Failed to download model weights
```

**解决**：
1. 检查网络连接
2. 手动下载权重到 `iqa_tools/weights/` 目录
3. 或跳过慢速工具：`--skip-slow`

### GPU 内存不足

```
CUDA out of memory
```

**解决**：
1. 使用 API 模型（GPT-4o, Claude）而非本地模型
2. 在 `configs/pipeline.yaml` 中设置 `gpu.enable: false`

### API 请求限流

**解决**：
1. 在 `configs/pipeline.yaml` 中启用缓存：`enable_cache: true`
2. 使用更便宜的模型：`openai.gpt-4o-mini`
3. 适当增加请求间隔

---

## 项目结构

```
agenticIQA/
├── .env.example          # 环境变量模板
├── configs/              # 配置文件
│   ├── model_backends.yaml   # VLM 模型配置
│   ├── pipeline.yaml         # 管道配置
│   └── api.yaml              # API 服务配置
├── src/agentic/          # 核心代码
│   ├── graph.py              # LangGraph 流程定义
│   ├── nodes/                # Planner, Executor, Summarizer
│   └── tool_registry.py      # IQA 工具注册
├── src/api/              # FastAPI 服务
├── scripts/              # 工具脚本
│   ├── run_api.py            # 启动 API 服务
│   ├── check_env.py          # 环境验证
│   └── verify_iqa_tools.py   # IQA 工具验证
├── run_pipeline.py       # 命令行批处理入口
└── requirements.txt      # 依赖列表
```

---

## 评估脚本

```bash
# 计算 SRCC/PLCC 相关系数
python scripts/eval_correlation.py --input results/output.jsonl

# MCQ 准确率评估
python scripts/eval_mcq_accuracy.py --input results/output.jsonl
```

---

## 文档

详细文档见 `docs/` 目录（中文）：
- [00_overview.md](docs/00_overview.md) - 系统概述
- [01_environment_setup.md](docs/01_environment_setup.md) - 环境配置

---

## License

参考原始论文和仓库的许可信息。
