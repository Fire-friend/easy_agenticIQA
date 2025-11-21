# AgenticIQA

> **⚠️ 非官方复现**
> This is an **unofficial reproduction** of the AgenticIQA paper.

AgenticIQA是一个图像质量评估框架，结合视觉-语言模型(VLM)和传统IQA工具，提供可解释的质量评估。

## 快速开始

### 1. 环境安装

```bash
# 创建环境
conda create -n agenticIQA python=3.10 -y
conda activate agenticIQA

# 安装PyTorch
pip install torch==2.3.0 torchvision==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 安装IQA-PyTorch
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch && pip install -e . && cd ..
```

### 2. 配置环境变量

```bash
# 设置项目路径
export AGENTIC_ROOT=$(pwd)
export AGENTIC_DATA_ROOT=${AGENTIC_ROOT}/data
export AGENTIC_TOOL_HOME=${AGENTIC_ROOT}/iqa_tools
export AGENTIC_LOG_ROOT=${AGENTIC_ROOT}/logs

# 配置API密钥（选择一个）
export OPENAI_API_KEY=<your_key>
# 或
export ANTHROPIC_API_KEY=<your_key>
```

### 3. 验证安装

```bash
python scripts/check_env.py
```

## 基本使用

### 运行评估

```bash
# 单个图像评估
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/sample.jsonl \
  --output outputs/results.jsonl
```

### 输入格式

输入文件为JSONL格式，每行一个样本：

```jsonl
{"query": "这张图片的质量如何？", "image_path": "path/to/image.jpg", "reference_path": null}
{"query": "与参考图相比质量如何？", "image_path": "path/to/test.jpg", "reference_path": "path/to/ref.jpg"}
```

**必需字段：**
- `query`: 关于图像质量的问题
- `image_path`: 待评估图像路径
- `reference_path`: 参考图像路径（可选，无参考评估时设为null）

### 配置VLM模型

编辑 `configs/model_backends.yaml`：

```yaml
planner:
  backend: openai.gpt-4o        # 可选: anthropic.claude-3.5-sonnet, qwen2.5-vl-local
  temperature: 0.0

executor:
  backend: openai.gpt-4o
  temperature: 0.0

summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
```

## 评估

```bash
# 计算SRCC/PLCC相关系数
python scripts/eval_srocc_plcc.py --input outputs/results.jsonl

# MCQ准确率评估
python scripts/eval_agenticqa_eval.py --input outputs/results.jsonl
```

## 验证IQA工具

```bash
# 验证所有工具是否正常
python scripts/verify_iqa_tools.py

# 只验证无参考(NR)工具，跳过慢速工具
python scripts/verify_iqa_tools.py --type NR --skip-slow
```

## 常见问题

**缺少依赖包？**
```bash
pip install -r requirements.txt
```

**API请求限流？**
- 启用缓存（`configs/pipeline.yaml` 中设置 `enable_cache: true`）
- 使用更便宜的模型（如 `openai.gpt-4o-mini`）

**GPU内存不足？**
- 使用API模型（GPT-4o, Claude）而不是本地模型
- 或禁用GPU（`configs/pipeline.yaml` 中设置 `gpu.enable: false`）

## 项目结构

```
agenticIQA/
├── configs/              # 配置文件
│   ├── model_backends.yaml    # VLM模型配置
│   └── pipeline.yaml          # 管道配置
├── src/agentic/          # 核心代码
│   ├── graph.py          # LangGraph流程定义
│   ├── nodes/            # Planner, Executor, Summarizer
│   └── tool_registry.py  # IQA工具注册
├── iqa_tools/            # IQA工具和权重
├── scripts/              # 实用脚本
│   ├── check_env.py      # 环境验证
│   └── verify_iqa_tools.py  # 工具验证
├── run_pipeline.py       # 主运行脚本
└── requirements.txt      # 依赖列表
```

## 文档

详细文档见 `docs/` 目录（中文）：
- [00_overview.md](docs/00_overview.md) - 系统概述
- [01_environment_setup.md](docs/01_environment_setup.md) - 环境配置
- [02-06](docs/) - 各模块详细说明

## License

参考原始论文和仓库的许可信息。

## Citation

```bibtex
@article{agenticIQA2024,
  title={AgenticIQA: Agentic Framework for Image Quality Assessment},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2024}
}
```
