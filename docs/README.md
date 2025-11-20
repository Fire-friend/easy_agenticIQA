# AgenticIQA 复现文档导航

## 目标与范围
仅基于论文《AgenticIQA》公开信息，复现其推理流程与实验逻辑，不涉及任何模型训练、微调或参数更新。所有模块均依赖现成大模型与 IQA tool 完成推理。

## 文档索引
- `00_overview.md`：系统背景、角色协同、数据流与成功标准
- `01_environment_setup.md`：环境搭建、依赖安装、配置校验
- `02_module_planner.md`：Planner 模块接口、Prompt、测试说明
- `03_module_executor.md`：Executor 子模块、tool 接入、示例流程
- `04_module_summarizer.md`：Summarizer 推理逻辑、评分融合、反思机制
- `05_inference_pipeline.md`：端到端管线设计、配置与调试
- `06_evaluation_protocol.md`：评测数据、指标计算、结果对齐

## 阶段划分
| 阶段 | 前置条件 | 产出 | 验收要点 |
| --- | --- | --- | --- |
| 1. 环境准备 | 获取 GPU/CPU 资源、API Key、tool 权重 | 已安装依赖与配置文件 | 自检脚本通过，依赖版本与硬件符合要求 |
| 2. 数据整理 | 环境可运行 | 统一的数据目录与 manifest | 数据数量、MOS、MCQ 标签校验无误 |
| 3. 模块实现 | 阶段 1+2 完成 | Planner/Executor/Summarizer 独立脚本与回归样例 | 固定输入得到确定 JSON 输出 |
| 4. 管线集成 | 模块可独立运行 | Orchestrator、日志、错误处理 | 提供端到端示例，保留中间产物 |
| 5. 评测复现 | 已完成阶段 4 | 指标计算脚本及报告 | MCQ 准确率、SRCC/PLCC 表格可生成 |

## 常见问题
- **为何不包含微调步骤？** 复现范围仅限推理；可使用公开 API 或开源大模型获得 Planner/Executor/Summarizer 能力。
- **若缺少专用模型？** 文档在各模块中列出替代模型/tool（如 GPT-4o、Claude 3.5、Qwen2.5-VL 原版、MiniCPM-V），以及性能预期差异记录方法。
- **难以获得 IQA tool 权重？** 推荐首选 IQA-PyTorch 官方仓库下载，若部分 tool 缺失，可 fallback 到 QAlign、LPIPS 等通用指标。
- **API 费用或速率限制？** 阶段 5 的评测脚本支持分批执行并缓存中间结果，降低重复调用成本。

## 快速开始
1. 依照 `01_environment_setup.md` 新建环境并运行验证脚本。
2. 下载数据并按 `02` 中的目录组织，生成 manifest。
3. 阅读 `02`~`04`，配置 Prompt 模板与模型访问。
4. 根据 `05` 运行端到端管线，在小样本上验证。
5. 使用 `06` 的命令完成评测并生成复现报告。
