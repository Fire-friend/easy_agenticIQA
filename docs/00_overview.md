# AgenticIQA 系统概览

## 背景与动机
传统 IQA 模型大多输出单一分值，缺乏策略自适应与人类可解释性；VLM 驱动的文本型 IQA 虽能给出说明，却难以保持精细评分稳定。AgenticIQA 通过“Planner–Executor–Summarizer”三角色协作，将 VLM 推理与经典 IQA tool 结合，构建可解释、可扩展的质量评估流程。本复现计划聚焦推理链路与评测逻辑，不涉及任何训练或微调。

## 核心思想
- **共享 VLM 内核**：三个角色（Planner、Executor、Summarizer）共享同一 VLM（如 GPT-4o、Qwen2.5-VL），通过不同 Prompt 执行角色专属任务。
- **任务分解**：Planner 解析用户查询与图像上下文，生成包含任务类型、范围、distortion 策略、tool 调用标志的 JSON 计划。
- **证据采集**：Executor 按计划逐步执行 distortion 识别、severity 分析、tool 选择与 tool 运行，统一输出结构化证据（`distortion_analysis`、`selected_tools`、`quality_scores` 等）。
- **证据融合**：Summarizer 汇总结构化证据与图像信息，输出最终答案与解释；在评分任务中使用加权融合公式结合工具分数与语言概率。
- **自我校正**：若证据不足，Summarizer 可触发 Planner 重新规划，形成闭环。

## 模块关系与数据流
```
用户输入(查询+图像+可选参考图)
        │
        ▼
Planner：生成任务计划 JSON（query_type、scope、distortion_source、distortions、reference_mode、control flags）
        │
        ▼
Executor：根据 control flags 依次执⾏
    ├─ Distortion Detection → distortion_set
    ├─ Distortion Analysis → distortion_analysis
    ├─ Tool Selection → selected_tools
    └─ Tool Execution → quality_scores(统一 1~5 分)
        │
        ▼
Summarizer：融合证据，输出 {final_answer, quality_reasoning}，必要时请求重规划
        │
        ▼
用户获得评分/解释或多项选择答案
```

## 模块职责摘要
- **Planner**：解析任务意图，决定是否需要侦测/分析/调用 tool，指派参考模式（Full-Reference / No-Reference），返回结构化计划供后续模块使用。
- **Executor**：在 Planner 的开关指导下完成证据采集，调用 IQA-PyTorch tool 并进行分值归一化，确保输出符合 Summarizer 需求。
- **Summarizer**：结合中间状态与 VLM 视觉理解，形成最终答案与质量推理，并执行评分融合算法。

## 项目结构
```
agenticIQA/
├── configs/
│   ├── model_backends.yaml        # 模型/服务端点与温度等设置
│   └── graph_settings.yaml        # LangGraph 节点参数（可选）
├── docs/
│   ├── 00_overview.md
│   └── 01_environment_setup.md    # 以及其余模块说明
├── scripts/
│   ├── check_env.py               # 环境自检脚本
│   └── launch_graph.py            # 启动 LangGraph Agent（示例）
├── src/
│   ├── agentic/
│   │   ├── graph.py               # LangGraph StateGraph 定义
│   │   ├── nodes/
│   │   │   ├── planner.py
│   │   │   ├── executor.py
│   │   │   └── summarizer.py
│   │   └── tool_registry.py
│   └── utils/                     # 通用工具函数、日志封装
├── iqa_tools/
│   └── weights/                   # 第三方 IQA 模型权重
├── data/
│   ├── raw/                       # 原始评测图像/题目
│   └── cache/                     # 中间结果、日志、prompt 缓存
├── logs/                          # LangGraph/工具运行日志（自动生成）
└── README.md
```
- `configs/` 与 `src/` 为核心管线所在；LangGraph 节点与状态模型集中于 `src/agentic/`。
- `iqa_tools/` 存放 IQA-PyTorch 等第三方工具及其权重，保持与环境变量 `AGENTIC_TOOL_HOME` 一致。
- `data/`、`logs/` 等目录会在运行时自动创建或由脚本初始化，可根据部署需求调整位置。

## 数据与评测逻辑
- **数据集**：AgenticIQA-Eval（Planner/Executor/Summarizer MCQ 任务），以及 TID2013、BID、AGIQA-3K（用于 SRCC/PLCC 评估）。
- **评测指标**：MCQ accuracy、SRCC、PLCC，以及必要的 qualitative case study（与论文 Figure 6–9 类似）。
- **执行原则**：仅使用预训练或 API 模型完成推理；若性能与论文存在差异，需要在 `06_evaluation_protocol.md` 中说明原因与替代方案。

## 复现成功标准
1. 三个模块的输入输出接口与论文描述一致，可独立运行并产生确定性 JSON 输出。
2. 管线 orchestrator 可复现 plan→execute→summarize 的闭环并保存中间信息。
3. 评测脚本能在目标数据集上生成 MCQ accuracy 与 SRCC/PLCC 表格，且记录与论文的偏差来源。
4. 文档中列出的替代模型/tool 在缺省场景下可运行，提供预期性能参考。
5. 关键脚本具备日志、缓存、错误处理，便于阶段化调试。
