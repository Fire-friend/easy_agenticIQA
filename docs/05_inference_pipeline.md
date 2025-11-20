# 推理管线与系统集成

本章描述如何将 Planner、Executor、Summarizer 串联成完整的 AgenticIQA 推理系统，涵盖架构设计、配置管理、伪代码实现、调试策略与运行示例。

## 1. 管线结构
```
load_config()
│
├─ init_backends()      # 解析模型/tool 配置
├─ load_metadata()      # 加载 tool metadata、logistic 参数
└─ for sample in dataset:
       plan = Planner.run(sample)
       evidence = Executor.run(sample, plan)
       result = Summarizer.run(sample, plan, evidence)
       if result.need_replan and retry < MAX_REPLAN:
           plan = Planner.run(sample, feedback=result)
           evidence = Executor.run(sample, plan)
           result = Summarizer.run(...)
       save_result(sample_id, plan, evidence, result)
```

## 2. 配置文件
推荐使用 YAML 管理模型后端、采样参数、tool 路径：
```yaml
# configs/pipeline.yaml
planner:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 512
executor:
  detection_backend: qwen2.5-vl-local
  analysis_backend: qwen2.5-vl-local
  selection_backend: openai.gpt-4o-mini
  tool_config: ${AGENTIC_TOOL_HOME}/metadata/tools.json
  logistic_params: ${AGENTIC_TOOL_HOME}/metadata/logistic.json
summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 512
pipeline:
  max_replan: 2
  cache_dir: ${AGENTIC_LOG_ROOT}/cache
  log_path: ${AGENTIC_LOG_ROOT}/pipeline.log
```

## 3. 核心组件设计
### 3.1 数据结构
```python
@dataclass
class Plan:
    query_type: str
    query_scope: Union[str, List[str]]
    distortion_source: str
    distortions: Optional[Dict[str, List[str]]]
    reference_mode: str
    required_tool: Optional[str]
    plan: Dict[str, bool]

@dataclass
class Evidence:
    distortion_set: Optional[Dict]
    distortion_analysis: Optional[Dict]
    selected_tools: Optional[Dict]
    quality_scores: Optional[Dict]
    tool_logs: List[Dict]

@dataclass
class Result:
    final_answer: str
    quality_reasoning: str
    need_replan: bool = False
```

### 3.2 Orchestrator 伪代码
```python
class AgenticPipeline:
    def __init__(self, cfg):
        self.planner = PlannerClient(cfg.planner)
        self.executor = ExecutorClient(cfg.executor)
        self.summarizer = SummarizerClient(cfg.summarizer)
        self.max_replan = cfg.pipeline.max_replan

    def process(self, sample):
        replan_count = 0
        feedback = None
        while True:
            plan = self.planner.generate(sample, feedback=feedback)
            evidence = self.executor.run(sample, plan)
            result = self.summarizer.generate(sample, plan, evidence)
            if not result.need_replan or replan_count >= self.max_replan:
                break
            replan_count += 1
            feedback = result
        return plan, evidence, result
```

## 4. 日志与缓存
- 建议使用 JSON Lines 日志，记录 `sample_id`、模型后端、耗时、token 使用量、是否触发重规划。
- tool 执行结果可缓存于 `{cache_dir}/{tool_name}/{image_hash}.json`，避免重复运行耗时指标。
- 对于 API 调用，提供请求/响应缓存（可选）以控制成本。

## 5. 错误处理与降级策略
- **模型超时/错误**：重试若干次（如 3 次），仍失败则切换备用模型或返回默认输出并在日志标记。
- **tool 执行失败**：记录异常，设置 `quality_scores` 对应项为 `null` 并附 `warning` 字段；Summarizer 需在解释中提示不确定性。
- **数据缺失**：检查 sample 是否存在参考图或 MOS，缺失则跳过 SRCC/PLCC 计算。

## 6. 批处理与并发
- 对 API 调用需控制并发（依据服务限制）。可使用队列或异步框架实现批量推理。
- tool 执行可按 distortion 类型并行（多线程/多进程），但需注意 GPU 显存与互斥资源。
- 建议实现断点续跑：管线每完成一个样本即写入结果文件，支持中断后继续。

## 7. 调试技巧
- 在小样本集（3~5 条）上启用 DEBUG 模式，打印 Planner/Executor/Summarizer 的完整 JSON。
- 使用 `TRACE_LEVEL` 环境变量控制日志粒度：`INFO` 保存概要，`DEBUG` 记录完整 Prompt 与响应（注意脱敏）。
- 对评分任务可绘制 tool 分数与最终得分，验证融合逻辑是否生效。

## 8. 运行示例
```bash
python run_pipeline.py \
  --config configs/pipeline.yaml \
  --input data/processed/agenticiqa_eval/planner.jsonl \
  --output outputs/agenticiqa_eval_results.jsonl
```
脚本应输出：
- `outputs/...jsonl`：包含 plan、evidence、result 的合并记录
- `logs/pipeline.log`：流水日志
- `artifacts/`：可选，保存中间可视化或 tool 输出

## 9. 集成测试
1. 准备测试数据（至少包含 FR/NR、质量评分、MCQ 三种问题）。
2. 运行管线并比较结果是否与预期 JSON fixture 一致。
3. 验证重规划流程（人为删除某项 evidence，检查 Summarizer 是否请求重新规划）。
4. 根据评测脚本计算准确率或相关系数，保证输出格式正确。

## 10. 扩展建议
- 构建 CLI/REST 接口，支持外部系统接入。
- 增加可视化模块，将 distortion 分析结果标注在图像上（虽非论文重点，但有助调试）。
- 记录成本与耗时统计，为 API 调用预算提供依据。
