# Executor 模块复现说明

Executor 在 Planner 输出的控制开关指导下，分阶段收集 distortion 证据与定量分值。四个子模块可独立调用，最终输出统一的中间表征 `Mt`，供 Summarizer 使用。本章提供接口约定、Prompt 模板、tool 接入、伪代码与测试示例。

## 1. 总体输入输出
- **输入**：Planner 生成的计划 JSON、图像（及可选参考图）、用户查询、tool metadata。
- **输出**：可能包含以下字段（由控制开关决定）：
  - `distortion_set`
  - `distortion_analysis`
  - `selected_tools`
  - `quality_scores`
 这些字段通常在 orchestrator 中封装为：
```json
{
  "distortion_set": {...} | null,
  "distortion_analysis": {...} | null,
  "selected_tools": {...} | null,
  "quality_scores": {...} | null,
  "tool_logs": [
    {"tool": "TOPIQ_FR", "params": {...}, "raw_score": 0.83, "normalized_score": 2.67}
  ]
}
```

## 2. Distortion Detection
### 作用
在 `distortion_source = "Inferred"` 时启用，识别潜在 distortion 类型，为分析与 tool 选择提供候选集合。

### Prompt 模板
```text
System:
You are an expert in distortion detection. Based on the user’s query, identify all possible distortions need to be
focused on to properly address the user’s intent.
Return a valid JSON object in the following format:
{
  "distortion_set": {
    "<object or Global>": ["<distortion_1>", "<distortion_2>", ...]
  }
}
Instructions:
1. Focus your analysis on query scope. Describe distortions for each individually.
2. Only include distortion types from the following valid categories: ["Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast"]

User:
User’s query: {query}
The image: <image>
```

### 输出校验
- 确保 distortion 类型属于限定列表。
- 对象名称与 Planner `query_scope` 对齐；缺失时回退为 `"Global"`。

## 3. Distortion Analysis
### 作用
估计 distortion severity 与视觉影响，输出简洁解释文本。

### Prompt 模板
```text
System:
You are a distortion analysis expert. Your task is to assess the severity and visual impact of various distortion
types for different regions of an image or the entire image.
The distortion information: {distortion_set}
Return a valid JSON object in the following format:
{
  "distortion_analysis": {
    "<object or Global>": [
      {
        "type": "<distortion>",
        "severity": "<none/slight/moderate/severe/extreme>",
        "explanation": "<brief visual explanation>"
      }
    ]
  }
}
Instructions:
1. Base your analysis on the listed distortion types and consider the user question.
2. Use "none" if a distortion is barely or not visible.
3. Keep explanations short and focused on visual quality. Focus solely on analyzing visual distortion effects.

User:
User’s query: {query}
The image: <image>
```

### 严重度标度
| 等级 | 解释 |
| --- | --- |
| none | 几乎无可见 distortion |
| slight | 轻微影响，需放大观察 |
| moderate | 明显影响核心细节，但仍可辨认 |
| severe | 显著损伤视觉质量 |
| extreme | 图像几乎无法用于任务 |

## 4. Tool Selection
### 作用
基于 distortion 类型和 tool 能力，分配最合适的 IQA tool；当 Planner 已指定 `required_tool` 时跳过。

### tool metadata 示例
```json
{
  "TOPIQ_FR": {
    "type": "FR",
    "strengths": ["Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast"]
  },
  "QAlign": {
    "type": "NR",
    "strengths": ["Blurs", "Color distortions", "Noise", "Brightness change", "Spatial distortions", "Sharpness"]
  },
  "LPIPS": {"type": "FR", "strengths": []}
}
```

### Prompt 模板
```text
System:
You are a tool executor. Your task is to assign the most appropriate IQA tool to each visual distortion type,
based on the descriptions of the tools.
The distortion information: {distortion_set}.
The available tools: {tool_description}.
Return a valid JSON object in the following format:
{
  "selected_tools": {
    "<object or Global>": {
      "<distortion>": "<tool_name>"
    }
  }
}
Instructions:
For each distortion, choose the tool whose description suggests it performs best for that type of distortion.
```

### 逻辑建议
- 按 FR/NR 匹配：若 `reference_mode="Full-Reference"` 优先选择 FR tool。
- 若 tool 库无明显匹配，则选通用 tool（如 QAlign、TOPIQ）。
- 支持同一 tool 对应多个 distortion，减少外部调用次数。

## 5. Tool Execution
### 作用
调用已选 tool 对图像进行评估，输出原始分数与统一映射后的分值。

### 流程伪代码
```python
for object_name, distortions in selected_tools.items():
    for distortion, tool_name in distortions.items():
        raw_score = run_tool(tool_name, image, reference)
        normalized = logistic_map(tool_name, raw_score)
        record(object_name, distortion, tool_name, raw_score, normalized)
```

### Logistic 归一化
采用五参数单调逻辑函数：
```
f(x) = (β1 - β2) / (1 + exp(-(x - β3)/|β4|)) + β2
```
参数可参考论文附录 A.3 或根据 tool 官方提供的标定数据，最终分数映射到 `[1, 5]`，数值越大质量越高。无参数时可使用经验值或线性缩放，并在评测文档中注明。

### tool 调用注意事项
- 确保输入图像尺寸与 tool 要求匹配；必要时使用 `opencv` 做大小/色彩空间转换。
- 统一 tool 输出方向（部分 tool 输出误差，需取反或倒数）。
- 建议对 tool 调用过程进行缓存（键：图像哈希、tool 名、参考哈希）。

## 6. 子模块间数据流
```
distortion_set --(补充 distortion 信息)--> distortion_analysis
distortion_analysis + tool_metadata --> selected_tools
selected_tools + 图像 --> quality_scores
```
缺失某环节时，后续模块需根据 Planner 控制位判断是否继续执行。

## 7. 备用策略
- **tool 缺失/报错**：记录日志，回退到可用的通用 NR tool（如 NIQE、BRISQUE）；保持 `quality_scores` 字段存在但注明 `fallback=true`。
- **VLM 延迟**：对 Detection/Analysis/Selection 设置超时（例如 30s），超时则切换备用模型或返回默认输出。
- **数值异常**：若 tool 输出 NaN/Inf，丢弃该项并标记警告；Summarizer 在评分时需忽略。

## 8. 测试示例
```json
Planner 输出:
{
  "query_scope": ["vehicle"],
  "distortion_source": "Explicit",
  "distortions": {"vehicle": ["Blurs"]},
  "plan": {
    "distortion_detection": false,
    "distortion_analysis": true,
    "tool_selection": true,
    "tool_execution": true
  },
  "reference_mode": "No-Reference"
}

Executor 期望输出:
{
  "distortion_set": null,
  "distortion_analysis": {
    "vehicle": [
      {
        "type": "Blurs",
        "severity": "severe",
        "explanation": "The vehicle edges appear smeared with strong motion blur."
      }
    ]
  },
  "selected_tools": {
    "vehicle": {
      "Blurs": "QAlign"
    }
  },
  "quality_scores": {
    "vehicle": {
      "Blurs": ["QAlign", 2.15]
    }
  }
}
```
建议编写回归测试，以固定图像与 tool 输出模拟 Executor 的整个链路。

## 9. 实施步骤
1. 整理 tool metadata 与 logistic 参数。
2. 编写四个子模块函数/类，并在 orchestrator 中按控制开关串联。
3. 实现缓存及日志记录（包括 tool 耗时、模型调用次数、异常堆栈）。
4. 准备包含图像样例与预期输出的测试夹具，保证推理可重复。
