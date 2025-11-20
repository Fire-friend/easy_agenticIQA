# Planner 模块复现说明

## 1. 角色职责
Planner 负责解析用户查询、图像上下文与可选参考图像，输出结构化计划 JSON，指导后续 Executor 的执行范围、tool 调用策略与 Summarizer 的输入。其核心目标是：
- 确认任务类型（IQA or Other）与参考模式（Full-Reference / No-Reference）
- 明确关注对象（Global 或对象列表）
- 判断 distortion 来源（Explicit / Inferred）
- 指定候选 distortion 类型及是否需要侦测/分析/tool 调用/tool 执行

## 2. 输入与输出接口
### 输入
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `image_path` 或图像对象 | str / PIL.Image | 必选，待评估图像 |
| `reference_path` | str / None | 可选，FR-IQA 场景提供参考图像 |
| `user_query` | str | 必选，用户自然语言问题/指令 |
| `prior_context` | dict / None | 可选，继续对话时的历史信息 |

### 输出 JSON Schema
```json
{
  "query_type": "IQA" | "Other",
  "query_scope": ["<object>", ...] | "Global",
  "distortion_source": "Explicit" | "Inferred",
  "distortions": {
    "<object or Global>": ["Blur", "Noise", ...]
  } | null,
  "reference_mode": "Full-Reference" | "No-Reference",
  "required_tool": "<tool_name>" | null,
  "plan": {
    "distortion_detection": true | false,
    "distortion_analysis": true | false,
    "tool_selection": true | false,
    "tool_execution": true | false
  }
}
```

## 3. Prompt 模板
按照论文附录 A.2，推荐直接复用以下模板（保持英文以利模型对齐）：
```text
System:
You are a planner in an image quality assessment (IQA) system. Your task is to analyze the user’s query and
generate a structured plan for downstream assessment.
Return a valid JSON object in the following format:
{
  "query_type": "IQA" or "Other",
  "query_scope": ["<object1>", "<object2>", ...] or "Global",
  "distortion_source": "Explicit" or "Inferred",
  "distortions": dict or null,
  "reference_mode": "Full-Reference" or "No-reference",
  "required_tool": "<tool_name>" or null,
  "plan": {
    "distortion_detection": true or false,
    "distortion_analysis": true or false,
    "tool_selection": true or false,
    "tool_execution": true or false
  }
}

User:
User’s query: {query}
The image: <image>
```
可按需追加对话上下文（如历史回答），但需保持输出结构稳定。推理时建议 `temperature=0`、`top_p=0.1`，并设置 `max_tokens` 足够覆盖完整 JSON。

## 4. 决策逻辑建议
1. **distortion 来源判断**  
   - 用户明确提及 distortion（如 “Is the vehicle blurry?”）→ `distortion_source = "Explicit"`，`distortions` 直接填入。
   - 未指定或开放式问题 → `distortion_source = "Inferred"`，需要后续侦测。
2. **控制开关**  
   - `distortion_detection = true` 当 `distortion_source = "Inferred"`；
   - `distortion_analysis = true` 只要需要解释或 severity 信息；
   - `tool_selection/tool_execution = true` 在需要定量评分或 tool 支持时；对于纯描述类问题可设为 `false`。
3. **参考模式**  
   - 若提供参考图 → `Full-Reference`；否则 `No-Reference`。
4. **作用范围**  
   - 查询指向具体对象时填写对象数组；否则使用 `"Global"`。
5. **required_tool**  
   - 用户指定某 tool 或任务明确（如“use LPIPS”）时填入；否则 `null`，由 Tool Selection 处理。

## 5. 异常处理与备用策略
- **模型拒答/输出不合法 JSON**：启用重试（更换提示词或添加“strict JSON output”指令）；超过阈值后切换备用模型。
- **字段缺失**：使用 Pydantic/JSON Schema 校验，若缺失则回退到默认值（例如 `distortion_source="Inferred"`）。
- **多轮对话**：可将历史 `plan` 与 Summarizer 反馈追加至 Prompt，促使 Planner 更新策略。

## 6. 推荐模型与配置
| 模型 | 渠道 | 备注 |
| --- | --- | --- |
| GPT-4o | OpenAI API | 最佳准确率，但成本较高 |
| Claude 3.5 Sonnet | Anthropic API | 语言理解强，响应稳定 |
| Qwen2.5-VL (原版) | 本地/阿里云 API | 可离线部署，需充足 GPU |
| MiniCPM-V 2.6 | 本地 | 模型体量小，适合边缘设备 |

> 若无 Qwen2.5-VL* 微调模型，可直接使用开源版本或商业 API，评估差异需记录在评测文档。

## 7. 测试示例
```json
输入:
{
  "user_query": "Is the vehicle blurry in this picture?",
  "image_path": "samples/car_blur.png",
  "reference_path": null
}

期望 Planner 输出(示例):
{
  "query_type": "IQA",
  "query_scope": ["vehicle"],
  "distortion_source": "Explicit",
  "distortions": {
    "vehicle": ["Blurs"]
  },
  "reference_mode": "No-Reference",
  "required_tool": null,
  "plan": {
    "distortion_detection": false,
    "distortion_analysis": true,
    "tool_selection": false,
    "tool_execution": false
  }
}
```
建议在单元测试中对 JSON 字段、布尔开关与候选 distortion 列表进行断言，确保 Planner 输出稳定。

## 8. 日志与监控
- 记录字段：请求 ID、输入摘要、模型后端、采样参数、token 消耗、运行时长、输出 JSON。
- 建议使用结构化日志（JSON Lines），便于后续评测与溯源；敏感信息（API Key）需脱敏。

## 9. 复现阶段任务
1. 按本章配置 Prompt 与模型参数，编写 `planner.py` 并封装 `generate_plan(query, image, reference=None)`。
2. 构建单元测试（本地 JSON fixture），确保固定输入生成稳定输出。
3. 将 Planner 输出示例存放于 `artifacts/planner/`，作为后续阶段的输入基准。
