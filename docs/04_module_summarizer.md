# Summarizer 模块复现说明

Summarizer 负责整合 Executor 的证据与图像内容，输出最终回答及质量解释。在评分任务中，需要结合 IQA tool 分数与语言模型推理结果进行加权融合；若判定信息不足，还要触发重新规划。本章描述接口、Prompt、融合算法、反思机制与测试指南。

## 1. 输入输出接口
- **输入**：
  - Executor 产生的 `distortion_analysis`、`selected_tools`、`quality_scores`
  - 图片/参考图（必要时直接观察）
  - 用户原始查询
  - Planner/Executor 运行日志（可选）
- **输出 JSON**：
```json
{
  "final_answer": "<多选题选项或质量分级/分值>",
  "quality_reasoning": "<基于证据的简要解释>",
  "need_replan": false | true
}
```
其中 `need_replan=true` 时，Orchestrator 应返回 Planner 重新生成计划。

## 2. Prompt 模板
### 解释/问答模式
```text
System:
You are a visual quality assessment assistant. Your task is to select the most appropriate answer to the user’s
question. You are given:
- Distortion analysis (severity and visual impact of listed distortions)
- Tool response (overall quality scores from IQA models)
- Image content
Decision process
1. First, understand what kind of visual information is needed to answer the user’s question.
2. Check if the provided distortion analysis or tool response already contains the required information.
3. If the provided information is sufficient, use it to answer.
4. If the information is unclear or insufficient, analyze the image directly to determine the best answer.
Return a valid JSON object in the following format:
{
  "final_answer": "<one of the above letters>",
  "quality_reasoning": "<brief explanation, based on either distortion analysis, tool response, or direct visual observation>"
}
```

### 评分模式
```text
System:
You are a visual quality assessment assistant. Given the question and the analysis (tool scores, distortion
analysis). Your task is to assess the image quality.
You must select one single answer from the following:
A. Excellent
B. Good
C. Fair
D. Poor
E. Bad
Return the JSON:
{
  "final_answer": "<one letter>",
  "quality_reasoning": "<concise justification referencing distortions or tool scores>"
}
```
在具体实现中，可根据任务类型动态切换 Prompt；需要将 `distortion_analysis` 与 `quality_scores` 以 JSON 或表格形式嵌入 `User` 消息。

## 3. 分数融合算法
当任务要求返回数值或离散质量等级时，遵循论文提出的融合策略：
1. 收集 tool 输出 `{q̂_i}`（均在 `[1,5]` 范围内，数值越大质量越好）。
2. 计算平均值 `q̄ = (1/n) Σ q̂_i`。
3. 构建离散质量级别集合 `C = {1, 2, 3, 4, 5}`，定义感知权重：
   ```
   α_c = exp(-η(q̄ - c)^2) / Σ_j exp(-η(q̄ - j)^2),  其中 η=1
   ```
4. Summarizer 通过 Prompt 推理得到各质量级别的 logit（或直接返回分类概率）`log p̂_c`，转化为概率：
   ```
   p_c = exp(log p̂_c) / Σ_j exp(log p̂_j)
   ```
5. 最终得分：
   ```
   q = Σ_c α_c · p_c · c
   ```
6. 若需要输出离散标签，可根据 `q` 与阈值映射，或直接使用概率最大的类别。

> 若替换模型无法方便地获取 logits，可令 VLM 直接输出离散等级，然后结合 tool 均值进行后处理，如线性插值或加权投票，并在评测报告中说明差异。

## 4. 反思与重规划
- Summarizer 应检查 `distortion_analysis`、`quality_scores` 是否覆盖用户问题；若关键信息缺失或矛盾，可设置 `need_replan=true` 并说明原因（例：“缺少针对 vehicle 区域的 tool 分数”）。
- Orchestrator 接收后重新调用 Planner（附带 Summarizer 的反馈），形成自校正循环。
- 建议限制重规划次数（如最多 2 次），避免无终止循环。

## 5. 模型推荐
| 模型 | 优点 | 注意事项 |
| --- | --- | --- |
| GPT-4o | 推理能力强，解释自然 | 成本高，需缓存结果 |
| Claude 3.5 Sonnet | 解释性良好 | 对图像输入需 Base64/URL |
| Qwen2.5-VL | 支持本地部署 | 中文输出更自然，但未微调时准确率可能略低 |
| GPT-4o-mini / GPT-4o-mini-high | 成本较低，可用于批量评测 | 需验证输出稳定性 |

## 6. 日志与可观察性
- 输出字段：`final_answer`、`quality_reasoning`、`used_evidence`（可选，指明使用的 tool 或分析条目 ID）、`need_replan`。
- 统计指标：平均响应时长、token 消耗、重规划次数。
- 对 MCQ 任务可记录预测选项与 GT，对评分任务记录 `q` 与 MOS 的差异，方便后续评估。

## 7. 测试示例
```json
输入:
{
  "user_query": "Rate the perceptual quality of this image.",
  "distortion_analysis": {
    "Global": [
      {"type": "Blurs", "severity": "moderate", "explanation": "Edges appear soft."},
      {"type": "Brightness change", "severity": "slight", "explanation": "Image slightly bright."}
    ]
  },
  "quality_scores": {
    "Global": {
      "Blurs": ["TOPIQ_FR", 2.6],
      "Brightness change": ["TOPIQ_FR", 2.8]
    }
  }
}

期望输出:
{
  "final_answer": "C",
  "quality_reasoning": "tool 分数集中在 2.6~2.8，对应一般质量；分析显示模糊与轻微亮度提升导致细节缺失。",
  "need_replan": false
}
```
可通过断言 `final_answer` 与 `quality_reasoning` 中关键句子，验证 Summarizer 对证据的引用。

## 8. 实施步骤
1. 实现 `summarize_response(evidence)` 函数：整合输入、渲染 Prompt、调用模型。
2. 编写分数融合工具类（utility），支持直接计算 `q` 并映射到等级。
3. 在 orchestrator 中监控 `need_replan` 并触发 Planner 重新调用。
4. 准备多个场景测试（MCQ、描述性、评分），确保输出稳定。
