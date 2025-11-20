# 评测协议与结果复现

本章提供 AgenticIQA 评估流程的完整指南，包括数据准备、指标计算、替代模型策略、结果记录与报告模板。目标是在“仅推理”前提下尽可能复刻论文表格与质性案例。

## 1. 数据准备
| 数据集 | 用途 | 获取方式 | 备注 |
| --- | --- | --- | --- |
| AgenticIQA-Eval | Planner/Executor/Summarizer MCQ 测试 | 官方发布或附录链接 | 750 NR + 250 FR 样本，三条子任务划分 |
| TID2013 | FR-IQA 评分 | http://www.ponomarenko.info/tid2013.htm | 含 24 种 distortion，提供 MOS |
| BID | NR-IQA 评分 | http://vision.okstate.edu/?loc=bilevel | 586 张真实 distortion 图，带 MOS |
| AGIQA-3K | NR-IQA 评分 | 官方仓库 | 生成式 distortion 图像 |
| （可选）LLVisionQA / Q-Bench | 语言问答型评测 | 公开仓库 | 验证泛化能力 |

### 1.1 目录结构
```
data/
  raw/
    agenticiqa_eval/
    tid2013/
    bid/
    agiqa-3k/
  processed/
    agenticiqa_eval/
      planner.jsonl
      executor_distortion.jsonl
      executor_tool.jsonl
      summarizer.jsonl
    tid2013/
      images/
      reference/
      mos.csv
      manifest.jsonl
    ...
metadata/
  tid2013_manifest_schema.json
  agenticiqa_eval_schema.json
```

### 1.2 Manifest Schema
以 TID2013 为例：
```json
{
  "sample_id": "tid2013_0001",
  "dataset": "tid2013",
  "distorted_path": "data/raw/tid2013/distorted/I01_01_1.bmp",
  "reference_path": "data/raw/tid2013/reference/I01.bmp",
  "mos": 5.432,
  "split": "test",
  "metadata": {
    "distortion_type": "JPEG",
    "level": 1
  }
}
```
AgenticIQA-Eval manifest 需包含 `question`, `options`, `answer`, `task_type`（planner/executor_distortion/executor_tool/summarizer）, `reference_mode` 等字段。

## 2. AgenticIQA-Eval 评估
### 2.1 MCQ 推理
1. 使用 `run_pipeline.py` 生成每个问题的 `final_answer`。
2. 将结果与 `answer` 对比，按任务类型分别计算准确率：
```python
import pandas as pd

df = pd.read_json("outputs/agenticiqa_eval_results.jsonl", lines=True)
for task, group in df.groupby("task_type"):
    acc = (group["pred"] == group["answer"]).mean()
    print(task, acc)
```
3. 统计整体准确率及每类问题（What/How/YesNo）的表现。

### 2.2 误差分析
- 输出混淆矩阵，识别常见错误类别（如 Planner 未正确识别对象）。
- 记录触发重规划的比例，分析是否与准确率下降相关。

## 3. SRCC/PLCC 评分评估
### 3.1 推理流程
1. 对 TID2013/BID/AGIQA-3K：
   ```bash
   python run_pipeline.py --config configs/pipeline.yaml \
     --input data/processed/tid2013/manifest.jsonl \
     --output outputs/tid2013_scores.jsonl
   ```
2. 结果文件需包含 `sample_id`, `final_score`（若输出离散等级可映射为 1~5）以及 tool 分数。

### 3.2 指标计算
```python
import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr

df = pd.read_json("outputs/tid2013_scores.jsonl", lines=True)
mos = df["mos"]
pred = df["final_score"]
srcc, _ = spearmanr(pred, mos)
plcc, _ = pearsonr(pred, mos)
print("SRCC:", srcc, "PLCC:", plcc)
```
可使用 bootstrap（如 1,000 次）计算置信区间，并在报告中说明：
```python
import numpy as np

def bootstrap_metric(pred, mos, metric_fn, num=1000):
    stats = []
    n = len(pred)
    for _ in range(num):
        idx = np.random.randint(0, n, n)
        stats.append(metric_fn(pred[idx], mos[idx])[0])
    return np.percentile(stats, [2.5, 97.5])
```

### 3.3 结果记录
- 创建对照表，列出每个数据集的 SRCC/PLCC，与论文提供的数值并列（若使用不同模型需标注）。
- 若性能明显低于论文，需分析原因（如未使用微调模型、tool 库不同、推理温度差异）。

## 4. 质性案例
- 选取代表性样本（成功+失败），保存 Planner/Executor/Summarizer 的 JSON 输出及解释文本。
- 可生成图像描述图（可选），展示 distortion 定位和 tool 得分，参考论文 Figure 6–9。
- 在报告中说明与论文示例的差异（例如使用的 tool 不同、解释语言风格差异）。

## 5. 替代模型策略
| 场景 | 推荐替代 | 预期影响 |
| --- | --- | --- |
| 无 Qwen2.5-VL* 权重 | 使用 GPT-4o/Claude 3.5 或 Qwen2.5-VL 原版 | 规划与解释准确率可能下降，需要记录 |
| tool 缺失 | 使用 QAlign、BRISQUE、NIQE | 分数范围与论文不同，需重新标定 |
| API 成本受限 | 使用 GPT-4o-mini 或本地 MiniCPM-V | 需要扩大重试次数，评估准确率 |

在最终报告中应明确说明所用替代项及其带来的指标偏差。

## 6. 报告模板
可在 `reports/replication_report.md` 生成如下结构：
```
1. 环境与版本
2. 数据集状态（数量、缺失情况）
3. 模型与 tool 配置
4. AgenticIQA-Eval 准确率表
5. TID2013/BID/AGIQA-3K SRCC/PLCC 表格
6. 指标差异分析（相对于论文）
7. 质性案例（图文说明）
8. 成本与耗时统计
9. 已知风险与后续改进建议
```

## 7. 自动化脚本
- `scripts/eval_agenticqa_eval.py`：输入管线输出 JSON，生成准确率与混淆矩阵。
- `scripts/eval_srocc_plcc.py`：读取多个数据集的预测，输出指标表和图形（散点图、趋势图）。
- `scripts/generate_report.py`：整合数值与案例，生成 Markdown 报告。

## 8. 风险与注意事项
- 使用 API 模型时需遵守使用条款并保护隐私数据。
- 若数据集包含版权限制，需确保仅用于研究与复现，不对外发布。
- 对比论文结果时，务必标注所用模型与 tool 差异，避免“虚假复现”。
