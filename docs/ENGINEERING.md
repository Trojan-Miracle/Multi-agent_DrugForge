# 数据交接、恢复、证据与评测

[返回首页](../README.md) · [模型配置](SETUP.md) · [验证边界](VALIDATION.md)

## 结构化阶段交接

[`contracts.py`](../contracts.py) 定义 Pydantic 契约。阶段输入包含任务、靶点、当前分子和指定上游阶段的数据，不再把整段多 Agent 聊天记录当作下一阶段的输入。

| 数据 | 内容 |
|---|---|
| Molecule | SMILES、稳定字符串 ID、来源证据 ID、原始结果路径 |
| Observation | 对应分子、端点、数值或类别、单位、计算 / 预测类型、来源 |
| Selection | 分子、筛选结果、原因与来源 |
| TrialData | 原始方案文本、生成组件及来源 |
| PatientData / PredictionData | 匹配 ID 与数量、可选预测概率及验证状态 |

工具调用前检查当前分子集合、显式靶点、患者文件及试验文本是否一致；工具返回后检查结果格式和分子覆盖范围。ADMET 单位未提供时保留为未知，不猜测单位。优化工具必须返回 `optimized_smiles` JSON 字段，自由文本建议不能直接进入下一轮。

SMILES ID 标识字符串，不代表完成了化学等价性判断；分子是否合法仍需领域工具验证。

## 进程重启后续跑

真实运行自动创建 `runs/{run_id}.sqlite`。重新启动时：

```bash
python DrugForge.py --resume YOUR_RUN_ID
```

- 同一运行使用稳定的 `thread_id`，主图与优化子图共享持久检查点。
- 已提交到检查点的节点不重跑；失败节点可以重新执行，其内部未完成的工具调用可能重复。
- 待审批步骤在重启后生成新的确认令牌，仍需人工确认。
- 同一任务只能由一个进程恢复；进程崩溃后，SQLite 互斥锁由操作系统释放。
- 恢复时检查图版本、执行模式、核心 LLM 配置及患者文件哈希。请保持其他模型权重与工具配置一致。
- 旧版日志没有检查点，不能直接恢复。完成的任务不能重复续跑。

可用离线演示亲自验证“先退出，再续跑”：

```bash
python demo.py --persist --pause-after 1
# 第一条命令在首轮审批处保存并退出，终端会打印 RUN_ID
python demo.py --resume RUN_ID
```

演示会自动确认，真实入口不会。`DRUGFORGE_RUNS_DIR` 可指定记录目录；恢复时须使用同一目录。检查点与日志的访问权限应与模型输入数据保持一致。

## 从数值追溯到工具来源

`runs/{run_id}.md` 直接由结构化数据生成；数值行的来源链接指向 `.evidence.html` 中相应工具的原始输入和输出。

`.evidence.json` 保存证据 ID、调用 ID、输入参数、原始内容、规范化载荷、JSON Pointer 和输入 / 输出 SHA-256。导出时验证数值与原始路径相符，并检查哈希；引用丢失或数值被修改时拒绝生成报告。

`*.draft.md` 是独立保存的 LLM 草稿，不用于构造报告数值。哈希用于完整性检查，不证明模型预测准确或实验结论成立。

## 任务评测

### 固定离线基准

```bash
python evaluate.py
```

默认运行 [`offline_cases.json`](../evaluations/offline_cases.json) 中的七个用例：完整流程、患者分支、预测不可用、分子输入错误、工具故障、漏调用工具和非结构化优化输出。

- **case_pass_rate**：任务结果是否符合用例预期，包括正确拒绝错误输入。
- **completion_rate**：实际跑到终点的任务比例。预期被拒绝的任务不会被计为完成。
- **tool_argument_accuracy / tool_success_rate**：输入契约通过比例 / 工具结果通过校验比例。
- **参考答案得分**：必要工具召回、候选分子与患者 ID 的 precision / recall。

输出 `runs/evaluation.json`、`runs/evaluation.md`，并在 `runs/evaluation.traces/` 保存每个用例的完整记录。固定用例不衡量真实模型的科学效果。

### 真实运行日志与人工参考答案

```bash
python evaluate.py --runs runs/RUN_ID.json --reference reference.json --prices prices.json
```

`reference.json` 以运行 ID 为键；各参考项可选：

```json
{
  "RUN_ID": {
    "required_tools": ["run_druggen", "run_docking"],
    "candidate_smiles": ["REFERENCE_SMILES"],
    "patient_ids": ["REFERENCE_PATIENT_ID"],
    "trial_outcome": 1
  }
}
```

候选比较使用最终规则筛选分子的精确 SMILES 字符串。提供真实二元结局时，可计算已有预测的 Brier score；没有参考答案的字段不生成质量得分。真实日志与离线日志不能混合统计。

### 耗时与成本

记录各阶段耗时及本次运行累计在线时间，暂停后的离线时段不计入累计时间。并行阶段的耗时之和可能大于总运行时间。强制终止进程可能丢失最后一次写入之后的时长与用量，恢复时会标记统计不完整。

LLM token 用量来自返回消息中的 usage metadata，包括优化判断调用。单价由用户提供并随运行记录保存，不内置供应商价格。参考 [`prices.example.json`](../evaluations/prices.example.json)，在 `models` 下按实际模型名填写 `input_per_million` 和 `output_per_million`，并设置 `currency`。

只有全部已观察调用具备用量和价格且没有失败阶段时，才给出完整的 **estimated_llm_cost**。否则总估算为 `null`，可展示已知部分 **priced_llm_subtotal** 和价格覆盖率。估算不包含 GPU、领域工具费用，以及供应商未返回用量的失败请求，不等同于账单。
