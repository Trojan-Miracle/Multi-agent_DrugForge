# DrugForge：基于多 Agent 的端到端药物研发模拟系统


## 系统概述

DrugForge 是一个基于 **LangGraph StateGraph** 的端到端药物研发模拟系统。流程由固定图结构编排，8 个专用 Agent 各司其职，覆盖从分子生成到临床试验预测的完整研发流水线。

**技术栈：**
- **编排框架**：LangGraph（StateGraph 固定流程 + 条件边 + 并行节点）
- **工具集成**：MCP（Model Context Protocol），9 个领域专用服务器
- **LLM 后端**：DeepSeek API（flash / pro + thinking 模式，按任务分级分配）
- **可观测性**：实时可视化面板（HTTP + viz.html）+ LangSmith 调用链追踪

---

## 工作流程

```
用户输入（靶点 / 任务描述）
          │
          ▼
    planning_agent（制定研发策略）
          │
          ▼
    druggen_agent ──→ 生成 SMILES 命中物（7个）
          │
          ▼
    admet_properties_agent ──→ 分子对接初筛（AutoDock Vina）
          │
          ▼
    chemical_agent ──→ 理化性质 + Lipinski/Veber 筛选
          │
          ▼
    admet_properties_agent ──→ ADMET 性质预测，选出先导化合物
          │
          ▼
    ┌─────────────────────────────────┐
    │     先导化合物优化循环（最多3次）  │◄─────────────┐
    │                                 │              │
    │  [Human-in-the-loop 暂停确认]   │              │
    │         ↓                       │              │
    │  molecule_optimization_agent    │    未满足条件  │
    │         ↓                       │              │
    │  admet_properties_agent（重评）  │              │
    │         ↓                       │              │
    │  chemical_agent（重筛）          │──────────────┘
    │         ↓                       │
    │  eval_opt（评估是否满足条件）     │──→ 满足或达3次 → 退出循环
    └─────────────────────────────────┘
          │
          ▼
    admet_properties_agent ──→ 最终毒性 / PK / PD 评估
          │
          ▼
    trial_generator_agent ──→ 临床试验方案
          │
          ├──→ patient_matching_agent ──┐  （并行）
          └──→ trial_prediction_agent ──┤
                                        ▼
                                 planning_agent（最终报告 FINAL）
```

---

## Agent 说明

| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| **planning_agent** | 制定研发策略；整合所有结果写最终报告 | 用户任务描述 | 研发计划 + 最终汇总报告 |
| **druggen_agent** | 根据靶点生成候选分子 SMILES | UniProt ID | SMILES 列表 |
| **chemical_agent** | 计算理化描述符，按 Lipinski / Veber 规则筛选先导化合物 | SMILES 列表 | MW、logP、TPSA 等描述符；筛选后先导化合物列表 |
| **admet_properties_agent** | 预测 ADMET 性质和分子对接评分 | SMILES 列表、靶点名称 | 口服生物利用度、溶解度、hERG、肝毒性等；对接评分 |
| **molecule_optimization_agent** | 针对指定性质优化先导化合物结构 | SMILES + 目标性质 + 方向（increase/decrease） | 优化后 SMILES |
| **trial_generator_agent** | 生成结构化临床试验方案 | 先导化合物 SMILES / 药物名称 | 包含分期、入组、终点、臂等完整试验文本 |
| **patient_matching_agent** | 将患者档案与试验入组标准匹配 | XML 患者数据路径 + 试验文本 | 匹配患者数量及 ID 列表 |
| **trial_prediction_agent** | 预测临床试验成功概率 | 结构化试验文本 | 成功概率（0–1） |

---

## MCP 工具服务器

| 文件 | 功能 |
|---|---|
| `druggen_mcp_server.py` | 调用 DrugGen 模型生成分子 |
| `docking_mcp_server.py` | AutoDock Vina + P2Rank 口袋检测 |
| `chemical_properties_mcp_server.py` | RDKit 理化描述符计算 |
| `admet_prediction_mcp_server.py` | ChemFM ADMET 预测 |
| `mol_opt_mcp_server.py` | 迭代式分子优化 |
| `name2smiles_mcp_server.py` | 药物名称 / InChI → SMILES |
| `patient_matching_mcp_server.py` | Panacea 患者-试验匹配 |
| `trialgen_mcp_server.py` | Panacea 临床试验方案生成 |
| `trialpred_mcp_server.py` | MediTab 试验成功率预测 |

---

## Harness

系统内置运行保障机制，确保长流程可靠执行：

| 组件 | 实现 | 作用 |
|---|---|---|
| **环境检查** | `check_env.py` | 运行前验证 API Key、Python 包、MCP 文件、DeepSeek 连通性 |
| **终止保护** | `interrupt_before` + `eval_opt` + `opt_count` | 优化循环最多3次，流程整体最多30轮 |
| **工具权限** | 每个 Agent 只绑定专属工具列表 | 防止 Agent 越权调用 |
| **输出验证** | `_validate_output()` | 流程结束后检查最终报告是否完整 |
| **运行日志** | `state.py` + `runs/{run_id}.json` | 实时记录所有 Agent 消息，支持历史回放 |
| **Human-in-the-loop** | `MemorySaver` + `interrupt_before` + 浏览器确认按钮 | 每次优化前暂停，浏览器面板点击"确认"继续 |

---

## 患者数据

合成患者由 [Synthea](https://github.com/synthetichealth/synthea) 生成（FHIR XML 格式）。  
患者临床叙述通过 [patient2trial](https://github.com/surdatta/patient2trial) 的 `patient_topic_expansion.py` 脚本构建。

---

## 环境配置

需要 Python 3.11。

```bash
git clone https://github.com/Trojan-Miracle/Multi-agent_DrugForge
cd Multi-agent_DrugForge
pip install -r requirements.txt
git clone https://github.com/mahsasheikh/DrugGen.git
git clone https://github.com/RyanWangZf/MediTab.git
```

额外工具（需指定版本）：AutoDock Vina 1.1.2、Open Babel 3.1.1、P2Rank 2.5.1

运行 `check_env.py` 验证环境：

```bash
python check_env.py
```

---

## 运行

设置 DeepSeek API Key：

```bash
export DEEPSEEK_API_KEY="sk-xxx..."   # Linux/macOS
$env:DEEPSEEK_API_KEY="sk-xxx..."    # Windows PowerShell
```

启动：

```bash
python DrugForge.py
```

浏览器打开 `http://localhost:8765` 查看实时可视化，按回车开始运行后按提示输入任务，例如：

```
模拟DPP4(P27487)的药物开发
```

优化阶段会暂停，在浏览器可视化面板底部点击"确认继续优化"按钮即可继续。

---

## LangSmith 追踪（可选）

在 [smith.langchain.com](https://smith.langchain.com) 注册后获取 API Key：

```bash
$env:LANGCHAIN_API_KEY="ls-xxx..."   # Windows PowerShell
export LANGCHAIN_API_KEY="ls-xxx..."  # Linux/macOS
```

启动后控制台提示 `[LangSmith] 追踪已启用`，所有 Agent 调用、工具调用、token 消耗可在 LangSmith 面板查看。

---

## 运行记录

每次运行自动生成日志文件 `runs/{run_id}.json`，记录所有 Agent 消息。

```bash
python state.py                          # 查看历史运行列表
python state.py --show 20250628_143012   # 回放某次运行
```
