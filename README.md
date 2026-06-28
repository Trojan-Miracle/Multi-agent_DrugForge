# DrugForge：基于多 Agent 的端到端药物研发模拟系统

---

## 系统概述

DrugForge 是一个模块化多 Agent 框架，由 planning_agent 统一调度，覆盖药物研发全生命周期。系统分三个阶段运行，各阶段由专用 Agent 自动执行，最终输出药物候选报告与临床试验可行性评估。

---

## 工作流程

```
用户输入（靶点 / 任务描述）
          │
          ▼
    planning_agent（规划与调度）
          │
          ├─── 药物发现阶段 ────────────────────────────────────────┐
          │                                                          │
          │   druggen_agent ──→ 生成 SMILES 命中物                  │
          │         │                                                │
          │         ▼                                                │
          │   admet_properties_agent ──→ 对接评分（初筛）            │
          │         │                                                │
          │         ▼                                                │
          │   chemical_agent ──→ 理化性质 + Lipinski/Veber 筛选     │
          │         │                                                │
          │         ▼                                                │
          │   admet_properties_agent ──→ ADMET 性质预测             │
          │         │                                                │
          │         └──→ 选出先导化合物                               │
          │                   │                                      │
          │           ┌───────┴──────────┐                          │
          │           │  先导化合物优化循环  │（最多 3 次）             │
          │           │                  │                          │
          │           │ mol_opt_agent    │                          │
          │           │   ↓              │                          │
          │           │ admet + docking  │                          │
          │           │   ↓              │                          │
          │           │ chemical_agent   │                          │
          │           └──────────────────┘                          │
          │                                                          │
          ├─── 临床前阶段 ──────────────────────────────────────────┤
          │                                                          │
          │   admet_properties_agent ──→ 毒性 / PD / PK 最终评估   │
          │                                                          │
          ├─── 临床阶段 ───────────────────────────────────────────┤
          │                                                          │
          │   trial_generator_agent ──→ 临床试验方案                │
          │         │                                                │
          │         ├──→ patient_matching_agent ──→ 患者匹配        │
          │         └──→ trial_prediction_agent ──→ 成功概率        │
          │                                                          │
          ▼                                                          │
      最终报告（FINAL）◄──────────────────────────────────────────┘
```

---

## Agent 说明

| Agent | 职责 | 输入 | 输出 |
|---|---|---|---|
| **planning_agent** | 拆解任务、分配子任务、监控流程、输出最终报告 | 用户任务描述 | 阶段性计划 + 最终汇总报告 |
| **druggen_agent** | 根据靶点生成候选分子 SMILES | UniProt ID、生成数量 | SMILES 列表 |
| **chemical_agent** | 计算理化描述符，按 Lipinski / Veber 规则筛选先导化合物 | SMILES 列表 | MW、logP、TPSA、HBD、HBA 等描述符；筛选后先导化合物列表 |
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

运行 `check_env.py` 验证环境（同时检查科研依赖和 API 连通）：

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

---

## 运行记录

每次运行自动生成日志文件 `runs/{run_id}.json`，记录所有 Agent 消息。

查看历史运行：

```bash
python state.py
```

回放某次运行：

```bash
python state.py --show 20250628_143012_abc123
```
