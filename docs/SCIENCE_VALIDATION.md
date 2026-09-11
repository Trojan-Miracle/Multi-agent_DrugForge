# 真实工具与公开实验验证

[返回首页](../README.md) · [版本记录](DEPENDENCIES.md) · [原始结果](../evaluations/results/2026-09-11/)

以下实验独立于离线演示，不使用 fixture 冒充领域模型输出。固定输入、参数和模型修订，保留失败结果；单个案例或公开基准复现不能证明新药有效。

## 已执行的实验

| 实验 | 结果 | 解释 |
|---|---|---|
| DeepSeek → MCP → RDKit | 阿司匹林与乙醇的实际描述符计算成功 | 初次因可选 pKa 模型依赖失败，修复后复验通过；失败记录保留 |
| 1IEP 重对接，已知口袋 | Top-1 重原子 RMSD **0.360 Å** | 最高分姿态通过本次 `<2 Å` 标准；Vina 分数 -13.25 kcal/mol |
| 1IEP 重对接，P2Rank Top-1 口袋 | Top-1 重原子 RMSD **12.644 Å** | 未通过；Vina 分数 -11.40 kcal/mol。两种口袋条件的中心与盒大小不同，不单独归因于某个参数 |
| TDC hERG，Morgan 指纹＋随机森林 | AUROC **0.786 / 0.794 / 0.811** | 官方固定测试集 132 个分子，随机种子 0 / 1 / 2 |
| TDC Ames，同一基线 | AUROC **0.843 / 0.844 / 0.845** | 官方固定测试集 1,457 个分子，同样三个种子 |
| DrugGen，DPP4 / P27487 | **7 / 7** 输出可解析，**3 / 7** 通过理化规则 | RTX 3090，seed=42；仅七个样本，不代表整体生成有效率 |
| ChemFM，TDC hERG | AUROC **0.972**，Brier **0.061** | 同一官方测试集 132 个分子；固定任务适配器，float32 GPU 推理 |
| ChemFM，TDC Ames | AUROC **0.977**，Brier **0.056** | 同一官方测试集 1,457 个分子；保留逐样本概率 |
| DrugAssist 三轮优化对照 | **0 / 3** 最终改善，9 次工具调用均产生有效分子 | 三条规则通过的 DrugGen 分子；hERG 风险预测均上升，不能宣称优化有效 |
| 三条 DrugGen 候选的 DPP4 对接 | **3 / 3** 执行成功 | 结构 2QT9 的 DBREF 对应 P27487；Vina 分数 -6.599 / -6.977 / -7.172 kcal/mol，不能解释为实验亲和力 |

两个端点的基线训练数据均未发现与测试集完全相同的 canonical SMILES。ChemFM 公开权重使用过 TDC 数据，检查点训练成员未独立核实，因此这里是**公开基准复现**，不是独立外部泛化证据，不能据此宣称真实药研效果。原始结果含 AUROC、AUPRC、Brier 及逐分子预测，基线与 ChemFM 明确区分。

## 复现

```bash
python -m pip install -r requirements-science.txt
bash scripts/install-science-tools.sh
source scripts/activate-tools.sh

python science_validation.py redocking --output runs/science/redocking-known
python science_validation.py redocking --pocket p2rank --output runs/science/redocking-p2rank

curl -fL https://dataverse.harvard.edu/api/access/datafile/4426004 -o runs/tdc-admet.zip
python evaluate_admet.py --data-archive runs/tdc-admet.zip
```

对接使用 Vina 官方 v1.2.7 教程输入，输入文件有 SHA-256 检查。Meeko 恢复配体键序，RDKit `CalcRMS` 在蛋白坐标系内计算对称性校正的重原子 RMSD，**不旋转或平移预测配体去贴合参考结构**。否则会掩盖位置错误。两种条件均为 exhaustiveness=32、seed=42，当前只有一个案例、一个种子。

真实语言模型联调需在环境中设置 `DEEPSEEK_API_KEY`：

```bash
python science_validation.py agent --output runs/science/agent
```

该脚本限制输出和图步数，保存真实 usage。已执行的失败和成功两次尝试合计输入 1,966、输出 651 token；按 2026-09-11 Flash 高峰未缓存价估算约 **0.00914 元**，不是完整流程费用或供应商账单。

## GPU 实验入口

```bash
python -m pip install -r requirements.txt
source scripts/activate-tools.sh
CUDA_VISIBLE_DEVICES=0 python validate_druggen.py --target P27487
CUDA_VISIBLE_DEVICES=0 python evaluate_admet.py --chemfm --data-archive runs/tdc-admet.zip --output runs/science/admet-chemfm
CUDA_VISIBLE_DEVICES=0 python evaluate_optimization.py
python validate_candidate_docking.py
```

DrugGen 入口使用官方生成器、真实 UniProt 序列和固定模型修订，记录有效分子比例与理化筛选。ChemFM 本地入口使用原作者的 SMILES 标记、392-token 任务词表、训练分类头及 sigmoid 后处理，目前仅支持 hERG/Ames，拒绝超长输入的静默截断。分类头与词嵌入逐张量核对原始权重；适配器切换时恢复各自的共享词嵌入，真实 hERG → Ames → hERG 往返检查通过。未实现回归模型的逆缩放，不输出溶解度等回归预测。

优化对照固定同一批初始分子，以零轮作为对照，最多执行三轮 DrugAssist 编辑。三组 hERG 预测分别从 0.0197 → 0.6439、0.0197 → 0.5619、0.0288 → 0.1402；最终都通过理化规则，Morgan 相似度约 0.69–0.70。该结果是三条相关分子的单种子小样本观察，评价使用同一 ChemFM 模型，**没有独立活性验证，也没有证明优化收益**。保存了每轮输入、输出、预测和失败状态，不从中挑最好的一轮作为最终结果。

模型加载或推理失败会记录为失败，不能用随机初始化的分类头或模拟概率替代。端到端临床链路和临床预测质量尚未验证。

参考：[Vina 官方教程](https://autodock-vina.readthedocs.io/en/stable/docking_basic.html)、[TDC ADMET 基准](https://tdcommons.ai/benchmark/admet_group/overview/)、[ChemFM 模型](https://huggingface.co/ChemFM/admet_herg)、[DeepSeek 官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。
