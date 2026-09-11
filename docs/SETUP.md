# 模型配置与运行指南

[返回项目首页](../README.md) · [验证范围](VALIDATION.md)

以下命令均在仓库根目录执行。首次体验可先运行 README 中的离线演示。

## 真实模型运行


轻量编排支持 Python 3.11 / 3.13；科学计算与完整模型环境使用 Python 3.12 以上，本机验证使用 3.13。依赖按用途拆分：`requirements-dev.txt` 为编排与测试，`requirements-science.txt` 增加 CPU 科学计算，`requirements.txt` 增加 GPU 模型适配器。版本与兼容性例外见[版本记录](DEPENDENCIES.md)。

```bash
python -m pip install -r requirements.txt
# 并配置各模型权重以及以下外部代码：
git clone https://github.com/mahsasheikh/DrugGen.git
git clone https://github.com/RyanWangZf/MediTab.git
```

对接使用 Vina 1.2.7、Open Babel 3.2.1、P2Rank 2.5.1 和 Java 25 LTS。Linux x86_64 可安装到仓库忽略的 `.tools/`，不覆盖系统工具：

```bash
bash scripts/install-science-tools.sh  # 需要 curl、tar、cmake、C++ 编译器
source scripts/activate-tools.sh
```

`llama-cpp-python` 默认安装不保证启用 CUDA。GPU 构建需 CUDA 编译工具，并按显卡架构配置 `CMAKE_ARGS`。Panacea、临床预测和可选 pKa 模型尚未完成新版环境联调。

本机 RTX 3090 的 CUDA 构建命令（其他显卡需调整架构号）：

```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86" \
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_BUILD_PARALLEL_LEVEL=4 \
python -m pip install --force-reinstall --no-deps --no-cache-dir \
  --no-binary=llama-cpp-python llama-cpp-python==0.3.35
```

参考 [配置模板](../.env.example) 设置环境变量，程序不会自动加载该文件：

| 变量 | 用途 |
|---|---|
| `DEEPSEEK_API_KEY` | LLM API 密钥 |
| `DEEPSEEK_MODEL` / `DEEPSEEK_PRO_MODEL` | 可覆盖的模型名，以账号实际可用模型为准；当前主流程使用前者 |
| `DEEPSEEK_BASE_URL` | 可覆盖的 API 地址 |
| `PANACEA_MODEL` | Panacea 模型路径或仓库 ID |
| `P2RANK_PATH` | P2Rank 安装目录 |
| `DRUGASSIST_MODEL_PATH` | 本地 GGUF 路径；未设置时尝试下载 |
| `HF_TOKEN` | 可选，用于 Hugging Face 服务或模型访问 |
| `MEDITAB_MODEL_PATH` | 可选，专门训练的单 logit 二分类检查点，正类必须代表成功；校准效果需独立验证 |
| `DEEPSEEK_MAX_TOKENS` | 每次请求输出上限，默认 4096，不是整个任务的费用上限 |
| `CHEMFM_BACKEND` | 默认远端 Space；`local` 使用本机 CUDA，目前只支持 hERG 和 Ames 分类端点 |

```bash
export DEEPSEEK_API_KEY="your-key"
python check_env.py --offline       # 本地环境检查
python check_env.py                 # 另检查 API 连通性
python DrugForge.py --task "模拟DPP4(P27487)的药物开发"
```

打开 `http://localhost:8765`，在优化前点击“确认继续优化”。命令行也支持：

```bash
python DrugForge.py --task "模拟DPP4(P27487)的药物开发" --patients diabetes_patients_final.xml
python DrugForge.py --port 8766 --exit-on-complete
```

启动时检查 API 密钥；工具连接或阶段执行失败时会保存失败状态并退出。成功后默认保留面板，按 `Ctrl+C` 关闭；`--exit-on-complete` 用于完成后退出。

## 运行记录

- `runs/{run_id}.json`：消息、阶段结果、耗时和运行状态。
- `runs/{run_id}.md`：从结构化工具数据生成的可追溯报告。
- `runs/{run_id}.draft.md`：LLM 文字草稿，不作为数值来源。
- `runs/{run_id}.evidence.json` / `.evidence.html`：机器可读证据与浏览器查看页。
- `runs/{run_id}.sqlite`：跨进程恢复所需的检查点。
- `runs/`、密钥配置和模型权重默认不纳入 Git。

```bash
python state.py
python state.py --show YOUR_RUN_ID
```

JSON 日志用于回放，SQLite 保存图状态。恢复命令：

```bash
python DrugForge.py --resume YOUR_RUN_ID
```

恢复时沿用原任务、靶点和患者文件，检查核心 LLM 配置与患者文件哈希；不要同时指定 `--task`、`--target` 或 `--patients`。旧版纯 JSON 日志不能升级为可恢复检查点。更多说明见[工程指南](ENGINEERING.md)。

## 可选追踪与记忆

设置 `LANGCHAIN_API_KEY` 可启用 LangSmith 调用追踪。安装可选的 Mem0 依赖并设置 `MEM0_API_KEY` 后，系统会检索和保存历史任务偏好。启用后，相应任务数据会发送到所配置的服务。

## 数据与验证边界

仓库附带合成患者 XML。原项目的数据生成说明引用了 [Synthea](https://github.com/synthetichealth/synthea) 和 [patient2trial](https://github.com/surdatta/patient2trial)；患者匹配解析器实际读取 `topic / text_version` 格式。

工程测试与真实工具实验分开报告，运行方法及实际结果见[科学验证](SCIENCE_VALIDATION.md)。患者匹配、临床预测校准和完整模型链路仍需独立验证。

## MCP 服务目录


| 文件 | 集成能力 |
|---|---|
| `druggen_mcp_server.py` | DrugGen 分子生成 |
| `docking_mcp_server.py` | AutoDock Vina / P2Rank 对接 |
| `chemical_properties_mcp_sever.py` | RDKit 理化性质与规则筛选（保留原文件名） |
| `admet_prediction_mcp_server.py` | ChemFM ADMET 预测 |
| `mol_opt_mcp_server.py` | DrugAssist 分子优化 |
| `name2smiles_mcp_server.py` | 名称与 SMILES 转换工具，供扩展使用 |
| `trialgen_mcp_server.py` | Panacea 临床方案组件生成 |
| `patient_matching_mcp_server.py` | Panacea 患者匹配，报告处理失败数量 |
| `trialpred_mcp_server.py` | MediTab 预测适配器，需配置专用检查点 |
