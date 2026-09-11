# 依赖版本与验证状态

[返回首页](../README.md)

2026-09-11 查询 PyPI 和各工具官方发布记录，将项目直接依赖升级并固定版本。删除旧环境中与源码无直接关系的大量传递依赖，由安装器解析它们；可选记忆服务独立到 `requirements-memory.txt`。

| 组件 | 当前版本 | 状态 |
|---|---|---|
| LangGraph / LangChain | 1.2.11 / 1.4.0 | 迁移到 `langchain.agents.create_agent`；图与持久恢复回归验证 |
| LangChain OpenAI / MCP adapters | 1.6.2 / 0.3.2 | DeepSeek + 本地 MCP/RDKit 真实调用通过 |
| MCP | 1.30.0 | 最新兼容 1.x；官方适配器要求 `<2`，不能与最新 2.2.0 同装 |
| SQLite checkpointer / aiosqlite | 3.1.1 / 0.22.1 | 持久恢复回归验证；图版本升至 3，拒绝旧图检查点 |
| RDKit / NumPy / pandas | 2026.3.6 / 2.5.3 / 3.0.5 | 实际描述符计算、科学评测 |
| scikit-learn / Gradio client | 1.9.1 / 2.7.0 | 基线评测通过；ChemFM 远端 Space 暂停 |
| AutoDock Vina / Open Babel | 1.2.7 / 3.2.1 | 官方重对接案例实际执行；Vina 新版移除旧 `--log` 参数，改为保存标准输出 |
| P2Rank | 2.5.1 | 最新发布版；自动口袋分支实际执行 |
| Java | Temurin 25.0.4.1 LTS | 最新 Java 26.0.2.1 与 P2Rank 内置 Groovy 不兼容，使用最新 LTS 补丁版 |
| Torch / Transformers / PEFT | 2.14.0 / 5.17.0 / 0.20.0 | DrugGen 生成、ChemFM hERG/Ames 真实 GPU 推理通过；旧适配器显式加载任务词表和分类头 |
| llama-cpp-python | 0.3.35 | CUDA 13 / RTX 3090 构建完成，DrugAssist 三组真实优化对照已执行 |

模型权重按任务匹配的官方修订固定；不同架构的新模型不能直接替换旧任务适配器。例如 ChemFM 的 hERG/Ames 适配器依赖 ChemFM-3B，不能只因存在更新的基础模型就替换骨干。

完整依赖与可选记忆依赖已在本机成功安装，`pip check` 通过；其中仍包含尚未联调的可选模型，不能把“安装成功”理解为全部工具已验证。

来源：[PyPI](https://pypi.org/)、[LangChain v1 迁移](https://docs.langchain.com/oss/python/migrate/langchain-v1)、[MCP 适配器](https://github.com/langchain-ai/langchain-mcp-adapters)、[Vina 发布](https://github.com/ccsb-scripps/AutoDock-Vina/releases)、[Open Babel 发布](https://github.com/openbabel/openbabel/releases)、[P2Rank 发布](https://github.com/rdk/p2rank/releases)、[Temurin](https://adoptium.net/)。
