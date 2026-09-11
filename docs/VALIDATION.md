# 验证范围与已知限制

## 可重复执行的验证

```bash
python -m pip install -r requirements-dev.txt
python check_env.py --demo
python -m pytest -q
python demo.py --with-patients
python evaluate.py
```

工程测试覆盖：

- 真实主图与子图执行、三轮优化上限、提前结束及并行分支汇总。
- 结构化交接、调用前输入检查、工具结果格式与嵌套错误检测。
- 独立进程退出后恢复，不重复执行已完成的生成步骤；失败节点重试。
- 恢复后重新等待人工确认，拒绝重复或过期确认，禁止同时恢复同一运行。
- 报告数值与原始 JSON 路径一致、引用完整，以及输入 / 输出被修改后的拒绝行为。
- 阶段耗时、已知 token 用量、部分价格覆盖与缺失数据处理。
- 参考分子 / 患者结果的 precision / recall、二元结局的 Brier score。
- JSON 原子写入保护、运行 ID 校验与本地 MCP 会话复用。

离线节点使用固定测试数据，不加载领域模型。七项任务基准验证预期成功与预期失败；case_pass_rate 不等同于真实任务完成率。CI 在 Python 3.11 / 3.13 上运行测试与基准，并生成演示、来源证据和评测产物。

## 已知限制

1. 完整模型环境与领域连接器尚未完成真实联调；权重、API 及对接工具需分别配置。
2. 结构化契约保证格式、阶段输入及引用一致性，不保证 SMILES 化学有效性、工具计算或模型预测正确。未知 ADMET 单位不会自动推断。
3. 优化达标仍由 LLM 判断；端点含义和阈值需结合真实模型验证。
4. 临床预测默认不可用。配置后仍须独立验证检查点任务、正类定义与校准质量，输出的 `validated` 保持为 `false`。
5. SQLite 恢复以图节点为边界，不能保证失败节点内部的外部工具只执行一次；不支持旧版无检查点日志或不同图版本的自动迁移。
6. 阶段超时不等于远端计算取消；底层同步推理或远程请求可能继续执行。
7. 成本只估算具备用量及价格信息的 LLM 调用，不能替代账单或覆盖工具 / GPU 费用。未上报的失败请求用量可能缺失。
8. 固定基准不能代替真实数据集上的模型质量评测；患者匹配仍需人工标注样本和独立验证。
9. 人工确认面板面向本机，尚未实现多用户认证；检查点和来源文件包含输入数据，按原数据权限管理。

实现参考：[LangGraph 检查点与恢复](https://docs.langchain.com/oss/python/langgraph/persistence)、[MCP 会话接口](https://github.com/langchain-ai/langchain-mcp-adapters/blob/main/langchain_mcp_adapters/client.py)。
