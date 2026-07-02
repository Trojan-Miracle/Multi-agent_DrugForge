import asyncio
import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from state import RunState

PROJECT_DIR = Path(__file__).parent

# ── 可视化服务器 ─────────────────────────────────────────────────────────────

messages_store: list = []
_run: RunState | None = None

class VizHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open("viz.html", "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/messages":
            data = json.dumps(messages_store, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format, *args):
        pass

def start_viz_server():
    server = HTTPServer(("localhost", 8765), VizHandler)
    server.serve_forever()

async def broadcast(agent: str, msg_type: str, content: str):
    messages_store.append({
        "id": len(messages_store),
        "agent": agent,
        "type": msg_type,
        "content": content,
    })
    if _run:
        _run.append(agent, msg_type, content)

def _unwrap_json_blocks(s: str) -> str:
    lines = s.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and '"type"' in line:
            try:
                blocks = json.loads(line)
                if isinstance(blocks, list):
                    texts = [b["text"] for b in blocks
                             if isinstance(b, dict) and b.get("type") == "text" and "text" in b]
                    if texts:
                        results.append("\n".join(texts))
                        continue
            except Exception:
                pass
        results.append(line)
    return "\n".join(results)

def _validate_output():
    for msg in reversed(messages_store):
        if msg["agent"] == "planning_agent" and "FINAL" in msg["content"]:
            if len(msg["content"]) > 200:
                print("[验证] 最终报告完整", flush=True)
            else:
                print("[验证] 警告：planning_agent 说了 FINAL 但报告内容过短", flush=True)
            return
    print("[验证] 警告：未检测到 FINAL 报告，流程可能提前终止", flush=True)

# ── LangGraph 状态 ────────────────────────────────────────────────────────────

class DrugForgeState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    turns: int

# ── 模型工厂 ──────────────────────────────────────────────────────────────────

def ds(thinking: bool = False, pro: bool = False) -> ChatOpenAI:
    model = "deepseek-v4-pro" if pro else "deepseek-v4-flash"
    kw = {"model_kwargs": {"extra_body": {"thinking": {"type": "enabled"}}}} if thinking else {}
    return ChatOpenAI(
        model=model,
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        **kw,
    )

# ── System Prompts ────────────────────────────────────────────────────────────

PLANNING_SYS = """
        你是规划agent。
        你的职责是将复杂任务拆解为更小的可管理子任务。
        你的团队成员包括：
            1. druggen_agent：仅根据给定的UniProt ID生成药物SMILES，如果任务中未提供药物则使用它。
            2. chemical_agent — 化学性质专家
                 职责范围：预测/报告化学性质并返回先导化合物（不生成试验/药物）。
                 可用工具：
                   - select_leads_from_smiles(smiles_list)
                   - predict_pka_batch(smiles_list)
                   - logd_acid_batch(smiles_list, pH=7.4)
                   - logd_base_batch(smiles_list, pH=7.4)
                   - rdkit_physchem_batch(smiles_list, pH=7.4)
                   - predict_all_batch(smiles_list, is_acid=None|bool, is_base=None|bool, pH=7.4)
            3. admet_properties_agent — ADMET性质专家
                 职责范围：预测/报告ADMET性质和对接；可解析名称/SMILES（不生成试验/药物）。
                 选择主要先导化合物时只选一个（最佳的）。
                 可用工具：
                   - get_drug_name_from_smiles(smiles_list)
                   - get_smiles_from_drug_name(drug_name)
                   - chemfm_list_properties()
                   - chemfm_get_description(property_name)
                   - chemfm_predict_single(smiles, property_name)
                   - chemfm_predict_many(smiles, properties_list)
                   - run_docking(target, smiles_list)
                 始终先规划chemfm_list_properties工具，以获取运行其他ADMET工具所需的精确性质名称。
            4. molecule_optimization_agent：针对需要增加/减少的性质优化先导化合物。
               必须传入ADMET或化学性质，不能传入对接相关参数。
            5. trial_generation_agent：为给定的先导化合物生成临床试验，在需要生成试验时使用。
               除非它是唯一需要的agent，否则始终在admet agent和chem agent之后最后调用。
               始终以如下结构化格式返回：
               - drug: SMILES和NAME

CLINICAL TRIAL:
- acronym: 字符串，简短研究名称
- brief_title: 字符串
- official_title: 字符串，描述性试验标题
- study_status: 字符串（例如"Recruiting"、"Completed"）
- study_start_date: 字符串，ISO格式（例如"2026-03"）
- primary_completion_date: 字符串，ISO格式
- completion_date: 字符串，ISO格式
- condition: 字符串，研究的临床适应症
- study_type: 字符串
- phase: 字符串
- intervention_model: 字符串
- allocation: 字符串
- masking: 字符串
- enrollment: 整数
- arms: 使用panacea_extract_components工具的输出
- intervention_description: 字符串，必须通过SMILES引用该分子
- primary_outcomes: 使用panacea_extract_components工具的输出
- secondary_outcomes: 使用panacea_extract_components工具的输出
- other_outcomes: 使用panacea_extract_components工具的输出
- eligibility_criteria: 使用panacea_extract_components工具的输出
- study_documents: 字符串列表
- brief_summary: 字符串，对试验目的、设计、干预措施和入组资格的简洁描述。
            6. patient_matching_agent：将患者与试验匹配。未提供XML患者摘要路径时不要规划此agent。
               不要给它编造的试验摘要文本，严格使用trial_generation_agent生成的试验摘要文本。
            7. trial_prediction_agent：根据trial_generation_agent的结构化输出预测试验成功概率。

        规则：
            严格遵循以下药物开发任务工作流程：
            - 药物发现阶段：
              - 命中物生成：druggen_agent生成>10个SMILES。
              - 对接：admet_properties_agent使用run_docking对命中物评分和筛选（保留评分最低的）。
              - 先导化合物鉴定：chemical_agent进行理化性质计算和select_leads_from_smiles（应用Lipinski/Veber过滤）；admet_properties_agent评估10个相关ADMET性质。
              - 选择单个最佳先导化合物（对接评分最低+通过过滤器+最佳ADMET，例如生物利用度>0.5，低hERG/肝毒性）。
            - 先导化合物优化阶段：
              - 对最佳先导化合物使用molecule_optimization_agent，针对薄弱性质（例如生物利用度<0.5时提高，降低毒性风险）。
              - 重新评估：admet_properties_agent（对接+ADMET），chemical_agent（理化性质/Lipinski/Veber）。
              - 循环：如果不满意，规划另一次优化。最多3次迭代；如果仍不满意，选择现有最佳并继续。
            - 临床前阶段：用admet_properties_agent对优化后的先导化合物进行最终ADMET重新评估。
            - 临床阶段：对最终先导化合物使用trial_generation_agent；如果有XML路径，使用patient_matching_agent；然后使用trial_prediction_agent。
            - 最终报告格式：
                药物发现性质：
                - 命中物：[初始SMILES列表]
                - 先导化合物：[带性质的已选先导化合物列表]
                - 优化先导化合物：带优化性质的SMILES
                - 化学性质：[来自chem agent的字典或列表]
                - ADMET性质：[来自admet agent的字典或列表]
                - 对接评分：[评分]

                临床试验报告：
                [来自trial agent的完整结构化试验]

                患者匹配：[结果（如适用）]

                试验成功概率：[概率]

                总结：[整体总结]
            - 不要编造agent的响应。
            - 所有任务完成后，以指定格式输出最终报告，并在最后单独一行写 FINAL。
            - 除终止流程外，在任何计划说明中都不要出现"FINAL"。
        """

DRUGGEN_SYS = """
    你是药物生成专家。你的任务是根据指定的生物靶点设计新颖的SMILES分子。
    你有一个工具run_druggen，可以根据给定的靶点返回药物SMILES。始终生成7个分子。
    在任何情况下都不要进行性质预测、分子对接或自由文本生成。
    只运行run_druggen工具并返回其结果。如果任务中已存在SMILES，则不要运行该工具。
    """

CHEM_PROPERTIES_SYS = """
    你是化学性质专家。你的任务是预测并报告分子的化学性质。
    在任何情况下都不要进行临床试验生成、药物生成、ADMET预测或分子对接。
    你有以下工具：
        - select_leads_from_smiles(smiles_list, n)：根据Lipinski/Veber规则筛选n个先导化合物。
        - predict_pka_batch：预测SMILES列表的pKa值。
        - logd_acid_batch / logd_base_batch：计算logD值。
        - rdkit_physchem_batch：计算理化描述符（MW、logP、TPSA、HBD、HBA、RotB、QED等）。
        - predict_all_batch：整合以上所有功能。
    """

ADMET_PROPERTIES_SYS = """
    你是ADMET性质专家。你的任务是预测并报告分子的ADMET性质。
    在任何情况下都不要生成临床试验、药物或自由文本。始终先运行chemfm_list_properties获取精确的性质名称。
    选择10个最相关的ADMET性质（例如口服生物利用度、溶解度、清除率、BBB通透性、hERG抑制、肝毒性）。
    你有以下工具：
        - get_drug_name_from_smiles(smiles)
        - get_smiles_from_drug_name
        - chemfm_list_properties
        - chemfm_get_description(property_name)
        - chemfm_predict_single(smiles, property_name)
        - chemfm_predict_many(smiles, properties)
        - run_docking：执行分子对接，评分越低越好。
    """

MOL_OPT_SYS = """
    你是分子优化专家。你有一个工具：
      - molecule_optimizer(smiles, properties, action)：输入SMILES、需要优化的性质以及期望的动作（increase或decrease）。
    不要生成临床试验或药物，不要预测性质或对接评分。只返回工具结果。
    如果ADMET和化学性质尚未预测和评估，则不要运行。
    """

TRIAL_SYS = """
    你是临床试验设计专家。
    你的任务是为药物生成真实的临床试验方案。如果药物由druggen agent生成，使用对接评分最低的药物。
    你不能预测性质、执行对接或生成分子。
    你必须在有SMILES或药物名称的情况下生成试验。

    首先，按以下格式输出初始试验文本：
- drug: SMILES（如有名称则附上NAME）

CLINICAL TRIAL:
- acronym: 字符串
- brief_title: 字符串
- official_title: 字符串
- study_status: 字符串
- study_start_date: ISO格式
- primary_completion_date: ISO格式
- completion_date: ISO格式
- condition: 字符串
- study_type: 字符串
- phase: 字符串
- enrollment: 整数

    然后，以整段文本作为trial_text参数调用panacea_extract_components工具，构建完整结构化报告。
    """

PATIENT_MATCHING_SYS = """
    你是患者匹配专家。你有一个工具：
        - match_patient_trial(xml_path: str, trial_text: str)：将患者与试验进行匹配。
        在未提供XML患者摘要路径的情况下，不要运行此工具。
    使用trial_generation_agent提供的精确结构化试验文本。
    返回匹配患者数量和匹配患者ID列表。
    """

TRIAL_PRED_SYS = """
    你是试验预测Agent。根据提供的试验文本预测试验成功概率。
    只返回概率分数（例如0.75）。不要生成试验或药物，不要预测性质或对接评分。只返回工具结果。
    """

SUPERVISOR_PROMPT = """你是任务调度器，负责决定下一个发言的 agent。

可用 agents：
- planning_agent：规划与调度，任务开始时最先介入，所有任务完成后输出最终报告并写 FINAL
- druggen_agent：根据 UniProt ID 生成候选分子 SMILES
- chemical_agent：理化性质计算，Lipinski/Veber 先导化合物筛选
- admet_properties_agent：ADMET 预测和分子对接评分
- molecule_optimization_agent：先导化合物结构优化（最多 3 次）
- trial_generator_agent：临床试验方案生成
- patient_matching_agent：患者-试验匹配（需提供 XML 路径）
- trial_prediction_agent：临床试验成功率预测

根据对话历史严格遵循 planning_agent 的计划，选择下一个 agent。
只回答 agent 的名字，不要加任何解释。"""

AGENT_NAMES = {
    "planning_agent", "druggen_agent", "chemical_agent", "admet_properties_agent",
    "molecule_optimization_agent", "trial_generator_agent",
    "patient_matching_agent", "trial_prediction_agent",
}

# ── Graph 构建 ────────────────────────────────────────────────────────────────

def make_agent_node(name: str, agent):
    async def node(state: DrugForgeState) -> dict:
        result = await agent.ainvoke({"messages": state["messages"]})
        new_msgs = result["messages"][len(state["messages"]):]
        return {"messages": new_msgs}
    node.__name__ = name
    return node

def build_graph(agent_nodes: dict, selector_llm: ChatOpenAI):
    async def supervisor_node(state: DrugForgeState) -> dict:
        messages = state["messages"]
        turns = state.get("turns", 0)

        if turns >= 30:
            return {"next_agent": END, "turns": turns + 1}

        for msg in reversed(messages[-5:]):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if "FINAL" in content:
                return {"next_agent": END, "turns": turns + 1}

        response = await selector_llm.ainvoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            *messages,
            HumanMessage(content="下一个agent是谁？只回答名字。"),
        ])
        next_agent = response.content.strip().split()[0]
        if next_agent not in AGENT_NAMES:
            next_agent = END
        return {"next_agent": next_agent, "turns": turns + 1}

    def route(state: DrugForgeState) -> str:
        n = state.get("next_agent", END)
        return n if n in agent_nodes else END

    workflow = StateGraph(DrugForgeState)
    workflow.add_node("supervisor", supervisor_node)
    for name, node in agent_nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", route, {**{n: n for n in agent_nodes}, END: END})
    for name in agent_nodes:
        workflow.add_edge(name, "supervisor")

    return workflow.compile()

# ── 可视化流 ──────────────────────────────────────────────────────────────────

async def run_with_viz(graph, task: str):
    initial = {
        "messages": [HumanMessage(content=task)],
        "next_agent": "",
        "turns": 0,
    }
    async for update in graph.astream(initial, stream_mode="updates"):
        for node_name, output in update.items():
            if node_name == "supervisor":
                continue
            for msg in output.get("messages", []):
                if isinstance(msg, AIMessage):
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    content = _unwrap_json_blocks(content)
                    if content.strip():
                        await broadcast(node_name, "AIMessage", content)
                elif isinstance(msg, ToolMessage):
                    content = _unwrap_json_blocks(str(msg.content))
                    if len(content) > 500:
                        content = content[:500] + "..."
                    await broadcast(node_name, "ToolResult", f"[工具结果] {content}")

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "DrugForge")
        print("[LangSmith] 追踪已启用", flush=True)
    else:
        print("[LangSmith] 未设置 LANGCHAIN_API_KEY，追踪关闭", flush=True)

    servers = {
        "druggen":     {"command": sys.executable, "args": ["druggen_mcp_server.py"],            "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "docking":     {"command": sys.executable, "args": ["docking_mcp_server.py"],            "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "chemical":    {"command": sys.executable, "args": ["chemical_properties_mcp_sever.py"], "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "admet":       {"command": sys.executable, "args": ["admet_prediction_mcp_server.py"],   "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "name2smiles": {"command": sys.executable, "args": ["name2smiles_mcp_server.py"],        "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "mol_opt":     {"command": sys.executable, "args": ["mol_opt_mcp_server.py"],            "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "patient":     {"command": sys.executable, "args": ["patient_matching_mcp_server.py"],   "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "trialgen":    {"command": sys.executable, "args": ["trialgen_mcp_server.py"],           "transport": "stdio", "cwd": str(PROJECT_DIR)},
        "trialpred":   {"command": sys.executable, "args": ["trialpred_mcp_server.py"],          "transport": "stdio", "cwd": str(PROJECT_DIR)},
    }

    async with MultiServerMCPClient(servers) as client:
        all_tools = client.get_tools()
        tool_map = {t.name: t for t in all_tools}

        def select(*names):
            return [tool_map[n] for n in names if n in tool_map]

        agent_nodes = {
            "planning_agent": make_agent_node("planning_agent", create_react_agent(
                ds(thinking=True), [], prompt=PLANNING_SYS,
            )),
            "druggen_agent": make_agent_node("druggen_agent", create_react_agent(
                ds(), select("run_druggen"), prompt=DRUGGEN_SYS,
            )),
            "chemical_agent": make_agent_node("chemical_agent", create_react_agent(
                ds(),
                select("select_leads_from_smiles", "predict_pka_batch", "logd_acid_batch",
                       "logd_base_batch", "rdkit_physchem_batch", "predict_all_batch"),
                prompt=CHEM_PROPERTIES_SYS,
            )),
            "admet_properties_agent": make_agent_node("admet_properties_agent", create_react_agent(
                ds(),
                select("get_drug_name_from_smiles", "get_smiles_from_drug_name",
                       "chemfm_list_properties", "chemfm_get_description",
                       "chemfm_predict_single", "chemfm_predict_many", "run_docking"),
                prompt=ADMET_PROPERTIES_SYS,
            )),
            "molecule_optimization_agent": make_agent_node("molecule_optimization_agent", create_react_agent(
                ds(thinking=True), select("molecule_optimizer"), prompt=MOL_OPT_SYS,
            )),
            "trial_generator_agent": make_agent_node("trial_generator_agent", create_react_agent(
                ds(thinking=True), select("panacea_extract_components"), prompt=TRIAL_SYS,
            )),
            "patient_matching_agent": make_agent_node("patient_matching_agent", create_react_agent(
                ds(), select("match_patient_trial"), prompt=PATIENT_MATCHING_SYS,
            )),
            "trial_prediction_agent": make_agent_node("trial_prediction_agent", create_react_agent(
                ds(), [t for t in all_tools if t not in select(
                    "run_druggen", "select_leads_from_smiles", "predict_pka_batch",
                    "logd_acid_batch", "logd_base_batch", "rdkit_physchem_batch",
                    "predict_all_batch", "get_drug_name_from_smiles", "get_smiles_from_drug_name",
                    "chemfm_list_properties", "chemfm_get_description", "chemfm_predict_single",
                    "chemfm_predict_many", "run_docking", "molecule_optimizer",
                    "panacea_extract_components", "match_patient_trial",
                )], prompt=TRIAL_PRED_SYS,
            )),
        }

        graph = build_graph(agent_nodes, ds())

        task = input("请输入你的任务：").strip() or "模拟DPP4(P27487)的药物开发"

        global _run
        _run = RunState(task=task)
        run: RunState = _run
        print(f"Run ID: {run.run_id}", flush=True)

        threading.Thread(target=start_viz_server, daemon=True).start()
        print("\n可视化面板：http://localhost:8765", flush=True)
        print("按回车开始运行...", flush=True)
        input()

        try:
            await run_with_viz(graph, task)
            _validate_output()
            run.done()
        except Exception as e:
            print(f"运行出错：{e}", flush=True)
            run.failed(str(e))
        finally:
            print(f"\n运行完成  run_id={run.run_id}", flush=True)
            print("服务器保持运行，可刷新浏览器查看结果", flush=True)
            while True:
                await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
