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
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from state import RunState

PROJECT_DIR = Path(__file__).parent

# ── 可视化服务器 ──────────────────────────────────────────────────────────────

messages_store: list = []
_run: RunState | None = None
_confirm_event: asyncio.Event | None = None
_main_loop: asyncio.AbstractEventLoop | None = None

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

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        if self.path == "/confirm":
            if _main_loop and _confirm_event:
                _main_loop.call_soon_threadsafe(_confirm_event.set)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

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
        if "FINAL" in msg["content"]:
            if len(msg["content"]) > 200:
                print("[验证] 最终报告完整", flush=True)
            else:
                print("[验证] 警告：包含 FINAL 但报告内容过短", flush=True)
            return
    print("[验证] 警告：未检测到 FINAL，流程可能提前终止", flush=True)

# ── 状态定义 ──────────────────────────────────────────────────────────────────

class DrugForgeState(TypedDict):
    messages: Annotated[list, add_messages]
    opt_count: int       # 已完成的优化轮次
    opt_satisfied: bool  # 当前优化是否满足条件
    has_xml: bool        # 是否提供患者 XML 路径

# OptState 与 DrugForgeState 共享字段，供优化子图使用
class OptState(TypedDict):
    messages: Annotated[list, add_messages]
    opt_count: int
    opt_satisfied: bool

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

PLANNING_START_SYS = """你是规划agent。分析用户的药物研发任务，制定总体研发策略，
说明将要经历的阶段（分子生成→筛选→优化→临床），以及针对该靶点的具体研发重点。
不要自己执行任何工具，只输出研发计划。"""

PLANNING_FINAL_SYS = """你是规划agent。整合所有agent的输出结果，写最终药物研发报告。

报告格式：
药物发现性质：
- 命中物：[初始SMILES列表]
- 先导化合物：[带性质的已选先导化合物]
- 优化先导化合物：带优化性质的SMILES
- 化学性质：[理化描述符]
- ADMET性质：[预测结果]
- 对接评分：[评分]

临床试验报告：[完整结构化试验]

患者匹配：[结果（如有）]

试验成功概率：[概率]

总结：[整体评价]

报告完成后在最后单独一行写：FINAL"""

DRUGGEN_SYS = """你是药物生成专家。根据给定的 UniProt ID 生成候选分子 SMILES。
始终生成7个分子，只运行 run_druggen 工具并返回结果，不做任何其他操作。"""

CHEM_PROPERTIES_SYS = """你是化学性质专家。预测并报告分子的理化性质，筛选先导化合物。
按 Lipinski（MW≤500, logP≤5, HBD≤5, HBA≤10）和 Veber（RotB≤10, TPSA≤140）规则筛选。
工具：select_leads_from_smiles, rdkit_physchem_batch, predict_pka_batch,
      logd_acid_batch, logd_base_batch, predict_all_batch。
不做 ADMET 预测、对接或临床试验生成。"""

ADMET_PROPERTIES_SYS = """你是 ADMET 性质专家。预测分子的 ADMET 性质和对接评分。
始终先运行 chemfm_list_properties 获取性质名称，再预测10个最相关性质
（口服生物利用度、溶解度、清除率、BBB通透性、hERG抑制、肝毒性等）。
工具：chemfm_list_properties, chemfm_get_description, chemfm_predict_single,
      chemfm_predict_many, run_docking, get_drug_name_from_smiles, get_smiles_from_drug_name。
不做临床试验生成或分子生成。"""

MOL_OPT_SYS = """你是分子优化专家。针对性质薄弱点优化先导化合物。
工具：molecule_optimizer(smiles, properties, action)，action 为 increase 或 decrease。
只在已有 ADMET 和化学性质评估结果后才运行，只返回工具结果。"""

TRIAL_SYS = """你是临床试验设计专家。为最终先导化合物生成完整临床试验方案。
先输出初始试验文本，再调用 panacea_extract_components 工具生成结构化组件，
最后输出包含 arms、outcomes、eligibility_criteria 等完整字段的结构化试验报告。"""

PATIENT_MATCHING_SYS = """你是患者匹配专家。
工具：match_patient_trial(xml_path, trial_text)。
使用 trial_generator_agent 生成的精确试验文本，返回匹配患者数量和 ID 列表。"""

TRIAL_PRED_SYS = """你是试验预测专家。根据结构化试验文本预测临床试验成功概率（0-1）。
只返回概率数值，不做其他操作。"""

# ── Agent 节点工厂 ────────────────────────────────────────────────────────────

def make_agent_node(display_name: str, agent):
    async def node(state) -> dict:
        result = await agent.ainvoke({"messages": state["messages"]})
        new_msgs = result["messages"][len(state["messages"]):]
        return {"messages": new_msgs}
    node.__name__ = display_name
    return node

# ── 优化子图 ──────────────────────────────────────────────────────────────────

def build_opt_subgraph(opt_nodes: dict, eval_llm: ChatOpenAI):
    """把 mol_opt → admet_reeval → chem_reeval → eval_opt 循环提取为独立子图。

    子图不带 checkpointer（由父图的 MemorySaver 统一管理），
    interrupt_before=["mol_opt_agent"] 会通过父图的 checkpointer 向上传播，
    使父图在每次进入优化前暂停，等待浏览器确认按钮触发继续。
    """

    async def eval_opt(state: OptState) -> dict:
        resp = await eval_llm.ainvoke([
            SystemMessage(content="""根据最近的 ADMET 和化学性质评估结果，
判断当前优化的先导化合物是否同时满足：通过 Lipinski/Veber 规则、口服生物利用度>0.5、低毒性风险。
只回答 satisfied 或 continue，不要加任何其他内容。"""),
            *state["messages"][-12:],
        ])
        satisfied = "satisfied" in resp.content.lower()
        return {
            "opt_satisfied": satisfied,
            "opt_count": state.get("opt_count", 0) + 1,
        }

    def route_opt(state: OptState) -> str:
        if state["opt_satisfied"] or state.get("opt_count", 0) >= 3:
            return END
        return "mol_opt_agent"

    g = StateGraph(OptState)
    g.add_node("mol_opt_agent", opt_nodes["mol_opt_agent"])
    g.add_node("admet_reeval",  opt_nodes["admet_reeval"])
    g.add_node("chem_reeval",   opt_nodes["chem_reeval"])
    g.add_node("eval_opt",      eval_opt)

    g.set_entry_point("mol_opt_agent")
    g.add_edge("mol_opt_agent", "admet_reeval")
    g.add_edge("admet_reeval",  "chem_reeval")
    g.add_edge("chem_reeval",   "eval_opt")
    g.add_conditional_edges("eval_opt", route_opt, {
        "mol_opt_agent": "mol_opt_agent",
        END: END,
    })

    # interrupt_before 设在子图内，父图 checkpointer 会捕获并向上冒泡
    return g.compile(interrupt_before=["mol_opt_agent"])

# ── 图构建 ────────────────────────────────────────────────────────────────────

def build_graph(nodes: dict, opt_subgraph):

    def route_clinical(state: DrugForgeState) -> list[str]:
        if state.get("has_xml"):
            return ["patient_matching_agent", "trial_prediction_agent"]
        return ["trial_prediction_agent"]

    wf = StateGraph(DrugForgeState)

    for name, node in nodes.items():
        wf.add_node(name, node)
    wf.add_node("optimization_loop", opt_subgraph)

    # ── 固定主干流程 ──
    wf.set_entry_point("planning_start")
    wf.add_edge("planning_start",      "druggen_agent")
    wf.add_edge("druggen_agent",       "admet_docking")
    wf.add_edge("admet_docking",       "chemical_filter")
    wf.add_edge("chemical_filter",     "admet_predict")
    # ── 优化子图（子图内 interrupt_before=["mol_opt_agent"]）──
    wf.add_edge("admet_predict",       "optimization_loop")
    wf.add_edge("optimization_loop",   "admet_final")
    # ── 临床前 + 临床阶段 ──
    wf.add_edge("admet_final",         "trial_generator_agent")
    wf.add_conditional_edges("trial_generator_agent", route_clinical, {
        "patient_matching_agent": "patient_matching_agent",
        "trial_prediction_agent": "trial_prediction_agent",
    })
    wf.add_edge("patient_matching_agent", "planning_final")
    wf.add_edge("trial_prediction_agent", "planning_final")
    wf.add_edge("planning_final", END)

    checkpointer = MemorySaver()
    return wf.compile(checkpointer=checkpointer)

# ── 可视化流（支持中断恢复）────────────────────────────────────────────────────

async def _stream_to_viz(graph, input_val, config: dict):
    # subgraphs=True 使子图内部各节点的更新也能实时推送到可视化面板
    async for chunk in graph.astream(input_val, config=config, stream_mode="updates", subgraphs=True):
        _ns, update = chunk
        for node_name, output in update.items():
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

async def run_with_viz(graph, task: str, has_xml: bool, config: dict):
    initial = {
        "messages": [HumanMessage(content=task)],
        "opt_count": 0,
        "opt_satisfied": False,
        "has_xml": has_xml,
    }

    await _stream_to_viz(graph, initial, config)

    # 处理子图内的中断：每次 mol_opt_agent 前暂停，等浏览器确认按钮
    while True:
        state = graph.get_state(config)
        if not state.next:
            break
        opt_n = state.values.get("opt_count", 0)
        await broadcast(
            "system", "WaitingConfirm",
            f"[优化循环 {opt_n + 1}/3] 先导化合物已选出，请在可视化面板点击"确认"继续优化",
        )
        _confirm_event.clear()
        await _confirm_event.wait()
        await broadcast("system", "System", "继续优化中...")
        await _stream_to_viz(graph, None, config)

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global _confirm_event, _main_loop
    _confirm_event = asyncio.Event()
    _main_loop = asyncio.get_event_loop()

    sys.stdout.reconfigure(encoding="utf-8")

    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "DrugForge")
        print("[LangSmith] 追踪已启用", flush=True)

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

        planning_r   = create_react_agent(ds(thinking=True), [], prompt=PLANNING_START_SYS)
        planning_f_r = create_react_agent(ds(thinking=True), [], prompt=PLANNING_FINAL_SYS)
        druggen_r    = create_react_agent(ds(), select("run_druggen"), prompt=DRUGGEN_SYS)
        chem_r       = create_react_agent(ds(), select(
            "select_leads_from_smiles", "predict_pka_batch", "logd_acid_batch",
            "logd_base_batch", "rdkit_physchem_batch", "predict_all_batch",
        ), prompt=CHEM_PROPERTIES_SYS)
        admet_r      = create_react_agent(ds(), select(
            "get_drug_name_from_smiles", "get_smiles_from_drug_name",
            "chemfm_list_properties", "chemfm_get_description",
            "chemfm_predict_single", "chemfm_predict_many", "run_docking",
        ), prompt=ADMET_PROPERTIES_SYS)
        mol_opt_r    = create_react_agent(ds(thinking=True), select("molecule_optimizer"), prompt=MOL_OPT_SYS)
        trial_gen_r  = create_react_agent(ds(thinking=True), select("panacea_extract_components"), prompt=TRIAL_SYS)
        patient_r    = create_react_agent(ds(), select("match_patient_trial"), prompt=PATIENT_MATCHING_SYS)
        known = {
            "run_druggen", "select_leads_from_smiles", "predict_pka_batch", "logd_acid_batch",
            "logd_base_batch", "rdkit_physchem_batch", "predict_all_batch",
            "get_drug_name_from_smiles", "get_smiles_from_drug_name", "chemfm_list_properties",
            "chemfm_get_description", "chemfm_predict_single", "chemfm_predict_many", "run_docking",
            "molecule_optimizer", "panacea_extract_components", "match_patient_trial",
        }
        trial_pred_r = create_react_agent(
            ds(), [t for t in all_tools if t.name not in known], prompt=TRIAL_PRED_SYS
        )

        nodes = {
            "planning_start":         make_agent_node("planning_agent",              planning_r),
            "druggen_agent":          make_agent_node("druggen_agent",               druggen_r),
            "admet_docking":          make_agent_node("admet_properties_agent",      admet_r),
            "chemical_filter":        make_agent_node("chemical_agent",              chem_r),
            "admet_predict":          make_agent_node("admet_properties_agent",      admet_r),
            # 优化子图节点（在 build_opt_subgraph 内部注册）
            "admet_final":            make_agent_node("admet_properties_agent",      admet_r),
            "trial_generator_agent":  make_agent_node("trial_generator_agent",       trial_gen_r),
            "patient_matching_agent": make_agent_node("patient_matching_agent",      patient_r),
            "trial_prediction_agent": make_agent_node("trial_prediction_agent",      trial_pred_r),
            "planning_final":         make_agent_node("planning_agent",              planning_f_r),
        }

        opt_nodes = {
            "mol_opt_agent": make_agent_node("molecule_optimization_agent", mol_opt_r),
            "admet_reeval":  make_agent_node("admet_properties_agent",      admet_r),
            "chem_reeval":   make_agent_node("chemical_agent",              chem_r),
        }

        opt_subgraph = build_opt_subgraph(opt_nodes, ds())
        graph = build_graph(nodes, opt_subgraph)

        task = input("请输入你的任务：").strip() or "模拟DPP4(P27487)的药物开发"
        has_xml = ".xml" in task or "xml" in task.lower()

        global _run
        _run = RunState(task=task)
        run: RunState = _run
        print(f"Run ID: {run.run_id}", flush=True)

        config = {"configurable": {"thread_id": run.run_id}}

        threading.Thread(target=start_viz_server, daemon=True).start()
        print("\n可视化面板：http://localhost:8765", flush=True)
        print("按回车开始运行...", flush=True)
        input()

        try:
            await run_with_viz(graph, task, has_xml, config)
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
