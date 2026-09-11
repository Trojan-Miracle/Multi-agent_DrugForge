import argparse
import asyncio
from contextlib import AsyncExitStack
import secrets
import hashlib
import time
import uuid
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
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from state import RunState
from workflow_runtime import make_agent_node, merge_stage_results, content_text, guard_tool, usage_from_messages
from contracts import handoff
from persistence import AsyncSqliteSaver, run_lease, checkpoint_path, validate_resume, collect_attempts, waiting_for_approval, GRAPH_VERSION
from metrics import score_run
from reporting import write_report

PROJECT_DIR = Path(__file__).parent

# ── Mem0 跨会话记忆（可选，设置 MEM0_API_KEY 后自动启用）────────────────────

try:
    from mem0 import MemoryClient as _Mem0Client
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False

_mem0: object | None = None
MEM0_USER_ID = "drugforge-default"

def _get_mem0():
    global _mem0
    if not _MEM0_AVAILABLE:
        return None
    key = os.environ.get("MEM0_API_KEY", "")
    if not key:
        return None
    if _mem0 is None:
        _mem0 = _Mem0Client(api_key=key)
    return _mem0

def mem0_recall(task: str) -> str:
    """搜索与当前任务相关的历史偏好，返回可拼入 prompt 的字符串。"""
    client = _get_mem0()
    if not client:
        return ""
    try:
        hits = client.search(f"用户对药物研发任务的偏好：{task}", user_id=MEM0_USER_ID, limit=3)
        if not hits:
            return ""
        lines = [h["memory"] for h in hits if h.get("score", 0) > 0.3]
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""

def mem0_save(task: str, opt_direction: str):
    """在优化结束后记住用户本次任务和优化偏好。"""
    client = _get_mem0()
    if not client:
        return
    try:
        msgs = [
            {"role": "user",      "content": f"任务：{task}；本次优化方向：{opt_direction}"},
            {"role": "assistant", "content": "好的，已记录本次药物研发任务的偏好。"},
        ]
        client.add(msgs, user_id=MEM0_USER_ID)
    except Exception:
        pass

# ── 可视化服务器 ──────────────────────────────────────────────────────────────

messages_store: list = []
_run: RunState | None = None
_confirm_event: asyncio.Event | None = None
_main_loop: asyncio.AbstractEventLoop | None = None

_approval_lock = threading.Lock()
_pending_approval: dict | None = None

class VizHandler(BaseHTTPRequestHandler):
    def send_json(self, code, value):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            data = (PROJECT_DIR / "viz.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/messages":
            self.send_json(200, messages_store)
        elif self.path == "/status":
            with _approval_lock:
                approval = dict(_pending_approval) if _pending_approval else None
            self.send_json(200, {"status": _run.status if _run else "starting",
                                 "approval": approval})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        global _pending_approval
        if self.path != "/confirm":
            self.send_json(404, {"error": "Not found"})
            return
        origin = self.headers.get("Origin")
        port = self.server.server_port
        if origin and origin not in {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}:
            self.send_json(403, {"error": "Origin rejected"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected JSON object")
        except (ValueError, TypeError):
            self.send_json(400, {"error": "Invalid confirmation request"})
            return
        with _approval_lock:
            if (not _pending_approval or body.get("token") != _pending_approval["token"]
                    or not _main_loop or _main_loop.is_closed() or not _confirm_event):
                self.send_json(409, {"error": "Confirmation expired or no approval pending"})
                return
            _pending_approval = None
            _main_loop.call_soon_threadsafe(_confirm_event.set)
        self.send_json(200, {"ok": True})

    def log_message(self, format, *args):
        pass


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

def _validate_output(report=None):
    report = report if report is not None else next((m["content"] for m in reversed(messages_store)
                   if m["agent"] == "planning_final" and m["type"] == "AIMessage"), "")
    sections = ("药物发现性质", "临床试验报告", "患者匹配", "试验成功概率", "总结")
    if not report.strip().endswith("\nFINAL") or not all(x in report for x in sections):
        raise RuntimeError("最终报告缺少必要章节或独立 FINAL 结束标记")
    print("[验证] 报告格式检查通过（不代表科学结果已验证）", flush=True)
    return report

# ── 状态定义 ──────────────────────────────────────────────────────────────────

class DrugForgeState(TypedDict):
    messages: Annotated[list, add_messages]
    opt_count: int       # 已完成的优化轮次
    opt_satisfied: bool  # 当前优化是否满足条件
    has_xml: bool        # 是否提供患者 XML 路径
    stage_results: Annotated[dict, merge_stage_results]
    attempts: Annotated[dict, merge_stage_results]
    task: str
    target_id: str | None
    patients_path: str | None

# OptState 与 DrugForgeState 共享字段，供优化子图使用
class OptState(TypedDict):
    stage_results: Annotated[dict, merge_stage_results]
    attempts: Annotated[dict, merge_stage_results]
    task: str
    target_id: str | None
    patients_path: str | None
    messages: Annotated[list, add_messages]
    opt_count: int
    opt_satisfied: bool

# ── 模型工厂 ──────────────────────────────────────────────────────────────────

def ds(thinking: bool = False, pro: bool = False) -> ChatOpenAI:
    model = os.environ.get("DEEPSEEK_PRO_MODEL" if pro else "DEEPSEEK_MODEL",
                           "deepseek-v4-pro" if pro else "deepseek-flash")
    kw = {"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}
    return ChatOpenAI(
        model=model,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=90,
        max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")),
        max_retries=2,
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

ADMET_PROPERTIES_SYS = """你是 ADMET 性质专家。只预测分子的 ADMET 性质。
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
必须调用 predict_trial_success；工具失败时说明不可用，禁止自行推测概率。"""

# ── Agent 节点工厂 ────────────────────────────────────────────────────────────

# ── 优化子图 ──────────────────────────────────────────────────────────────────

def build_opt_subgraph(opt_nodes: dict, eval_llm: ChatOpenAI, record_sink=None):
    """把 mol_opt → admet_reeval → chem_reeval → eval_opt 循环提取为独立子图。

    子图不带 checkpointer（由父图的检查点后端统一管理），
    interrupt_before=["mol_opt_agent"] 会通过父图的 checkpointer 向上传播，
    使父图在每次进入优化前暂停，等待浏览器确认按钮触发继续。
    """

    async def eval_opt(state: OptState) -> dict:
        started = time.monotonic()
        record = {'id': uuid.uuid4().hex, 'stage': 'eval_opt', 'round': state.get('opt_count', 0) + 1,
                  'status': 'failed', 'data': {}, 'tool_results': [], 'llm_calls': []}
        try:
            resp = await asyncio.wait_for(eval_llm.ainvoke([
                SystemMessage(content="""根据结构化 ADMET 与理化筛选结果，判断是否同时通过 Lipinski/Veber、口服生物利用度>0.5、低毒性风险。
端点含义或单位未知、证据不足时回答 continue。只有明确达标才回答 satisfied。只输出一个词。"""),
                HumanMessage(content=json.dumps(handoff('admet_final', state), ensure_ascii=False)),
            ]), timeout=120)
            satisfied = isinstance(resp.content, str) and resp.content.strip().lower() == 'satisfied'
            record.update(status='completed', summary=str(resp.content), llm_calls=usage_from_messages([resp]))
        except BaseException as exc:
            record['error'] = str(exc) or type(exc).__name__
            raise
        finally:
            record['duration_seconds'] = time.monotonic() - started
            if record_sink: record_sink(record)
        return {'opt_satisfied': satisfied, 'opt_count': state.get('opt_count', 0) + 1,
                'attempts': {record['id']: record}}

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

def build_graph(nodes: dict, opt_subgraph, checkpointer=None):

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

    checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return wf.compile(checkpointer=checkpointer)

# ── 可视化流（支持中断恢复）────────────────────────────────────────────────────

async def _stream_to_viz(graph, input_val, config: dict):
    seen = set()
    # subgraphs=True 使子图内部各节点的更新也能实时推送到可视化面板
    async for chunk in graph.astream(input_val, config=config, stream_mode="updates", subgraphs=True):
        _ns, update = chunk
        for node_name, output in update.items():
            if node_name == "__interrupt__" or not isinstance(output, dict):
                continue
            if _run and output.get("stage_results"):
                _run.record_stages(output["stage_results"])
            for msg in output.get("messages", []):
                if msg.id and msg.id in seen:
                    continue
                if msg.id:
                    seen.add(msg.id)
                if isinstance(msg, AIMessage):
                    content = content_text(msg.content)
                    content = _unwrap_json_blocks(content)
                    if content.strip():
                        await broadcast(node_name, "AIMessage", content)
                elif isinstance(msg, ToolMessage):
                    content = _unwrap_json_blocks(content_text(msg.content))
                    await broadcast(node_name, "ToolResult", f"[工具结果] {content}")

async def run_with_viz(graph, task: str, has_xml: bool, config: dict, *, resume=False, target_id=None, patients_path=None):
    global _pending_approval
    # 从 Mem0 召回历史偏好，拼入首条消息给 planning_start 参考
    past = "" if resume else await asyncio.to_thread(mem0_recall, task)
    task_msg = task if not past else f"{task}\n\n[用户历史偏好]\n{past}"
    if past:
        print(f"[Mem0] 召回 {len(past.splitlines())} 条历史偏好", flush=True)

    initial = {
        "messages": [HumanMessage(content=task_msg)],
        "opt_count": 0,
        "opt_satisfied": False,
        "has_xml": has_xml,
        "stage_results": {},
        "attempts": {},
        "task": task_msg,
        "target_id": target_id,
        "patients_path": patients_path,
    }

    if not resume:
        await _stream_to_viz(graph, initial, config)
    else:
        snapshot = await graph.aget_state(config, subgraphs=True)
        if not snapshot.values:
            raise ValueError("No checkpoint state exists for this run")
        if _run: _run.reconcile(collect_attempts(snapshot))
        if snapshot.next and not waiting_for_approval(snapshot):
            await _stream_to_viz(graph, None, config)

    # 处理子图内的中断：每次 mol_opt_agent 前暂停，等浏览器确认按钮
    while True:
        state = await graph.aget_state(config, subgraphs=True)
        if _run: _run.reconcile(collect_attempts(state))
        if not state.next:
            break
        opt_n = state.values.get("opt_count", 0)
        for pending in state.tasks:
            nested = getattr(pending, "state", None)
            if hasattr(nested, "values"):
                opt_n = nested.values.get("opt_count", opt_n)
        _confirm_event.clear()
        with _approval_lock:
            _pending_approval = {"token": secrets.token_urlsafe(24), "round": opt_n + 1}
        await broadcast(
            "system", "WaitingConfirm",
            f"[优化循环 {opt_n + 1}/3] 先导化合物已选出，请在可视化面板点击“确认”继续优化",
        )
        await _confirm_event.wait()
        if _run:
            _run.data.setdefault("approvals", []).append({"round": opt_n + 1, "decision": "approved", "time": time.time()})
            _run._flush()
        # 用户确认后记录本轮优化偏好（异步保存，不阻塞主流程）
        await asyncio.to_thread(mem0_save, task, f"第{opt_n + 1}轮优化，用户确认继续")
        await broadcast("system", "System", "继续优化中...")
        await _stream_to_viz(graph, None, config)

# ── Main ──────────────────────────────────────────────────────────────────────

def build_live_graph(all_tools, checkpointer=None, record_sink=None):
    tool_map = {t.name: t for t in all_tools}
    if len(tool_map) != len(all_tools):
        raise RuntimeError("Duplicate MCP tool names")

    def select(*names):
        missing = sorted(set(names) - tool_map.keys())
        if missing:
            raise RuntimeError(f"Missing required MCP tools: {missing}")
        return [guard_tool(tool_map[n]) for n in names]

    def agent(prompt, names=(), thinking=False):
        return create_agent(ds(thinking=thinking), select(*names), system_prompt=prompt)

    planning = agent(PLANNING_START_SYS, thinking=True)
    final = agent(PLANNING_FINAL_SYS, thinking=True)
    druggen = agent(DRUGGEN_SYS, ("run_druggen",))
    chemical = agent(CHEM_PROPERTIES_SYS, (
        "select_leads_from_smiles", "predict_pka_batch", "logd_acid_batch",
        "logd_base_batch", "rdkit_physchem_batch", "predict_all_batch"))
    docking = agent("你负责分子对接初筛，只使用 run_docking，报告工具实际返回的评分和失败原因。", ("run_docking",))
    admet = agent(ADMET_PROPERTIES_SYS, (
        "chemfm_list_properties", "chemfm_get_description", "chemfm_predict_single", "chemfm_predict_many"))
    optimizer = agent(MOL_OPT_SYS, ("molecule_optimizer",), thinking=True)
    trial = agent(TRIAL_SYS, ("panacea_extract_components",), thinking=True)
    prediction = agent(TRIAL_PRED_SYS, ("predict_trial_success",))
    patient = agent(PATIENT_MATCHING_SYS, ("match_patient_trial",)) if "match_patient_trial" in tool_map else None

    async def no_patient(state):
        raise RuntimeError("Patient matching was not configured")

    agents = {"planning_start": planning, "druggen_agent": druggen, "admet_docking": docking,
              "chemical_filter": chemical, "admet_predict": admet, "admet_final": admet,
              "trial_generator_agent": trial, "trial_prediction_agent": prediction,
              "planning_final": final}
    nodes = {name: make_agent_node(name, worker, record_sink=record_sink) for name, worker in agents.items()}
    nodes["patient_matching_agent"] = make_agent_node("patient_matching_agent", patient, record_sink=record_sink) if patient else no_patient
    opt_nodes = {name: make_agent_node(name, worker, record_sink=record_sink) for name, worker in
                 {"mol_opt_agent": optimizer, "admet_reeval": admet, "chem_reeval": chemical}.items()}
    return build_graph(nodes, build_opt_subgraph(opt_nodes, ds(), record_sink=record_sink), checkpointer=checkpointer)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DrugForge live workflow; use demo.py for offline mode")
    parser.add_argument("--task", help="Task description; otherwise prompt interactively")
    parser.add_argument("--patients", type=Path, help="Explicit patient XML file (optional)")
    parser.add_argument("--resume", help="Resume a saved live run ID")
    parser.add_argument("--target", help="Explicit UniProt target ID")
    parser.add_argument("--prices", type=Path, help="Optional user-supplied LLM price JSON")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--exit-on-complete", action="store_true", help="Close the viewer after the run")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.resume and (args.task is not None or args.patients or args.target):
        parser.error("--resume restores the original task, target and patients; do not override them")
    if args.patients:
        args.patients = args.patients.expanduser().resolve()
        if not args.patients.is_file() or args.patients.suffix.lower() != ".xml":
            parser.error("--patients must point to an existing XML file")
    return args


def model_configuration():
    return {'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'),
            'base_url': os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')}


async def main(args):
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print('缺少 DEEPSEEK_API_KEY。无需密钥的演示请运行 python demo.py。', file=sys.stderr)
        return 1
    try:
        prices = json.loads(args.prices.read_text(encoding='utf-8')) if args.prices else None
        if prices is not None:
            score_run({'attempts': {}}, prices)
        if args.resume:
            run = RunState.reopen(args.resume)
            validate_resume(run, 'live')
        else:
            task = (args.task if args.task is not None else input('请输入你的任务：')).strip() or '模拟DPP4(P27487)的药物开发'
            configuration = {'mode': 'live', 'graph_version': GRAPH_VERSION, 'model': model_configuration(),
                             'patients_path': str(args.patients) if args.patients else None,
                             'patients_sha256': hashlib.sha256(args.patients.read_bytes()).hexdigest() if args.patients else None,
                             'target_id': args.target}
            run = RunState(task, configuration=configuration)
        with run_lease(checkpoint_path(run)):
            if args.resume:
                run = RunState.reopen(args.resume)
                validate_resume(run, 'live')
                configuration = run.data['configuration']
                if configuration['model'] != model_configuration():
                    raise ValueError('Restore the original model configuration before resuming')
                if configuration['patients_path']:
                    patient = Path(configuration['patients_path'])
                    if hashlib.sha256(patient.read_bytes()).hexdigest() != configuration['patients_sha256']:
                        raise ValueError('Patient file changed since this run started')
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path(run))) as saver:
                config = {'configurable': {'thread_id': run.run_id}}
                if args.resume:
                    if not await saver.aget_tuple(config):
                        raise ValueError('Checkpoint is empty; cannot resume')
                    run.resume()
                if prices is not None:
                    run.data['price_schedule'] = prices
                    run._flush()
                return await execute_live(args, run, saver)
    except Exception as exc:
        print(f'无法启动或恢复：{exc}', file=sys.stderr)
        return 1


async def execute_live(args, run, saver):
    global _confirm_event, _main_loop, _run, _pending_approval
    _confirm_event = asyncio.Event()
    _main_loop = asyncio.get_running_loop()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "DrugForge")
    task = run.data['task']
    configuration = run.data['configuration']
    patients_path = configuration['patients_path']
    _run = run
    messages_store[:] = run.messages
    server = None
    try:
        server = HTTPServer(("127.0.0.1", args.port), VizHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"运行记录：{_run.run_id}\n可视化：http://localhost:{args.port}", flush=True)
        await broadcast("system", "System", "正在连接模型工具，请稍候…")
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

        if not patients_path:
            servers.pop("patient")
        # Explicit sessions keep model processes alive across calls and close them after the run.
        client = MultiServerMCPClient(servers)
        async with AsyncExitStack() as stack:
            all_tools = []
            for name in servers:
                await broadcast("system", "System", f"连接工具服务：{name}")
                session = await stack.enter_async_context(client.session(name))
                all_tools.extend(await load_mcp_tools(session))
            graph = build_live_graph(all_tools, checkpointer=saver, record_sink=run.record_attempt)
            config = {"configurable": {"thread_id": _run.run_id}, "recursion_limit": 30}
            await run_with_viz(graph, task, bool(patients_path), config, resume=bool(args.resume),
                               target_id=configuration['target_id'], patients_path=patients_path)
            snapshot = await graph.aget_state(config)
            run.reconcile(snapshot.values.get('attempts', {}))
            run.record_stages(snapshot.values.get('stage_results', {}))
            run.data['committed_attempts'] = snapshot.values.get('attempts', {})
            draft = _validate_output(run.data['stage_results']['planning_final']['summary'])
            run.path.with_suffix('.draft.md').write_text(draft, encoding='utf-8')
        _run.done()
        run.data['metrics'] = score_run(run.data)
        write_report(run.data, run.path.with_suffix('.md'))
        run._flush()
        await broadcast("system", "System", "流程已完成，报告和运行记录已保存。")
        if not args.exit_on_complete:
            print("面板保持可用，按 Ctrl+C 退出。", flush=True)
            await asyncio.Event().wait()
        return 0
    except asyncio.CancelledError:
        if _run.status == "running":
            _run.pause()
        raise
    except Exception as exc:
        _run.failed(str(exc))
        run.data['metrics'] = score_run(run.data)
        run._flush()
        await broadcast("system", "Error", f"运行失败：{exc}")
        print(f"运行失败：{exc}\n详情：{_run.path}", file=sys.stderr, flush=True)
        return 1
    finally:
        with _approval_lock:
            _pending_approval = None
        if server:
            await asyncio.to_thread(server.shutdown)
            server.server_close()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse_args())))
    except KeyboardInterrupt:
        print("\n已退出。")
        sys.exit(130)
