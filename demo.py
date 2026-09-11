"""Offline engineering demo using the production graph and deterministic stub nodes."""
import argparse
import asyncio
import html
import json
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from DrugForge import build_graph, build_opt_subgraph

NODE_NAMES = (
    "planning_start", "druggen_agent", "admet_docking", "chemical_filter",
    "admet_predict", "admet_final", "trial_generator_agent",
    "patient_matching_agent", "trial_prediction_agent", "planning_final",
)

class DemoEvaluator:
    async def ainvoke(self, messages):
        return AIMessage(content="continue")

def demo_node(name):
    async def node(state):
        content = f"[DEMO / 模拟输出] {name} completed; no scientific inference performed."
        if name == "planning_final":
            content += "\n工程流程已结束。此演示不生成药效、毒性或临床成功率结论。\nFINAL"
        return {"messages": [AIMessage(content=content, name=name)]}
    return node

def make_demo_graph(evaluator=None):
    nodes = {name: demo_node(name) for name in NODE_NAMES}
    opt = build_opt_subgraph(
        {name: demo_node(name) for name in ("mol_opt_agent", "admet_reeval", "chem_reeval")},
        evaluator or DemoEvaluator(),
    )
    return build_graph(nodes, opt)

async def run_demo(has_xml=False):
    graph = make_demo_graph()
    config = {"configurable": {"thread_id": "offline-demo"}, "recursion_limit": 30}
    initial = {"messages": [HumanMessage(content="Offline workflow demonstration")],
               "opt_count": 0, "opt_satisfied": False, "has_xml": has_xml}
    events = []
    confirmations = 0
    value = initial
    while True:
        async for namespace, update in graph.astream(value, config, stream_mode="updates", subgraphs=True):
            for name, output in update.items():
                if name == "__interrupt__" or not isinstance(output, dict):
                    continue
                events.append({"node": name, "namespace": list(namespace),
                               "messages": [str(m.content) for m in output.get("messages", [])]})
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        confirmations += 1
        if confirmations > 3:
            raise RuntimeError("Demo exceeded the three-confirmation limit")
        events.append({"node": "demo_confirmation", "namespace": [],
                       "messages": ["DEMO: automatic approval; production waits for user confirmation."]})
        value = None
    return {"mode": "offline_demo", "scientific_results": False,
            "has_xml": has_xml, "confirmations": confirmations,
            "optimization_rounds": snapshot.values["opt_count"], "events": events}

def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = "".join(f'<article><h2>{html.escape(e["node"])}</h2><pre>{html.escape(chr(10).join(e["messages"]))}</pre></article>' for e in report["events"])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DrugForge · Offline demo</title>
<style>body{font:16px system-ui;background:#101827;color:#e5edf7;max-width:960px;margin:48px auto;padding:0 24px}h1{font-size:38px}p{line-height:1.8;color:#b5c6dd}article{background:#1b293d;border:1px solid #34465f;border-radius:12px;padding:12px 24px;margin:16px 0}h2{font-size:18px;color:#79dec8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 monospace}</style>
<h1>DrugForge · Agent 工作流演示</h1><p>真实 LangGraph 编排 · 模拟节点输出 · 无需 API Key / GPU<br>仅验证流程执行，不包含科学推断。演示自动确认；真实运行由用户确认。</p>'''
    page += f'<p>优化轮次：{report["optimization_rounds"]} · 暂停 / 恢复：{report["confirmations"]} · 患者分支：{report["has_xml"]}</p>' + cards + '</html>'
    output.with_suffix('.html').write_text(page, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-patients', action='store_true', help='Exercise the synthetic patient branch')
    parser.add_argument('--output', type=Path, default=Path('runs/demo.json'))
    args = parser.parse_args()
    report = asyncio.run(run_demo(args.with_patients))
    save_report(report, args.output)
    print(f'DEMO complete: {report["optimization_rounds"]} optimization rounds; {report["confirmations"]} resumes')
    print(f'JSON: {args.output}\nHTML: {args.output.with_suffix(".html")}')

if __name__ == '__main__':
    main()
