"""Offline engineering demo using the production graph and deterministic stub nodes."""
import argparse
import asyncio
import html
import json
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from DrugForge import build_graph, build_opt_subgraph

from fixtures import NODES as NODE_NAMES, FixtureAgent
from workflow_runtime import make_agent_node
from metrics import score_run
from reporting import write_report
from persistence import AsyncSqliteSaver, run_lease, GRAPH_VERSION, checkpoint_path, validate_resume, collect_attempts
from state import RunState
import time

class DemoEvaluator:
    async def ainvoke(self, messages):
        return AIMessage(content="continue")


def make_demo_graph(evaluator=None, checkpointer=None, record_sink=None, scenario='success'):
    nodes = {name: make_agent_node(name, FixtureAgent(name, scenario), record_sink=record_sink) for name in NODE_NAMES}
    opt = build_opt_subgraph(
        {name: make_agent_node(name, FixtureAgent(name, scenario), record_sink=record_sink)
         for name in ("mol_opt_agent", "admet_reeval", "chem_reeval")},
        evaluator or DemoEvaluator(), record_sink=record_sink,
    )
    return build_graph(nodes, opt, checkpointer=checkpointer)


async def execute_demo(graph, config, initial, *, stop_after_pause=None, events=None):
    events = events if events is not None else []
    confirmations = 0
    value = initial
    while True:
        if value is not None or confirmations:
            async for namespace, update in graph.astream(value, config, stream_mode="updates", subgraphs=True):
                for name, output in update.items():
                    if name == '__interrupt__' or not isinstance(output, dict):
                        continue
                    events.append({'node': name, 'namespace': list(namespace),
                                   'messages': [str(m.content) for m in output.get('messages', [])]})
        snapshot = await graph.aget_state(config, subgraphs=True)
        if not snapshot.next:
            return snapshot, events, confirmations, False
        if stop_after_pause and confirmations + 1 >= stop_after_pause:
            return snapshot, events, confirmations, True
        confirmations += 1
        if confirmations > 3:
            raise RuntimeError('Demo exceeded three confirmations')
        events.append({'node': 'demo_confirmation', 'namespace': [],
                       'messages': ['DEMO: automatic approval; production requires a human.']})
        value = None


def initial_demo(has_xml):
    return {'messages': [HumanMessage(content='Offline workflow demonstration')],
            'task': 'Offline workflow demonstration', 'target_id': 'P27487',
            'patients_path': 'demo-patients.xml' if has_xml else None,
            'opt_count': 0, 'opt_satisfied': False, 'has_xml': has_xml, 'attempts': {}, 'stage_results': {}}


async def run_demo(has_xml=False, scenario='success', prices=None, reference=None):
    started = time.monotonic()
    attempts = {}
    graph = make_demo_graph(record_sink=lambda r: attempts.update({r['id']: r}), scenario=scenario)
    config = {'configurable': {'thread_id': 'offline-demo'}, 'recursion_limit': 30}
    events = []
    report = {'run_id': 'offline-demo', 'mode': 'offline_demo', 'scientific_results': False,
              'has_xml': has_xml, 'status': 'running', 'attempts': attempts, 'events': events, 'price_schedule': prices}
    try:
        snapshot, events, confirmations, _ = await execute_demo(graph, config, initial_demo(has_xml), events=events)
        report.update(status='done', confirmations=confirmations, optimization_rounds=snapshot.values['opt_count'],
                      stage_results=snapshot.values.get('stage_results', {}), committed_attempts=snapshot.values.get('attempts', {}))
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        report['stage_results'] = {r['stage']: r for r in attempts.values() if r['status'] in {'completed', 'unavailable'}}
    report['active_seconds'] = time.monotonic() - started
    report['metrics'] = score_run(report, prices, reference)
    return report


async def persistent_demo(args):
    if args.resume:
        run = RunState.reopen(args.resume)
        validate_resume(run, 'offline_demo')
    else:
        run = RunState('Offline workflow demonstration', configuration={
            'mode': 'offline_demo', 'graph_version': GRAPH_VERSION, 'has_xml': args.with_patients})
    with run_lease(checkpoint_path(run)):
        if args.resume:
            run = RunState.reopen(args.resume)
            run.resume()
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path(run))) as saver:
            graph = make_demo_graph(checkpointer=saver, record_sink=run.record_attempt)
            config = {'configurable': {'thread_id': run.run_id}, 'recursion_limit': 30}
            if args.resume and not (await graph.aget_state(config)).values:
                raise ValueError('Checkpoint is empty; cannot resume')
            try:
                snapshot, events, count, paused = await execute_demo(graph, config,
                    None if args.resume else initial_demo(run.data['configuration']['has_xml']),
                    stop_after_pause=args.pause_after)
                run.reconcile(collect_attempts(snapshot))
                run.data['events'] = [*run.data.get('events', []), *events]
                run.data['mode'] = 'offline_demo'
                run.data['has_xml'] = run.data['configuration']['has_xml']
                run.data['confirmations'] = run.data.get('confirmations', 0) + count
                run.data['optimization_rounds'] = snapshot.values.get('opt_count', 0)
                if paused:
                    run.pause()
                    print(f'Paused. Resume in a new process: python demo.py --resume {run.run_id}')
                else:
                    run.data['committed_attempts'] = snapshot.values.get('attempts', {})
                    run.done()
                run.data['metrics'] = score_run(run.data)
                run._flush()
                save_report(run.data, args.output)
            except BaseException:
                if run.status == 'running': run.pause()
                raise
    return run.data


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report, output.with_suffix('.md'))
    cards = "".join(f'<article><h2>{html.escape(e["node"])}</h2><pre>{html.escape(chr(10).join(e["messages"]))}</pre></article>' for e in report["events"])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DrugForge · Offline demo</title>
<style>body{font:16px system-ui;background:#101827;color:#e5edf7;max-width:960px;margin:48px auto;padding:0 24px}h1{font-size:38px}p{line-height:1.8;color:#b5c6dd}article{background:#1b293d;border:1px solid #34465f;border-radius:12px;padding:12px 24px;margin:16px 0}h2{font-size:18px;color:#79dec8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 monospace}</style>
<h1>DrugForge · Agent 工作流演示</h1><p>真实 LangGraph 编排 · 模拟节点输出 · 无需 API Key / GPU<br>仅验证流程执行，不包含科学推断。演示自动确认；真实运行由用户确认。</p>'''
    page += f'<p>状态：{html.escape(report["status"])} · 优化轮次：{report.get("optimization_rounds", 0)} · 暂停 / 恢复：{report.get("confirmations", 0)} · 患者分支：{report["has_xml"]}</p><p><a style="color:#79dec8" href="{html.escape(output.with_suffix(".evidence.html").name)}">查看工具输入、输出与证据来源</a></p>' + cards + '</html>'
    output.with_suffix('.html').write_text(page, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-patients', action='store_true', help='Exercise the synthetic patient branch')
    parser.add_argument('--output', type=Path, default=Path('runs/demo.json'))
    parser.add_argument('--persist', action='store_true', help='Use disk checkpoints')
    parser.add_argument('--pause-after', type=int, choices=[1, 2, 3], help='Stop at an optimization approval boundary')
    parser.add_argument('--resume', help='Resume an offline run ID')
    args = parser.parse_args()
    if args.pause_after and not (args.persist or args.resume):
        parser.error('--pause-after requires --persist or --resume')
    report = asyncio.run(persistent_demo(args)) if args.persist or args.resume else asyncio.run(run_demo(args.with_patients))
    save_report(report, args.output)
    print(f'DEMO complete: {report.get("optimization_rounds", 0)} optimization rounds; {report.get("confirmations", 0)} resumes')
    print(f'JSON: {args.output}\nHTML: {args.output.with_suffix(".html")}')

if __name__ == '__main__':
    main()
