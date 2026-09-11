import ast
import asyncio
import json
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage
import DrugForge as app
from demo import run_demo, make_demo_graph

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('path', sorted(ROOT.glob('*.py')), ids=lambda p: p.name)
def test_python_sources_parse(path):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

@pytest.mark.parametrize('has_xml', [False, True])
def test_workflow_interrupts_limit_and_join(has_xml):
    report = asyncio.run(run_demo(has_xml))
    names = [e['node'] for e in report['events']]
    assert report['optimization_rounds'] == report['confirmations'] == 3
    assert names.count('mol_opt_agent') == 3
    assert names.count('planning_final') == 1
    assert ('patient_matching_agent' in names) == has_xml
    assert names.index('trial_prediction_agent') < names.index('planning_final')
    if has_xml:
        assert names.index('patient_matching_agent') < names.index('planning_final')

@pytest.mark.parametrize('answer,rounds', [('satisfied', 1), (' SATISFIED\n', 1), ('not satisfied', 3), ('unsatisfied', 3)])
def test_optimization_accepts_only_exact_satisfied(answer, rounds):
    class Evaluator:
        async def ainvoke(self, messages):
            return AIMessage(content=answer)
    async def run():
        graph = make_demo_graph(Evaluator())
        config = {'configurable': {'thread_id': 'test'}, 'recursion_limit': 30}
        value = {'messages': [], 'has_xml': False, 'opt_count': 0, 'opt_satisfied': False}
        for _ in range(5):
            await graph.ainvoke(value, config)
            state = graph.get_state(config)
            if not state.next:
                return state.values['opt_count']
            value = None
        pytest.fail('Workflow failed to terminate')
    assert asyncio.run(run()) == rounds

def test_interrupt_stream_is_not_treated_as_node_output(monkeypatch):
    class Graph:
        async def astream(self, *args, **kwargs):
            yield (), {'__interrupt__': ()}
            yield (), {'planning_final': {'messages': [AIMessage(content='done')]}}
    monkeypatch.setattr(app, 'messages_store', [])
    monkeypatch.setattr(app, '_run', None)
    asyncio.run(app._stream_to_viz(Graph(), None, {}))
    assert app.messages_store[0]['content'] == 'done'

def test_environment_checks_match_repository(monkeypatch):
    import check_env
    assert check_env.check_mcp_files()['ok']
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-secret-key')
    assert 'test-secret' not in json.dumps(check_env.check_api_key())

def test_optimizer_validation_and_error(monkeypatch):
    import mol_opt_mcp_server as opt
    def fail(prompt):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(opt, '_optimize', fail)
    assert 'action must' in asyncio.run(opt.molecule_optimizer('CCO', 'solubility', 'invalid'))
    assert 'model unavailable' in asyncio.run(opt.molecule_optimizer('CCO', 'solubility', 'increase'))

def test_live_confirmation_uses_each_subgraph_round(monkeypatch):
    monkeypatch.setattr(app, '_run', None)
    monkeypatch.setattr(app, 'messages_store', [])
    monkeypatch.setattr(app, '_pending_approval', None)
    monkeypatch.setattr(app, 'mem0_recall', lambda task: '')
    monkeypatch.setattr(app, 'mem0_save', lambda *args: None)
    async def run():
        monkeypatch.setattr(app, '_confirm_event', asyncio.Event())
        graph = make_demo_graph()
        config = {'configurable': {'thread_id': 'live-confirm'}, 'recursion_limit': 30}
        task = asyncio.create_task(app.run_with_viz(graph, 'demo', False, config))
        rounds = []
        last_token = None
        try:
            while not task.done():
                pending = app._pending_approval
                if pending and pending['token'] != last_token:
                    last_token = pending['token']
                    rounds.append(pending['round'])
                    app._pending_approval = None
                    app._confirm_event.set()
                await asyncio.sleep(0.001)
            await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert rounds == [1, 2, 3]
    asyncio.run(asyncio.wait_for(run(), timeout=5))
