import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from persistence import AsyncSqliteSaver, run_lease, validate_resume
from demo import make_demo_graph, initial_demo

ROOT = Path(__file__).resolve().parents[1]

def test_resume_across_real_process_exit_without_repeating_completed_stages(tmp_path):
    env = {**os.environ, 'DRUGFORGE_RUNS_DIR': str(tmp_path / 'runs')}
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    subprocess.run([sys.executable, str(ROOT / 'demo.py'), '--persist', '--pause-after', '1', '--output', str(first)],
                   env=env, check=True, capture_output=True, timeout=20)
    paused = json.loads(first.read_text())
    assert paused['status'] == 'paused'
    run_id = paused['run_id']
    subprocess.run([sys.executable, str(ROOT / 'demo.py'), '--resume', run_id, '--output', str(second)],
                   env=env, check=True, capture_output=True, timeout=20)
    resumed = json.loads(second.read_text())
    assert resumed['status'] == 'done'
    stages = [r['stage'] for r in resumed['attempts'].values()]
    assert stages.count('druggen_agent') == 1
    assert stages.count('mol_opt_agent') == 3
    assert resumed['resume_count'] == 1
    assert resumed['optimization_rounds'] == 3

def test_failed_node_retries_after_reopening_database(tmp_path):
    async def run():
        db = str(tmp_path / 'run.sqlite')
        config = {'configurable': {'thread_id': 'failure-test'}, 'recursion_limit': 30}
        attempts = {}
        sink = lambda r: attempts.update({r['id']: r})
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = make_demo_graph(checkpointer=saver, record_sink=sink, scenario='tool_error')
            with pytest.raises(RuntimeError, match='outage'):
                await graph.ainvoke(initial_demo(False), config)
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = make_demo_graph(checkpointer=saver, record_sink=sink)
            for _ in range(5):
                await graph.ainvoke(None, config)
                state = await graph.aget_state(config)
                if not state.next: break
            assert not state.next
        stages = [r['stage'] for r in attempts.values()]
        assert stages.count('druggen_agent') == 1
        assert stages.count('admet_docking') == 2
    asyncio.run(run())

def test_exclusive_lease_releases_after_close(tmp_path):
    path = tmp_path / 'run.sqlite'
    with run_lease(path):
        with pytest.raises(RuntimeError, match='another process'):
            with run_lease(path): pass
    with run_lease(path): pass

def test_legacy_or_wrong_mode_run_cannot_resume(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(state, 'RUNS_DIR', tmp_path)
    run = state.RunState('old run')
    with pytest.raises(ValueError, match='version'):
        validate_resume(run, 'live')

def test_live_resume_waits_for_fresh_human_confirmation(tmp_path, monkeypatch):
    import DrugForge as app
    monkeypatch.setattr(app, '_run', None)
    monkeypatch.setattr(app, '_pending_approval', None)
    monkeypatch.setattr(app, 'messages_store', [])
    monkeypatch.setattr(app, 'mem0_save', lambda *args: None)
    async def run():
        db = str(tmp_path / 'approval.sqlite')
        config = {'configurable': {'thread_id': 'live-resume'}, 'recursion_limit': 30}
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = make_demo_graph(checkpointer=saver)
            await graph.ainvoke(initial_demo(False), config)
        attempts = []
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            graph = make_demo_graph(checkpointer=saver, record_sink=attempts.append)
            monkeypatch.setattr(app, '_confirm_event', asyncio.Event())
            task = asyncio.create_task(app.run_with_viz(graph, 'demo', False, config, resume=True))
            try:
                for _ in range(500):
                    if app._pending_approval: break
                    await asyncio.sleep(0.001)
                assert app._pending_approval['round'] == 1
                assert not attempts, 'No model stage may execute before fresh approval'
                while not task.done():
                    if app._pending_approval:
                        app._pending_approval = None
                        app._confirm_event.set()
                    await asyncio.sleep(0.001)
                await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            assert [r['stage'] for r in attempts].count('mol_opt_agent') == 3
    asyncio.run(asyncio.wait_for(run(), timeout=10))
