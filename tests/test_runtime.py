import asyncio
import json
import sys
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from workflow_runtime import make_agent_node, contains_error

class Agent:
    def __init__(self, messages):
        self.messages = messages
    async def ainvoke(self, inputs, config):
        self.inputs = inputs
        self.config = config
        return {'messages': [*inputs['messages'], *self.messages]}

def tool(content, name='run_docking', status='success'):
    return ToolMessage(content=content, name=name, tool_call_id='call-1', status=status)

def test_stage_requires_real_tool_evidence():
    agent = Agent([AIMessage(content='I performed docking')])
    with pytest.raises(RuntimeError, match='no successful'):
        asyncio.run(make_agent_node('admet_docking', agent)({'messages': []}))

@pytest.mark.parametrize('payload', ['{"error":"offline"}', '{"ok":false}', '{"results":[{"error":"bad molecule"}]}'])
def test_stage_rejects_embedded_tool_errors(payload):
    agent = Agent([tool(payload), AIMessage(content='looks good')])
    with pytest.raises(RuntimeError, match='no successful'):
        asyncio.run(make_agent_node('admet_docking', agent)({'messages': []}))

def test_stage_records_evidence_and_specific_instruction():
    agent = Agent([tool('{"score":-7}'), AIMessage(content='tool estimate')])
    result = asyncio.run(make_agent_node('admet_docking', agent)({'messages': []}))
    assert '仅对当前候选分子' in agent.inputs['messages'][-1].content
    assert agent.config['recursion_limit'] == 24
    assert result['stage_results']['admet_docking']['tool_results'][0]['call_id'] == 'call-1'
    assert len(result['messages']) == 2

def test_optional_prediction_cannot_turn_error_into_probability():
    agent = Agent([tool('{"error":"missing checkpoint"}', 'predict_trial_success'), AIMessage(content='95% success')])
    result = asyncio.run(make_agent_node('trial_prediction_agent', agent)({'messages': []}))
    assert result['stage_results']['trial_prediction_agent']['status'] == 'unavailable'
    assert '95%' not in result['messages'][-1].content
    assert '不可用' in result['messages'][-1].content

def test_stage_timeout():
    class Slow:
        async def ainvoke(self, inputs, config):
            await asyncio.sleep(10)
    with pytest.raises(RuntimeError, match='stage timeout'):
        asyncio.run(make_agent_node('planning_start', Slow(), timeout=0.01)({'messages': []}))

def test_missing_trial_checkpoint_does_not_load_models(monkeypatch):
    import trialpred_mcp_server as trial
    monkeypatch.delenv('MEDITAB_MODEL_PATH', raising=False)
    assert 'error' in json.loads(trial.predict_trial_success('test'))

def test_report_requires_final_agent_and_sections(monkeypatch):
    import DrugForge as app
    monkeypatch.setattr(app, 'messages_store', [{'agent': 'druggen_agent', 'type': 'AIMessage', 'content': 'x' * 300 + '\nFINAL'}])
    with pytest.raises(RuntimeError):
        app._validate_output()
    report = '药物发现性质\n临床试验报告\n患者匹配\n试验成功概率\n总结\nFINAL'
    app.messages_store.append({'agent': 'planning_final', 'type': 'AIMessage', 'content': report})
    assert app._validate_output() == report

def test_atomic_run_storage_and_invalid_ids(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(state, 'RUNS_DIR', tmp_path)
    run = state.RunState('test', 'test-run')
    run.record_stages({'stage': {'status': 'completed'}})
    run.append('stage', 'ToolResult', 'x' * 2000)
    run.done()
    data = state.RunState.load('test-run')
    assert data['status'] == 'done'
    assert len(data['messages'][0]['content']) == 2000
    assert data['stage_results']['stage']['status'] == 'completed'
    assert not list(tmp_path.glob('*.tmp'))
    with pytest.raises(FileExistsError):
        state.RunState('duplicate', 'test-run')
    with pytest.raises(ValueError):
        state.RunState.load('../outside')

def test_atomic_write_failure_preserves_previous_log(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(state, 'RUNS_DIR', tmp_path)
    run = state.RunState('test', 'test-run')
    before = run.path.read_bytes()
    def fail(*args):
        raise OSError('disk error')
    monkeypatch.setattr(state.os, 'replace', fail)
    with pytest.raises(OSError):
        run.append('stage', 'AIMessage', 'new')
    assert run.path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))

def test_explicit_mcp_session_keeps_process_state():
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    async def run():
        server = Path(__file__).parent / 'fixtures/mcp_counter.py'
        client = MultiServerMCPClient({'counter': {'command': sys.executable, 'args': [str(server)], 'transport': 'stdio'}})
        async with client.session('counter') as session:
            tools = await load_mcp_tools(session)
            assert await tools[0].ainvoke({}) == '1'
            assert await tools[0].ainvoke({}) == '2'
    asyncio.run(asyncio.wait_for(run(), timeout=15))
