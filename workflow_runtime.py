"""Stage boundaries and execution evidence shared by the live workflow."""
import asyncio
import json
import time
import uuid
import hashlib
from contextvars import ContextVar
from langchain_core.tools import StructuredTool
from contracts import handoff, decode_payload, validate_arguments, normalize, merge_data
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

STAGE_TASKS = {
    'planning_start': '制定研发计划，明确用户提供的靶点与缺失信息；不执行实验或捏造结果。',
    'druggen_agent': '根据用户提供的 UniProt ID 调用 run_druggen，num_generated=7；输出实际生成的 SMILES。',
    'admet_docking': '仅对当前候选分子调用 run_docking 做对接初筛，保留评分和单位；本阶段不要预测 ADMET。',
    'chemical_filter': '对当前候选调用 select_leads_from_smiles，报告实际筛选结果和未通过原因。',
    'admet_predict': '对筛选后的候选进行 ADMET 预测，保留工具返回的端点名称、单位及原始数值。',
    'mol_opt_agent': '根据最近的性质结果调用 molecule_optimizer 优化当前先导分子；保留优化前后的 SMILES。',
    'admet_reeval': '仅对本轮新优化的分子重新预测 ADMET，不要复用旧分子的结果。',
    'chem_reeval': '对本轮新优化的分子重新执行 select_leads_from_smiles，报告规则筛选结果。',
    'admet_final': '对最终分子进行 ADMET 复核，注明预测端点及不确定性，不把预测写成实验结论。',
    'trial_generator_agent': '依据当前结果起草模拟试验方案，再调用 panacea_extract_components；明确标记假设和未验证内容。',
    'patient_matching_agent': '使用用户明确提供的 XML 文件和已有试验方案调用 match_patient_trial，报告成功处理、失败和匹配数量。',
    'trial_prediction_agent': '调用 predict_trial_success；若模型未配置或结果不可用，明确说明原因，禁止自行编造概率。',
    'planning_final': '汇总已有工具证据，区分工具预测、方案假设与缺失信息。输出指定报告格式，最后单独一行 FINAL。',
}
REQUIRED_TOOLS = {
    'druggen_agent': {'run_druggen'},
    'admet_docking': {'run_docking'},
    'chemical_filter': {'select_leads_from_smiles'},
    'admet_predict': {'chemfm_predict_single', 'chemfm_predict_many'},
    'mol_opt_agent': {'molecule_optimizer'},
    'admet_reeval': {'chemfm_predict_single', 'chemfm_predict_many'},
    'chem_reeval': {'select_leads_from_smiles'},
    'admet_final': {'chemfm_predict_single', 'chemfm_predict_many'},
    'trial_generator_agent': {'panacea_extract_components'},
    'patient_matching_agent': {'match_patient_trial'},
    'trial_prediction_agent': {'predict_trial_success'},
}

def merge_stage_results(left, right):
    return {**(left or {}), **(right or {})}

def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(block if isinstance(block, str) else str(block.get('text', ''))
                         for block in content if isinstance(block, (str, dict)))
    return str(content)

def contains_error(value):
    if isinstance(value, str):
        try:
            return contains_error(json.loads(value))
        except (ValueError, TypeError):
            return False
    if isinstance(value, list):
        return any(contains_error(item) for item in value)
    if isinstance(value, dict):
        if value.get('error') or value.get('errors') or value.get('ok') is False or value.get('isError') is True:
            return True
        return any(contains_error(item) for item in value.values())
    return False

ACTIVE_HANDOFF = ContextVar('drugforge_handoff', default=None)


def guard_tool(tool):
    """Validate structured stage inputs before a domain tool executes."""
    async def guarded(**kwargs):
        context = ACTIVE_HANDOFF.get()
        if context is None:
            raise ValueError('Tool invoked outside a configured stage')
        validate_arguments(tool.name, kwargs, context)
        return await tool.ainvoke(kwargs)
    return StructuredTool.from_function(name=tool.name, description=tool.description,
                                        args_schema=tool.args_schema, coroutine=guarded)


class StageExecutionError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"{record['stage']}: {record.get('error', 'stage failed')}")


def usage_from_messages(messages):
    calls = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata
        metadata = message.response_metadata or {}
        calls.append({'model': metadata.get('model_name') or metadata.get('model'),
                      'input_tokens': usage.get('input_tokens') if usage else None,
                      'output_tokens': usage.get('output_tokens') if usage else None})
    return calls


def make_agent_node(stage, agent, *, timeout=600, max_steps=24, record_sink=None):
    async def node(state):
        started = time.monotonic()
        attempt_id = uuid.uuid4().hex
        context = handoff(stage, state)
        # Agents receive a typed handoff, not the accumulated multi-agent transcript.
        inputs = [HumanMessage(content=json.dumps(context, ensure_ascii=False)),
                  HumanMessage(content=STAGE_TASKS[stage])]
        record = {'id': attempt_id, 'stage': stage, 'round': context['round'],
                  'status': 'failed', 'started_at': time.time(), 'summary': '', 'tool_results': [], 'llm_calls': [],
                  'data': merge_data([]), 'input': context}
        token = ACTIVE_HANDOFF.set(context)
        messages = []
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({'messages': inputs}, config={'recursion_limit': max_steps}), timeout=timeout)
            messages = result['messages'][len(inputs):]
            record['llm_calls'] = usage_from_messages(messages)
            calls = {call['id']: call for message in messages if isinstance(message, AIMessage)
                     for call in message.tool_calls}
            parts = []
            for msg in messages:
                if not isinstance(msg, ToolMessage):
                    continue
                call = calls.get(msg.tool_call_id)
                entry = {'id': 'E-' + attempt_id + '-' + str(len(record['tool_results'])),
                         'tool': msg.name, 'call_id': msg.tool_call_id, 'args': call['args'] if call else None,
                         'ok': False, 'arguments_valid': False, 'content': msg.content, 'payload': None}
                entry['input_sha256'] = hashlib.sha256(json.dumps(entry['args'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                entry['sha256'] = hashlib.sha256(json.dumps(msg.content, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                record['tool_results'].append(entry)
                try:
                    if not call or call['name'] != msg.name:
                        raise ValueError('Tool result lacks a matching call and input arguments')
                    validate_arguments(msg.name, call['args'], context)
                    entry['arguments_valid'] = True
                    if msg.status == 'error' or contains_error(msg.content):
                        raise ValueError(content_text(msg.content))
                    entry['payload'] = decode_payload(msg.content)
                    if msg.name in REQUIRED_TOOLS.get(stage, set()):
                        parts.append(normalize(msg.name, entry['payload'], call['args'], entry['id'], context))
                    entry['ok'] = True
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    entry['error'] = str(exc)
            required = REQUIRED_TOOLS.get(stage, set())
            successful = {e['tool'] for e in record['tool_results'] if e['ok']}
            optional = stage == 'trial_prediction_agent'
            if required and not required.intersection(successful):
                if not optional or not required.intersection(e['tool'] for e in record['tool_results']):
                    errors = [e.get('error', '') for e in record['tool_results']]
                    raise ValueError('no successful required tool result; ' + ('; '.join(errors) or 'no tool evidence'))
                messages = [m for m in messages if not isinstance(m, AIMessage) or m.tool_calls]
                messages.append(AIMessage(content='试验成功概率：不可用。预测工具未产生有效概率，详见证据记录。', name=stage))
            record['summary'] = next((content_text(m.content) for m in reversed(messages)
                                      if isinstance(m, AIMessage) and not m.tool_calls), '')
            if not record['summary'].strip():
                raise ValueError('empty stage summary')
            record['data'] = merge_data(parts, context.get('target_id'))
            record['status'] = 'unavailable' if optional and not required.intersection(successful) else 'completed'
        except asyncio.CancelledError:
            record['error'] = 'Stage interrupted; partial provider usage may be unavailable'
            raise
        except Exception as exc:
            record['error'] = f'exceeded {timeout}s stage timeout' if isinstance(exc, asyncio.TimeoutError) else str(exc)
            raise StageExecutionError(record) from exc
        finally:
            ACTIVE_HANDOFF.reset(token)
            record['duration_seconds'] = round(time.monotonic() - started, 6)
            if record_sink:
                record_sink(record)
        return {'messages': messages, 'stage_results': {stage: record}, 'attempts': {attempt_id: record}}
    node.__name__ = stage
    return node
