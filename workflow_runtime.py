"""Stage boundaries and execution evidence shared by the live workflow."""
import asyncio
import json
import time
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
        if value.get('error') or value.get('ok') is False or value.get('isError') is True:
            return True
        return any(contains_error(item) for item in value.values())
    return False

def make_agent_node(stage, agent, *, timeout=600, max_steps=24):
    async def node(state):
        started = time.monotonic()
        inputs = [*state['messages'], HumanMessage(content=STAGE_TASKS[stage])]
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({'messages': inputs}, config={'recursion_limit': max_steps}),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f'{stage}: exceeded {timeout}s stage timeout') from exc
        messages = result['messages'][len(inputs):]
        evidence = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                evidence.append({'tool': msg.name, 'call_id': msg.tool_call_id,
                                 'ok': msg.status != 'error' and bool(content_text(msg.content).strip())
                                       and not contains_error(msg.content),
                                 'content': msg.content})
        required = REQUIRED_TOOLS.get(stage, set())
        successful = {item['tool'] for item in evidence if item['ok']}
        # Trial prediction is optional: an explicit tool failure must be disclosed, not fabricated.
        optional = stage == 'trial_prediction_agent'
        if required and not required.intersection(successful):
            if not optional or not required.intersection(item['tool'] for item in evidence):
                if evidence:
                    detail = content_text(evidence[-1]['content'])[:500]
                else:
                    detail = 'agent returned no tool evidence'
                raise RuntimeError(f'{stage}: no successful required tool result; {detail}')
            messages = [m for m in messages if not isinstance(m, AIMessage) or m.tool_calls]
            messages.append(AIMessage(content='试验成功概率：不可用。预测工具执行失败，未产生有效概率；详见工具错误记录。', name=stage))
        summary = next((content_text(m.content) for m in reversed(messages)
                        if isinstance(m, AIMessage) and not m.tool_calls), '')
        if not summary.strip():
            raise RuntimeError(f'{stage}: empty stage summary')
        record = {'status': 'unavailable' if optional and not required.intersection(successful) else 'completed',
                  'duration_seconds': round(time.monotonic() - started, 3),
                  'summary': summary, 'tool_results': evidence}
        return {'messages': messages, 'stage_results': {stage: record}}
    node.__name__ = stage
    return node
