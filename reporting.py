"""Deterministic, source-linked reports; agent prose is saved separately as a draft."""
import hashlib
import html
import json
from pathlib import Path
from contracts import pointer_get


def build_evidence(attempts):
    sources, claims = {}, []
    for record in attempts.values():
        for entry in record.get('tool_results', []):
            sources[entry['id']] = {**entry, 'stage': record['stage'], 'round': record['round']}
        if record.get('status') != 'completed':
            continue
        data = record.get('data', {})
        for mol in data.get('molecules', []):
            claims.append({'type': 'molecule', 'value': mol['smiles'], 'source': mol['source'], 'pointer': mol['pointer']})
        for observation in data.get('observations', []):
            claims.append({'type': 'observation', **observation})
        if data.get('prediction'):
            p = data['prediction']
            claims.append({'type': 'probability', 'value': p['probability'], 'source': p['source'], 'pointer': p['pointer']})
    for entry in sources.values():
        digest = hashlib.sha256(json.dumps(entry['content'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        input_digest = hashlib.sha256(json.dumps(entry['args'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if input_digest != entry['input_sha256']:
            raise ValueError('Evidence input hash mismatch')
        if digest != entry['sha256']:
            raise ValueError('Evidence content hash mismatch')
        # Payload must still match the original response, not an edited intermediate value.
        if entry.get('ok'):
            from contracts import decode_payload
            if decode_payload(entry['content']) != entry['payload']:
                raise ValueError('Evidence payload differs from raw response')
    for claim in claims:
        source = sources.get(claim['source'])
        if not source or not source['ok']:
            raise ValueError('Claim references missing or failed evidence')
        actual = pointer_get(source['payload'], claim['pointer'])
        if actual != claim['value']:
            raise ValueError('Claim does not match its source value')
    return {'sources': sources, 'claims': claims}


def cell(value):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return html.escape(text).replace('|', '&#124;').replace('\n', '<br>')


def write_report(run, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts = run.get('committed_attempts', run.get('attempts', {}))
    evidence = build_evidence(attempts)
    prefix = output.with_suffix('')
    json_path = prefix.with_suffix('.evidence.json')
    html_path = prefix.with_suffix('.evidence.html')
    def ref(source):
        if source not in evidence['sources']:
            raise ValueError('Report references unknown evidence')
        return f'[{source}]({html_path.name}#{source})'
    lines = ['# DrugForge · 可追溯运行报告', '',
             f"运行：`{run.get('run_id', 'demo')}` · 状态：**{run.get('status', 'unknown')}**", '',
             '模式：**离线模拟**。所有数值均为固定测试数据。' if run.get('mode', run.get('configuration', {}).get('mode')) == 'offline_demo'
             else '模式：真实工具运行。工具预测与生成方案不等同于经过验证的实验结论。', '',
             '本报告由结构化工具结果直接生成。来源链接包含原始输入、输出与哈希；LLM 文字草稿另存，不用于生成数值。', '',
             '## 分子与性质', '', '| 阶段 / 轮次 | 分子 SMILES | 性质 | 值 | 单位 | 来源 |',
             '|---|---|---|---|---|---|']
    molecules = {m['id']: m['smiles'] for r in attempts.values() for m in r.get('data', {}).get('molecules', [])}
    for record in attempts.values():
        for obs in record.get('data', {}).get('observations', []):
            lines.append('| ' + ' | '.join([cell(f"{record['stage']} / {record['round']}"),
                cell(molecules.get(obs['molecule_id'], '—')), cell(obs['endpoint']), cell(obs['value']),
                cell(obs['unit'] or '未知 / 未提供'), ref(obs['source'])]) + ' |')
    lines += ['', '## 筛选记录', '', '| 分子 | 结果 | 原因 | 来源 |', '|---|---|---|---|']
    for record in attempts.values():
        for selection in record.get('data', {}).get('selections', []):
            # Verify complete selection objects from their source, beyond merely checking the ID.
            raw = pointer_get(evidence['sources'][selection['source']]['payload'], selection['pointer'])
            compound = raw if selection['accepted'] else raw['compound']
            if (compound.get('canonical_smiles') or compound.get('smiles')) != selection['smiles']:
                raise ValueError('Selection molecule differs from source')
            if not selection['accepted'] and raw.get('reason', 'unspecified') != selection['reason']:
                raise ValueError('Selection reason differs from source')
            lines.append(f"| {cell(selection['smiles'])} | {'通过' if selection['accepted'] else '未通过'} | {cell(selection['reason'])} | {ref(selection['source'])} |")
    stages = run.get('stage_results', {})
    trial = stages.get('trial_generator_agent', {}).get('data', {}).get('trial', {})
    lines += ['', '## 模拟临床方案', '']
    if trial:
        source = evidence['sources'][trial['source']]
        if source['args']['trial_text'] != trial['text'] or source['payload']['_raw'] != trial['components']:
            raise ValueError('Trial proposal differs from source')
        lines += ['生成方案，尚未验证。来源：' + ref(trial['source']), '']
        for key, value in trial['components'].items():
            lines += [f'**{cell(key)}**', '', cell(value), '']
    else:
        lines.append('未生成。')
    patients = stages.get('patient_matching_agent', {}).get('data', {}).get('patients', {})
    lines += ['', '## 患者匹配', '']
    if patients:
        raw = evidence['sources'][patients['source']]['payload']
        if [str(m['pid']) for m in raw['matches']] != patients['matched_ids']:
            raise ValueError('Patient IDs differ from source')
        lines.append(f"匹配 ID：{cell(patients['matched_ids'])}。来源：{ref(patients['source'])}")
    else:
        lines.append('未执行或不可用。')
    prediction = stages.get('trial_prediction_agent', {}).get('data', {}).get('prediction', {})
    lines += ['', '## 试验成功概率', '']
    if prediction:
        lines.append(f"模型输出：{prediction['probability']}（未经校准验证）。来源：{ref(prediction['source'])}")
    else:
        lines.append('不可用，未生成替代概率。')
    lines += ['', '## 执行指标', '', '```json', json.dumps(run.get('metrics', {}), ensure_ascii=False, indent=2), '```', '',
              f'[完整证据 JSON]({json_path.name}) · [证据浏览器]({html_path.name})', '']
    json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    output.write_text('\n'.join(lines), encoding='utf-8')
    cards = []
    for source_id, entry in evidence['sources'].items():
        cards.append(f'<details id="{source_id}" open><summary>{html.escape(entry["stage"])} · {html.escape(entry["tool"] or "unknown")} · {source_id}</summary><pre>{html.escape(json.dumps(entry, ensure_ascii=False, indent=2))}</pre></details>')
    html_path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DrugForge evidence</title><style>body{max-width:1000px;margin:40px auto;padding:0 20px;font:15px system-ui;background:#101827;color:#e5edf7}details{padding:20px;margin:20px 0;background:#1b293d;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}summary{color:#79dec8}</style><h1>DrugForge · 证据浏览器</h1><p>原始工具输入与输出；哈希用于完整性检查，不证明科学有效性。</p>' + ''.join(cards) + '</html>', encoding='utf-8')
    return evidence
