"""Evaluate fixed offline tasks or saved live run logs against optional references."""
import argparse
import asyncio
import json
from pathlib import Path
from demo import run_demo
from metrics import score_run, aggregate

async def run_suite(path, prices=None):
    suite = json.loads(Path(path).read_text(encoding='utf-8'))
    runs, cases = [], []
    for case in suite['cases']:
        run = await run_demo(case.get('patients', False), case['scenario'], prices, case.get('reference'))
        run['run_id'] = case['id']
        runs.append(run)
        checks = {'expected_status': run['status'] == case['expected_status']}
        if 'error_contains' in case:
            checks['expected_error'] = case['error_contains'] in run.get('error', '')
        if case.get('prediction_unavailable'):
            prediction = run.get('stage_results', {}).get('trial_prediction_agent', {})
            checks['unavailable_is_explicit'] = prediction.get('status') == 'unavailable' and not prediction.get('data', {}).get('prediction')
        for key, value in run['metrics']['reference_scores'].items():
            if key != 'brier_score': checks[key] = value == 1.0
        cases.append({'id': case['id'], 'passed': all(checks.values()), 'checks': checks,
                      'status': run['status'], 'metrics': run['metrics']})
    return {'mode': 'offline_fixture_benchmark', 'scientific_validation': False,
            'summary': {**aggregate(runs), 'case_pass_rate': sum(c['passed'] for c in cases) / len(cases)},
            'cases': cases}, runs


def markdown(result):
    lines = ['# DrugForge · 任务评测', '', f"模式：`{result['mode']}`", '',
             '离线用例只验证固定任务及错误处理；真实日志指标也不能单独证明科学有效性。', '',
             '```json', json.dumps(result['summary'], ensure_ascii=False, indent=2), '```', '',
             '| 用例 / 运行 | 状态 | 预期检查通过 | 工具成功率 | 耗时（秒） | LLM 估算成本 |', '|---|---|---|---|---|---|']
    for case in result['cases']:
        m = case['metrics']
        cost = m['estimated_llm_cost']
        rate = m['tool_success_rate']
        lines.append(f"| {case['id']} | {case['status']} | {case.get('passed', '未提供预期')} | {rate if rate is not None else '未知'} | {(m['active_wall_seconds'] or 0):.3f} | {cost if cost is not None else '未知 / 不完整'} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--suite', type=Path, default=None)
    group.add_argument('--runs', type=Path, nargs='+', help='Existing run JSON files')
    parser.add_argument('--reference', type=Path, help='JSON mapping run IDs to expected tools/molecules/patients/outcomes')
    parser.add_argument('--prices', type=Path, help='User-supplied per-million token prices')
    parser.add_argument('--output', type=Path, default=Path('runs/evaluation.json'))
    args = parser.parse_args()
    prices = json.loads(args.prices.read_text()) if args.prices else None
    if args.runs:
        references = json.loads(args.reference.read_text()) if args.reference else {}
        runs, cases = [], []
        for path in args.runs:
            run = json.loads(path.read_text(encoding='utf-8'))
            if 'run_id' not in run or 'attempts' not in run or 'status' not in run:
                parser.error(f'{path} is not a structured run log')
            if prices is not None: run['price_schedule'] = prices
            run['metrics'] = score_run(run, prices, references.get(run['run_id']))
            runs.append(run)
            cases.append({'id': run['run_id'], 'status': run['status'], 'metrics': run['metrics']})
        modes = {r.get('mode', r.get('configuration', {}).get('mode', 'unknown')) for r in runs}
        if len(modes) != 1:
            parser.error('Evaluate live and offline runs separately')
        result = {'mode': next(iter(modes)) + '_log_evaluation', 'scientific_validation': False,
                  'summary': aggregate(runs), 'cases': cases}
    else:
        result, runs = asyncio.run(run_suite(args.suite or Path(__file__).parent / 'evaluations/offline_cases.json', prices))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traces = args.output.parent / (args.output.stem + '.traces')
    traces.mkdir(parents=True, exist_ok=True)
    for index, (case, run) in enumerate(zip(result['cases'], runs), 1):
        path = traces / f'case-{index:03d}.json'
        path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding='utf-8')
        case['trace'] = str(path.relative_to(args.output.parent))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    if any(c.get('passed') is False for c in result['cases']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
