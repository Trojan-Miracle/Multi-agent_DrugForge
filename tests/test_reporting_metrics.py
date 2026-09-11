import asyncio
import copy
import json
import pytest
from demo import run_demo
from reporting import build_evidence, write_report
from metrics import score_run
from evaluate import run_suite
from pathlib import Path

def test_report_links_exact_raw_values_and_rejects_tampering(tmp_path):
    run = asyncio.run(run_demo(True))
    assert run['status'] == 'done', run.get('error')
    evidence = write_report(run, tmp_path / 'report.md')
    assert evidence['claims']
    assert all(c['source'] in evidence['sources'] for c in evidence['claims'])
    assert (tmp_path / 'report.evidence.html').exists()
    damaged = copy.deepcopy(run['committed_attempts'])
    record = next(r for r in damaged.values() if r.get('data', {}).get('observations'))
    record['data']['observations'][0]['value'] = 99999
    with pytest.raises(ValueError, match='source value'):
        build_evidence(damaged)
    damaged = copy.deepcopy(run['committed_attempts'])
    record = next(r for r in damaged.values() if r.get('tool_results'))
    record['tool_results'][0]['content'] = '{}'
    with pytest.raises(ValueError, match='hash mismatch'):
        build_evidence(damaged)

def test_cost_unknown_without_rates_and_includes_all_attempts():
    run = {'status': 'done', 'attempts': {'one': {'status': 'completed', 'llm_calls': [
        {'model': 'example', 'input_tokens': 1000, 'output_tokens': 500}], 'tool_results': []}}}
    assert score_run(run)['estimated_llm_cost'] is None
    prices = {'currency': 'test-unit', 'models': {'example': {'input_per_million': 2, 'output_per_million': 4}}}
    assert score_run(run, prices)['estimated_llm_cost'] == 0.004
    run['attempts']['failure'] = {'status': 'failed', 'llm_calls': [], 'tool_results': []}
    result = score_run(run, prices)
    assert result['estimated_llm_cost'] is None
    assert result['priced_llm_subtotal'] == 0.004
    assert result['failed_attempt_count'] == 1

def test_reference_scores_use_actual_results():
    run = {'status': 'done', 'stage_results': {
        'patient_matching_agent': {'data': {'patients': {'matched_ids': ['a', 'wrong']}}},
        'trial_prediction_agent': {'data': {'prediction': {'probability': 0.6}}}}}
    scores = score_run(run, reference={'patient_ids': ['a', 'b'], 'trial_outcome': 1})['reference_scores']
    assert scores['patient_ids_precision'] == scores['patient_ids_recall'] == 0.5
    assert scores['brier_score'] == pytest.approx(0.16)

def test_offline_suite_distinguishes_rejections_from_task_completion():
    result, _ = asyncio.run(run_suite(Path(__file__).resolve().parents[1] / 'evaluations/offline_cases.json'))
    assert result['summary']['case_pass_rate'] == 1.0, result['cases']
    assert result['summary']['completion_rate'] < 1.0
    assert result['scientific_validation'] is False

def test_ungraceful_restart_marks_duration_and_cost_incomplete(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(state, 'RUNS_DIR', tmp_path)
    run = state.RunState('test', 'crash-test')
    run.data['price_schedule'] = {'currency': 'test-unit', 'models': {'test': {'input_per_million': 1, 'output_per_million': 1}}}
    run.record_attempt({'id': 'one', 'stage': 'planning_start', 'status': 'completed', 'llm_calls': [
        {'model': 'test', 'input_tokens': 100, 'output_tokens': 100}]})
    resumed = state.RunState.reopen('crash-test')
    resumed.resume()
    metrics = score_run(resumed.data)
    assert metrics['estimated_llm_cost'] is None
    assert metrics['priced_llm_subtotal'] == 0.0002
    assert metrics['active_wall_seconds_complete'] is False
