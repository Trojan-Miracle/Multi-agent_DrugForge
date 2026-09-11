"""Measured execution metrics and optional user-supplied LLM price estimates."""
import math

def score_run(run, prices=None, reference=None):
    prices = (run.get("price_schedule") if prices is None else prices) or {}
    currency = prices.get('currency')
    rates = prices.get('models', {})
    for rate in rates.values():
        for key in ('input_per_million', 'output_per_million'):
            value = rate[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Token rates must be finite non-negative numbers')
    if rates and not currency:
        raise ValueError('Price configuration must specify currency')
    attempts = list(run.get('attempts', {}).values())
    tool_calls = [t for r in attempts for t in r.get('tool_results', [])]
    llm_calls = [c for r in attempts for c in r.get('llm_calls', [])]
    covered, amount, inputs, outputs = 0, 0.0, 0, 0
    for call in llm_calls:
        if call.get('input_tokens') is None or call.get('output_tokens') is None:
            continue
        inputs += call['input_tokens']; outputs += call['output_tokens']
        rate = rates.get(call.get('model'))
        if rate:
            covered += 1
            amount += (call['input_tokens'] * rate['input_per_million'] + call['output_tokens'] * rate['output_per_million']) / 1_000_000
    incomplete_attempts = sum(r.get('status') == 'failed' for r in attempts)
    full = bool(llm_calls) and covered == len(llm_calls) and not incomplete_attempts and not run.get("interrupted_without_shutdown")
    metrics = {'completed': run.get('status') == 'done',
               'attempt_count': len(attempts), 'failed_attempt_count': incomplete_attempts,
               'stage_seconds_total': sum(r.get('duration_seconds', 0) for r in attempts),
               'active_wall_seconds': run.get('active_seconds'),
               'active_wall_seconds_complete': not run.get('interrupted_without_shutdown', False),
               'tool_calls': len(tool_calls),
               'tool_success_rate': sum(bool(t['ok']) for t in tool_calls) / len(tool_calls) if tool_calls else None,
               'tool_argument_accuracy': sum(bool(t.get('arguments_valid')) for t in tool_calls) / len(tool_calls) if tool_calls else None,
               'llm_calls_observed': len(llm_calls), 'input_tokens_observed': inputs, 'output_tokens_observed': outputs,
               'estimated_llm_cost': round(amount, 8) if full else None,
               'priced_llm_subtotal': round(amount, 8) if covered else None,
               'cost_coverage': covered / len(llm_calls) if llm_calls else None,
               'currency': currency, 'cost_scope': 'LLM only; excludes tool/GPU charges and unreported failed requests',
               'reference_scores': {}}
    if reference:
        stages = run.get('stage_results', {})
        scores = metrics['reference_scores']
        expected = set(reference.get('required_tools', []))
        called = {t['tool'] for t in tool_calls if t['ok']}
        if expected:
            scores['required_tool_recall'] = len(expected & called) / len(expected)
        for key, actual in [
            ('candidate_smiles', {m['smiles'] for m in stages.get('chem_reeval', stages.get('chemical_filter', {})).get('data', {}).get('molecules', [])}),
            ('patient_ids', set(stages.get('patient_matching_agent', {}).get('data', {}).get('patients', {}).get('matched_ids', [])))]:
            if key in reference:
                truth = set(reference[key]); hits = len(actual & truth)
                scores[key + '_precision'] = hits / len(actual) if actual else (1.0 if not truth else 0.0)
                scores[key + '_recall'] = hits / len(truth) if truth else (1.0 if not actual else 0.0)
        if 'trial_outcome' in reference:
            outcome = reference['trial_outcome']
            if outcome not in (0, 1): raise ValueError('trial_outcome must be 0 or 1')
            probability = stages.get('trial_prediction_agent', {}).get('data', {}).get('prediction', {}).get('probability')
            scores['brier_score'] = (probability - outcome) ** 2 if probability is not None else None
    return metrics


def aggregate(runs):
    if not runs:
        raise ValueError('No runs to evaluate')
    metrics = [r['metrics'] for r in runs]
    calls = sum(m['tool_calls'] for m in metrics)
    return {'runs': len(runs), 'completion_rate': sum(m['completed'] for m in metrics) / len(runs),
            'mean_active_wall_seconds': sum(m.get('active_wall_seconds') or 0 for m in metrics) / len(runs),
            'tool_success_rate': sum((m['tool_success_rate'] or 0) * m['tool_calls'] for m in metrics) / calls if calls else None,
            'fully_priced_runs': sum(m['estimated_llm_cost'] is not None for m in metrics)}
