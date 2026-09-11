"""Paired pilot: identical DrugGen leads with zero vs up to three real optimization rounds."""
import argparse
import asyncio
import json
from pathlib import Path
import time


def assess(smiles):
    from chemfm_local import predict_local
    from chemical_properties_mcp_sever import select_leads_from_smiles
    prediction = predict_local([smiles], 'hERG Channel Blockage')
    selection = select_leads_from_smiles([smiles])
    return {'smiles': smiles, 'herg_probability': prediction['results'][0]['prediction'],
            'passes_rules': bool(selection.get('leads')), 'selection': selection,
            'prediction_source': prediction['source']}


async def run(source, output, count=3):
    from mol_opt_mcp_server import molecule_optimizer
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generated = json.loads(source.read_text())
    leads = [x['canonical_smiles'] for x in generated['selection']['leads']][:count]
    if not leads:
        raise ValueError('No actual DrugGen leads passed the initial rules')
    output.mkdir(parents=True, exist_ok=True)
    report = {'kind': 'paired_real_optimization_pilot', 'target': generated['target'],
              'seed': 42, 'requested_candidates': count, 'n': len(leads), 'max_rounds': 3,
              'objective': 'decrease predicted hERG blockage risk',
              'optimizer': {'repo': 'blazerye/DrugAssist-7B', 'revision': '83337f83d30caca6c1dae77dccf8f3f13e119cb7'},
              'scope': 'Same-model proxy score, three-candidate pilot; no independent efficacy claim. No docking rerun.',
              'cases': []}
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for smi in leads:
        started = time.monotonic()
        case = {'baseline': assess(smi), 'rounds': []}
        current = smi
        for iteration in range(1, 4):
            raw = json.loads(await molecule_optimizer(current, 'hERG channel blockage risk', 'decrease'))
            step = {'round': iteration, 'input_smiles': current, 'tool_output': raw}
            if raw.get('error'):
                step['status'] = 'failed'
                case['rounds'].append(step)
                break
            current = raw['optimized_smiles']
            step.update(status='completed', assessment=assess(current))
            step['unchanged'] = Chem.MolToSmiles(Chem.MolFromSmiles(current)) == Chem.MolToSmiles(Chem.MolFromSmiles(step['input_smiles']))
            case['rounds'].append(step)
        completed = [s for s in case['rounds'] if s['status'] == 'completed']
        final = completed[-1]['assessment'] if completed else case['baseline']
        case['final'] = final
        case['completed_rounds'] = len(completed)
        case['delta_predicted_risk'] = final['herg_probability'] - case['baseline']['herg_probability']
        case['improved_and_passes_rules'] = bool(completed) and case['delta_predicted_risk'] < 0 and final['passes_rules']
        case['morgan_tanimoto_to_start'] = DataStructs.TanimotoSimilarity(
            fp.GetFingerprint(Chem.MolFromSmiles(smi)), fp.GetFingerprint(Chem.MolFromSmiles(final['smiles'])))
        case['seconds'] = time.monotonic()-started
        report['cases'].append(case)
        report['summary'] = {'evaluated': len(report['cases']),
            'improved_and_passes_rules': sum(c['improved_and_passes_rules'] for c in report['cases']),
            'failed_tool_rounds': sum(s['status'] == 'failed' for c in report['cases'] for s in c['rounds'])}
        (output / 'result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('runs/science/druggen/result.json'))
    p.add_argument('--output', type=Path, default=Path('runs/science/optimization'))
    p.add_argument('--count', type=int, default=3)
    args = p.parse_args()
    print(json.dumps(asyncio.run(run(args.source, args.output, args.count)), indent=2))
