"""Deterministic offline agents. All structures and numeric values are test fixtures."""
import json
import uuid
from langchain_core.messages import AIMessage, ToolMessage

NODES = ('planning_start', 'druggen_agent', 'admet_docking', 'chemical_filter', 'admet_predict',
         'admet_final', 'trial_generator_agent', 'patient_matching_agent', 'trial_prediction_agent', 'planning_final')

class FixtureAgent:
    def __init__(self, stage, scenario='success'):
        self.stage, self.scenario = stage, scenario

    async def ainvoke(self, inputs, config):
        context = json.loads(inputs['messages'][0].content)
        molecules = [m['smiles'] for m in context['molecules']]
        stage = self.stage
        args, tool, payload = {}, None, None
        if stage == 'druggen_agent':
            tool, args, payload = 'run_druggen', {'uniprot_id': 'P27487', 'num_generated': 7}, {'smiles': ['CCO', 'CCN']}
        elif stage == 'admet_docking':
            tool, args = 'run_docking', {'uniprot_id': context['target_id'], 'smiles_list': molecules}
            payload = {'smiles': molecules, 'scores': [[-6.0 - i, [0, 0, 0], 20] for i in range(len(molecules))]}
            if self.scenario == 'wrong_molecule': args['smiles_list'] = ['INVALID']
            if self.scenario == 'tool_error': payload = {'error': 'Fixture docking outage'}
        elif stage in {'chemical_filter', 'chem_reeval'}:
            tool, args = 'select_leads_from_smiles', {'smiles_list': molecules}
            payload = {'leads': [{'canonical_smiles': molecules[0], 'MW': 46.0, 'logP': 1.0,
                                  'TPSA': 20.0, 'HBD': 1, 'HBA': 1, 'RotB': 1}],
                       'rejected': [{'compound': {'canonical_smiles': s}, 'reason': 'fixture rejection'} for s in molecules[1:]]}
        elif stage in {'admet_predict', 'admet_reeval', 'admet_final'}:
            tool, args = 'chemfm_predict_many', {'smiles': molecules, 'properties': ['Drug Oral Bioavailability']}
            payload = {'ok': True, 'results': {s: {'Drug Oral Bioavailability': {'prediction': 0.7}} for s in molecules}}
        elif stage == 'mol_opt_agent':
            tool, args = 'molecule_optimizer', {'smiles': molecules[0], 'properties': 'solubility', 'action': 'increase'}
            payload = {'optimized_smiles': 'C' + molecules[0], 'original_smiles': molecules[0]}
            if self.scenario == 'malformed_optimization': payload = {'message': 'Maybe improve this molecule'}
        elif stage == 'trial_generator_agent':
            tool, args = 'panacea_extract_components', {'trial_text': 'DEMO proposal: ' + ','.join(molecules)}
            payload = {'_raw': {key: 'DEMO component: ' + key for key in ('inclusion_criteria', 'exclusion_criteria', 'outcomes', 'arms')}}
        elif stage in {'patient_matching_agent', 'trial_prediction_agent'}:
            args = {'trial_text': context['upstream']['trial_generator_agent']['trial']['text']}
            if stage == 'patient_matching_agent':
                tool = 'match_patient_trial'
                args['xml_path'] = context['patients_path']
                payload = {'matched_patients_count': 1, 'processed_patients_count': 2,
                           'matches': [{'pid': 'demo-patient-1'}]}
            else:
                tool = 'predict_trial_success'
                payload = {'error': 'DEMO: checkpoint unavailable'} if self.scenario == 'prediction_unavailable' else {'success_probability': 0.6, 'validated': False}
        summary = f'[DEMO / 模拟输出] {stage}'
        if stage == 'planning_final':
            summary += '\n药物发现性质\n临床试验报告\n患者匹配\n试验成功概率\n总结\nFINAL'
        messages = []
        if tool and not (self.scenario == 'missing_tool' and stage == 'admet_docking'):
            call_id = uuid.uuid4().hex
            messages += [AIMessage(content='', tool_calls=[{'name': tool, 'args': args, 'id': call_id}]),
                         ToolMessage(content=json.dumps(payload), name=tool, tool_call_id=call_id)]
        messages.append(AIMessage(content=summary))
        return {'messages': [*inputs['messages'], *messages]}
