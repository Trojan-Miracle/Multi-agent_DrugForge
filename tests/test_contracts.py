import asyncio
import json
import pytest
from langchain_core.tools import StructuredTool
from contracts import molecule, normalize, handoff, validate_arguments
from workflow_runtime import ACTIVE_HANDOFF, guard_tool

def context():
    return {'molecules': [molecule('CCO', 'E-input', '/smiles/0').model_dump()],
            'target_id': 'P27487', 'upstream': {}, 'patients_path': None}

def test_guard_prevents_wrong_molecule_before_execution():
    called = []
    async def docking(smiles_list: list[str], uniprot_id: str):
        called.append(smiles_list)
        return '{}'
    original = StructuredTool.from_function(coroutine=docking, name='run_docking', description='test')
    async def run():
        token = ACTIVE_HANDOFF.set(context())
        try:
            with pytest.raises(ValueError, match='handoff'):
                await guard_tool(original).ainvoke({'smiles_list': ['CCN'], 'uniprot_id': 'P27487'})
            assert not called
            await guard_tool(original).ainvoke({'smiles_list': ['CCO'], 'uniprot_id': 'P27487'})
            assert called == [['CCO']]
        finally:
            ACTIVE_HANDOFF.reset(token)
    asyncio.run(run())

def test_legacy_docking_tuple_and_unknown_admet_units():
    data = normalize('run_docking', {'smiles': ['CCO'], 'scores': [[-7.2, [0, 0, 0], 20]]}, {}, 'E-test', context())
    assert data['observations'][0]['pointer'] == '/scores/0/0'
    assert data['observations'][0]['unit'] == 'kcal/mol'
    data = normalize('chemfm_predict_single', {'smiles': 'CCO', 'property': 'endpoint', 'prediction': 'high'}, {}, 'E-test', context())
    assert data['observations'][0]['unit'] is None
    assert data['observations'][0]['value'] == 'high'

def test_rejects_empty_leads_and_unstructured_optimization():
    with pytest.raises(ValueError, match='No lead'):
        normalize('select_leads_from_smiles', {'leads': []}, {}, 'E', context())
    with pytest.raises(KeyError):
        normalize('molecule_optimizer', {'message': 'some SMILES maybe'}, {}, 'E', context())

def test_trial_text_cannot_be_rewritten_between_agents():
    ctx = context()
    ctx['upstream'] = {'trial_generator_agent': {'trial': {'text': 'exact original trial'}}}
    with pytest.raises(ValueError, match='Trial text'):
        validate_arguments('predict_trial_success', {'trial_text': 'rewritten'}, ctx)

def test_handoff_uses_current_round_instead_of_old_conversation():
    initial = {'data': {'molecules': [molecule('CCO', 'E1', '/smiles/0').model_dump()]}}
    updated = {'data': {'molecules': [molecule('CCCO', 'E2', '/optimized_smiles').model_dump()]}}
    data = handoff('mol_opt_agent', {'opt_count': 1, 'messages': [],
                   'stage_results': {'chemical_filter': initial, 'chem_reeval': updated}})
    assert data['molecules'][0]['smiles'] == 'CCCO'
    assert 'messages' not in data
