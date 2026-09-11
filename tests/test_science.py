import pytest

Chem = pytest.importorskip('rdkit.Chem')
from rdkit.Geometry import Point3D
from science_validation import pose_rmsd


def test_pose_metric_does_not_align_away_translation():
    ref = Chem.MolFromSmiles('CCO')
    conf = Chem.Conformer(3)
    for i in range(3):
        conf.SetAtomPosition(i, Point3D(i, i % 2, 0))
    ref.AddConformer(conf)
    pose = Chem.Mol(ref)
    for i in range(3):
        p = pose.GetConformer().GetAtomPosition(i)
        pose.GetConformer().SetAtomPosition(i, Point3D(p.x, p.y, p.z + 5))
    assert pose_rmsd(ref, pose) == pytest.approx(5)
    assert pose.GetConformer().GetAtomPosition(0).z == 5


def test_physchem_does_not_require_optional_pka_model(monkeypatch):
    import chemical_properties_mcp_sever as chemical
    def forbidden():
        raise AssertionError('RDKit must not load the optional pKa model')
    monkeypatch.setattr(chemical, '_cached_model', forbidden)
    result = chemical.rdkit_physchem_batch(['CCO'])
    assert result['results'][0]['MW'] == pytest.approx(46.069, abs=0.001)
    assert not result['errors']
    assert 'pKa' not in result['results'][0]
    assert chemical.select_leads_from_smiles(['invalid'])['error']


def test_local_chemfm_output_matches_workflow_contract(monkeypatch):
    pytest.importorskip('gradio_client')
    import chemfm_local
    import admet_prediction_mcp_server as adapter
    from contracts import molecule, normalize
    monkeypatch.setenv('CHEMFM_BACKEND', 'local')
    def predict(smiles, prop):
        return {'ok': True, 'results': [{'smiles': s, 'prediction': 0.2, 'unit': 'probability'} for s in smiles], 'source': {}}
    monkeypatch.setattr(chemfm_local, 'predict_local', predict)
    args = {'smiles': ['CCO'], 'properties': ['hERG Channel Blockage']}
    actual = adapter.chemfm_predict_many(**args)
    context = {'molecules': [molecule('CCO', 'input', '/smiles').model_dump()]}
    data = normalize('chemfm_predict_many', actual, args, 'source', context)
    assert data['observations'][0]['unit'] == 'probability'
    assert data['observations'][0]['value'] == 0.2


def test_probability_metrics_reject_missing_and_nonfinite_predictions():
    pytest.importorskip('sklearn')
    from evaluate_admet import metrics
    for p in ([0.2], [0.2, float('nan')], [0.2, 1.1]):
        with pytest.raises(ValueError):
            metrics([0, 1], p)


def test_optimizer_rejects_invalid_smiles_before_handoff(monkeypatch):
    import asyncio
    import json
    import mol_opt_mcp_server as optimizer
    monkeypatch.setattr(optimizer, '_optimize', lambda prompt: {
        'content': json.dumps({'optimized_smiles': 'not-a-molecule'})})
    result = json.loads(asyncio.run(optimizer.molecule_optimizer('CCO', 'hERG risk', 'decrease')))
    assert 'invalid SMILES' in result['error']


def test_docking_never_silently_changes_boron_to_carbon():
    import docking_module
    with pytest.raises(ValueError, match='refusing to replace'):
        docking_module.smiles_to_pdbqt('B(O)O')


def test_gradio_constructor_matches_installed_sdk(monkeypatch):
    pytest.importorskip('gradio_client')
    import inspect
    import admet_prediction_mcp_server as adapter
    signature = inspect.signature(adapter.Client)
    def constructor(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        return bound.arguments
    monkeypatch.setattr(adapter, 'Client', constructor)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    assert adapter._client()['token'] is None
