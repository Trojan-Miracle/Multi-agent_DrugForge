"""Typed handoffs built from tool outputs, never extracted from agent prose.

SMILES IDs identify exact strings; chemical equivalence is not inferred here.
Unknown ADMET units remain unknown. Each observation points into a raw payload.
"""
import hashlib
import json
import math
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Molecule(Contract):
    id: str
    smiles: str = Field(min_length=1)
    source: str
    pointer: str

class Observation(Contract):
    molecule_id: str | None = None
    endpoint: str
    value: str | int | float | bool
    unit: str | None = None
    unit_origin: Literal['adapter', 'tool', 'unknown'] = 'unknown'
    kind: Literal['computed', 'predicted', 'selection']
    source: str
    pointer: str

class Selection(Contract):
    smiles: str
    accepted: bool
    reason: str
    source: str
    pointer: str

class TrialData(Contract):
    text: str = Field(min_length=1)
    components: dict[str, str]
    source: str
    pointer: str
    type: Literal['generated_proposal']

class PatientData(Contract):
    matched_ids: list[str]
    matched_count: int = Field(ge=0)
    processed_count: int | None = Field(default=None, ge=0)
    source: str
    pointer: str

    @model_validator(mode='after')
    def validate_counts(self):
        if len(set(self.matched_ids)) != self.matched_count:
            raise ValueError('Patient matches must have unique IDs and an accurate count')
        if self.processed_count is not None and self.matched_count > self.processed_count:
            raise ValueError('Matches exceed processed patients')
        return self

class PredictionData(Contract):
    probability: float = Field(ge=0, le=1)
    source: str
    pointer: str
    validated: Literal[False] = False

class StageData(Contract):
    schema_version: int = 1
    molecules: list[Molecule] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    selections: list[Selection] = Field(default_factory=list)
    trial: dict = Field(default_factory=dict)
    patients: dict = Field(default_factory=dict)
    prediction: dict = Field(default_factory=dict)
    target_id: str | None = None

    @model_validator(mode='after')
    def validate_nested_data(self):
        for field, schema in [('trial', TrialData), ('patients', PatientData), ('prediction', PredictionData)]:
            value = getattr(self, field)
            if value:
                setattr(self, field, schema.model_validate(value).model_dump())
        return self

# First entry supplies the current molecules. Optional earlier stages provide context.
DEPENDENCIES = {
    'planning_start': [], 'druggen_agent': ['planning_start'],
    'admet_docking': ['druggen_agent'], 'chemical_filter': ['admet_docking'],
    'admet_predict': ['chemical_filter'],
    'mol_opt_agent': ['chemical_filter', 'admet_predict'],
    'admet_reeval': ['mol_opt_agent'], 'chem_reeval': ['admet_reeval'],
    'admet_final': ['chem_reeval', 'admet_reeval'],
    'trial_generator_agent': ['admet_final', 'chem_reeval'],
    'patient_matching_agent': ['trial_generator_agent'],
    'trial_prediction_agent': ['trial_generator_agent'],
    'planning_final': ['admet_final', 'chem_reeval', 'trial_generator_agent',
                       'patient_matching_agent', 'trial_prediction_agent'],
}

def handoff(stage, state):
    stages = state.get('stage_results', {})
    dependencies = DEPENDENCIES[stage]
    if stage == 'mol_opt_agent' and state.get('opt_count', 0):
        dependencies = ['chem_reeval', 'admet_reeval']
    upstream = {key: StageData.model_validate(stages[key]['data']).model_dump()
                for key in dependencies if key in stages and stages[key].get('data')}
    molecules = next((value['molecules'] for value in upstream.values() if value['molecules']), [])
    target = state.get('target_id') or next((v['target_id'] for v in upstream.values() if v['target_id']), None)
    task = state.get('task') or next((str(m.content) for m in state.get('messages', [])
                                    if getattr(m, 'type', '') == 'human'), '')
    return {'task': task, 'target_id': target, 'patients_path': state.get('patients_path'),
            'round': state.get('opt_count', 0) + 1, 'molecules': molecules, 'upstream': upstream}

def decode_payload(content):
    if isinstance(content, list):
        blocks = [x if isinstance(x, str) else x.get('text', '') for x in content]
        if len(blocks) != 1:
            raise ValueError('Expected one JSON tool result')
        content = blocks[0]
    if isinstance(content, str):
        content = json.loads(content)
    if not isinstance(content, dict):
        raise ValueError('Expected a JSON object tool result')
    # Reject NaN/infinity anywhere in the payload, including nested structures.
    json.dumps(content, allow_nan=False)
    return content

def pointer_get(document, pointer):
    value = document
    for part in pointer.split('/')[1:]:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value

def escape_pointer(value):
    return str(value).replace('~', '~0').replace('/', '~1')

def molecule(smiles, source, pointer):
    if not isinstance(smiles, str) or not smiles.strip() or any(c.isspace() for c in smiles):
        raise ValueError('Expected a non-empty SMILES string without whitespace')
    return Molecule(id='mol-' + hashlib.sha256(smiles.encode()).hexdigest()[:16],
                    smiles=smiles, source=source, pointer=pointer)

def validate_arguments(tool, args, context):
    allowed = {m['smiles'] for m in context['molecules']}
    if tool in {'run_docking', 'select_leads_from_smiles', 'chemfm_predict_single',
                'chemfm_predict_many', 'molecule_optimizer'}:
        value = args.get('smiles_list', args.get('smiles'))
        if isinstance(value, str):
            if value.startswith('['):
                value = json.loads(value)
            else:
                value = [value]
        if not allowed or not isinstance(value, list) or not value or not set(value) <= allowed:
            raise ValueError(f'{tool}: input molecules do not match the structured handoff')
    if tool in {'run_druggen', 'run_docking'} and context.get('target_id'):
        if args.get('uniprot_id') != context['target_id']:
            raise ValueError('Target ID differs from the handoff')
    if tool in {'match_patient_trial', 'predict_trial_success'}:
        trial = context['upstream'].get('trial_generator_agent', {}).get('trial', {})
        if not trial or args.get('trial_text') != trial.get('text'):
            raise ValueError('Trial text differs from the structured handoff')
    if tool == 'match_patient_trial' and args.get('xml_path') != context.get('patients_path'):
        raise ValueError('Patient file differs from the configured input')

def normalize(tool, payload, args, source, context):
    """Normalize supported primary tools. Auxiliary descriptions produce no claims."""
    data = StageData(target_id=context.get('target_id'))
    current = {m['smiles']: Molecule.model_validate(m) for m in context['molecules']}
    def observe(smiles, endpoint, value, path, unit=None, kind='predicted'):
        if smiles not in current:
            raise ValueError('Tool returned a molecule absent from its handoff')
        data.observations.append(Observation(molecule_id=current[smiles].id, endpoint=endpoint,
            value=value, unit=unit, unit_origin='adapter' if unit else 'unknown',
            kind=kind, source=source, pointer=path))
    if tool == 'run_druggen':
        smiles = payload.get('smiles')
        if not isinstance(smiles, list) or not smiles:
            raise ValueError('DrugGen returned no candidate molecules')
        data.molecules = [molecule(s, source, f'/smiles/{i}') for i, s in enumerate(smiles)]
        data.target_id = args.get('uniprot_id')
    elif tool == 'run_docking':
        smiles, scores = payload.get('smiles'), payload.get('scores')
        if not smiles or not isinstance(scores, list) or len(smiles) != len(scores):
            raise ValueError('Docking must return aligned smiles and scores')
        for i, (smi, score) in enumerate(zip(smiles, scores)):
            path = f'/scores/{i}'
            # Legacy adapter returns [score, pocket_center, box_size].
            if isinstance(score, list):
                score, path = score[0], path + '/0'
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError('Docking score must be finite')
            observe(smi, 'vina_score', score, path, 'kcal/mol', 'computed')
        data.molecules = [current[s] for s in smiles]
    elif tool == 'select_leads_from_smiles':
        leads = payload.get('leads')
        if not isinstance(leads, list) or not leads:
            raise ValueError('No lead molecules passed the filter')
        units = {'MW': 'g/mol', 'ExactMW': 'g/mol', 'TPSA': 'Å²'}
        for i, compound in enumerate(leads):
            smi = compound.get('canonical_smiles') or compound.get('smiles')
            key = 'canonical_smiles' if 'canonical_smiles' in compound else 'smiles'
            mol = molecule(smi, source, f'/leads/{i}/{key}')
            data.molecules.append(mol)
            data.selections.append(Selection(smiles=smi, accepted=True, reason='selected_by_tool',
                                            source=source, pointer=f'/leads/{i}'))
            for endpoint, value in compound.items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    data.observations.append(Observation(molecule_id=mol.id, endpoint=endpoint,
                        value=value, unit=units.get(endpoint), unit_origin='adapter' if endpoint in units else 'unknown',
                        kind='predicted' if endpoint in {'pKa', 'logD_acid', 'logD_base'} else 'computed',
                        source=source, pointer=f'/leads/{i}/{escape_pointer(endpoint)}'))
        for i, rejected in enumerate(payload.get('rejected', [])):
            compound = rejected['compound']
            data.selections.append(Selection(smiles=compound.get('canonical_smiles') or compound['smiles'],
                accepted=False, reason=rejected.get('reason', 'unspecified'), source=source, pointer=f'/rejected/{i}'))
    elif tool in {'chemfm_predict_single', 'chemfm_predict_many'}:
        def add(smi, endpoint, result, path):
            if result.get('prediction') is None:
                raise ValueError('ADMET result has no prediction')
            observe(smi, endpoint, result['prediction'], path + '/prediction')
        if tool == 'chemfm_predict_single':
            endpoint = payload['property']
            if 'smiles' in payload:
                add(payload['smiles'], endpoint, payload, '')
            else:
                for i, result in enumerate(payload['results']):
                    add(result['smiles'], endpoint, result, f'/results/{i}')
        elif 'smiles' in payload:
            for endpoint, result in payload['results'].items():
                add(payload['smiles'], endpoint, result, f'/results/{escape_pointer(endpoint)}')
        else:
            for smi, results in payload['results'].items():
                for endpoint, result in results.items():
                    add(smi, endpoint, result, f'/results/{escape_pointer(smi)}/{escape_pointer(endpoint)}')
        if not data.observations:
            raise ValueError('ADMET returned no observations')
        observed_ids = {o.molecule_id for o in data.observations}
        data.molecules = [m for m in current.values() if m.id in observed_ids]
    elif tool == 'molecule_optimizer':
        # Free-form model prose cannot silently become the next molecule.
        data.molecules = [molecule(payload['optimized_smiles'], source, '/optimized_smiles')]
    elif tool == 'panacea_extract_components':
        raw = payload['_raw']
        for key in ('inclusion_criteria', 'exclusion_criteria', 'outcomes', 'arms'):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f'Missing trial component: {key}')
        if not args.get('trial_text'):
            raise ValueError('Trial proposal input is missing')
        data.trial = {'text': args['trial_text'], 'components': raw, 'source': source,
                      'pointer': '/_raw', 'type': 'generated_proposal'}
        data.molecules = list(current.values())
    elif tool == 'match_patient_trial':
        matches = payload['matches']
        if payload['matched_patients_count'] != len(matches):
            raise ValueError('Patient match count does not agree with match list')
        data.patients = {'matched_ids': [str(m['pid']) for m in matches],
                         'matched_count': len(matches), 'source': source, 'pointer': '/matches',
                         'processed_count': payload.get('processed_patients_count', payload.get('total_patients_parsed'))}
    elif tool == 'predict_trial_success':
        probability = payload['success_probability']
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError('Probability must be between zero and one')
        data.prediction = {'probability': probability, 'source': source,
                           'pointer': '/success_probability', 'validated': False}
    if tool in {'run_docking', 'chemfm_predict_single', 'chemfm_predict_many'}:
        requested = args.get('smiles_list', args.get('smiles'))
        if requested is not None:
            if isinstance(requested, str):
                requested = json.loads(requested) if requested.startswith('[') else [requested]
            if {m.smiles for m in data.molecules} != set(requested):
                raise ValueError('Tool output does not cover exactly the requested molecules')
        endpoints = args.get('properties') or ([args['property_name']] if 'property_name' in args else None)
        if endpoints and any(o.endpoint not in endpoints for o in data.observations):
            raise ValueError('Tool returned an unrequested endpoint')
    return StageData.model_validate(data.model_dump()).model_dump()

def merge_data(parts, target_id=None):
    result = StageData(target_id=target_id).model_dump()
    molecules = {}
    for part in parts:
        for mol in part['molecules']:
            molecules[mol['id']] = mol
        for key in ('observations', 'selections'):
            result[key].extend(part[key])
        for key in ('trial', 'patients', 'prediction'):
            if part[key]: result[key] = part[key]
        result['target_id'] = part.get('target_id') or result['target_id']
    result['molecules'] = list(molecules.values())
    return StageData.model_validate(result).model_dump()
