"""Small real DrugGen generation experiment with reproducible inputs and raw output."""
import argparse
import hashlib
import json
from pathlib import Path
import time

MODEL = 'alimotahharynia/DrugGen'
REVISION = '81c03fe27f2a6601e5b09046a549c71701b4fb05'


def run(output, target='P27487', seed=42, count=7):
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
    from rdkit import Chem
    from DrugGen.drugGen_generator import SMILESGenerator
    from science_validation import fetch
    from chemical_properties_mcp_sever import rdkit_physchem_batch, select_leads_from_smiles
    output.mkdir(parents=True, exist_ok=True)
    fasta = output / 'target.fasta'
    source = fetch(f'https://rest.uniprot.org/uniprotkb/{target}.fasta', fasta)
    sequence = ''.join(fasta.read_text().splitlines()[1:])
    if not sequence or not torch.cuda.is_available():
        raise RuntimeError('Requires a valid target sequence and working CUDA')
    set_seed(seed)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = GPT2LMHeadModel.from_pretrained(MODEL, revision=REVISION).to('cuda:0')
    model.eval()
    generator = SMILESGenerator(model, tokenizer, {target: sequence}, str(output / 'generated.tsv'))
    # Stable ordering after the upstream generator's set-based uniqueness filter.
    generated = sorted(generator.generate_smiles(sequence, count))[:count]
    valid = [s for s in generated if Chem.MolFromSmiles(s) is not None]
    report = {'kind': 'real_druggen_generation', 'target': target, 'seed': seed,
              'model': MODEL, 'revision': REVISION, 'target_source': source,
              'generated_smiles': generated, 'requested_count': count,
              'returned_count': len(generated), 'valid_count': len(valid),
              'valid_fraction': len(valid)/len(generated) if generated else None,
              'canonical_unique_count': len({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in valid}),
              'physchem': rdkit_physchem_batch(valid),
              'selection': select_leads_from_smiles(valid),
              'seconds': time.monotonic()-started, 'gpu': torch.cuda.get_device_name(0),
              'scope': 'Generation and descriptor check only; no demonstrated binding or efficacy.'}
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/science/druggen'))
    p.add_argument('--target', default='P27487')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    print(json.dumps(run(args.output, args.target, args.seed), indent=2))
