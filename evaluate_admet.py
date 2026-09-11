"""Evaluate two classification endpoints on the unmodified official TDC test sets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

DATA_URL = 'https://dataverse.harvard.edu/api/access/datafile/4426004'
ENDPOINTS = {'herg': 'hERG Channel Blockage', 'ames': 'Drug Mutagenicity'}


def metrics(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected one finite probability in [0,1] per label')
    if set(y) != {0, 1}:
        raise ValueError('Evaluation requires both binary classes')
    return {'n': len(y), 'auroc': roc_auc_score(y, p),
            'auprc': average_precision_score(y, p), 'brier': brier_score_loss(y, p)}


def evaluate(output, with_chemfm=False, data_archive=None):
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.ensemble import RandomForestClassifier
    from science_validation import fetch
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'tdc-admet.zip'
    if data_archive:
        archive.write_bytes(Path(data_archive).read_bytes())
        source = {'url': DATA_URL, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    else:
        source = fetch(DATA_URL, archive)
    report = {'dataset': 'TDC ADMET group', 'source': source, 'split': 'official fixed test',
              'model_training_overlap': 'ChemFM adapters were trained on TDC; checkpoint training membership has not been independently verified.',
              'scope': 'Retrospective benchmark reproduction, not external clinical validation.', 'endpoints': {}}
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    with zipfile.ZipFile(archive) as z:
        for endpoint, prop in ENDPOINTS.items():
            frames = {}
            for split in ('train_val', 'test'):
                raw = z.read(f'admet_group/{endpoint}/{split}.csv')
                path = output / f'{endpoint}-{split}.csv'
                path.write_bytes(raw)
                df = pd.read_csv(path)
                mols = [Chem.MolFromSmiles(s) for s in df.Drug]
                if any(m is None for m in mols):
                    raise ValueError('Invalid molecule in official dataset; no silent exclusions')
                frames[split] = (df, mols, [Chem.MolToSmiles(m) for m in mols])
            train, train_mols, train_ids = frames['train_val']
            test, test_mols, test_ids = frames['test']
            # Remove exact test identities from baseline training, if present.
            keep = [s not in set(test_ids) for s in train_ids]
            xtrain = np.asarray([fingerprint.GetFingerprintAsNumPy(m) for m,k in zip(train_mols,keep) if k])
            xtest = np.asarray([fingerprint.GetFingerprintAsNumPy(m) for m in test_mols])
            ytrain, ytest = train.Y.to_numpy()[keep], test.Y.to_numpy()
            entry = {'test_size': len(test), 'test_sha256': hashlib.sha256((output / f'{endpoint}-test.csv').read_bytes()).hexdigest(),
                     'train_test_exact_overlap_removed': keep.count(False), 'baseline': [], 'chemfm': {'status': 'not_run'}}
            prediction_table = test.copy()
            for seed in (0, 1, 2):
                model = RandomForestClassifier(n_estimators=300, random_state=seed,
                                               n_jobs=4, class_weight='balanced')
                model.fit(xtrain, ytrain)
                p = model.predict_proba(xtest)[:, 1]
                entry['baseline'].append({'seed': seed, **metrics(ytest, p)})
                prediction_table[f'morgan_rf_seed_{seed}'] = p
            if with_chemfm:
                from chemfm_local import predict_local
                started = time.monotonic()
                try:
                    actual = predict_local(test.Drug.tolist(), prop)
                    p = [r['prediction'] for r in actual['results']]
                    entry['chemfm'] = {'status': 'completed', **metrics(ytest, p),
                        'source': actual['source'], 'seconds': time.monotonic()-started}
                    prediction_table['chemfm_probability'] = p
                except Exception as exc:
                    entry['chemfm'] = {'status': 'failed', 'error': str(exc)}
            prediction_table.to_csv(output / f'{endpoint}-predictions.csv', index=False)
            report['endpoints'][endpoint] = entry
            (output / 'result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('runs/science/admet'))
    p.add_argument('--chemfm', action='store_true', help='Load official ChemFM weights on CUDA')
    p.add_argument('--data-archive', type=Path, help='Previously downloaded official ADMET ZIP')
    a = p.parse_args()
    print(json.dumps(evaluate(a.output, a.chemfm, a.data_archive), indent=2))
