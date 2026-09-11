"""Local ChemFM classification inference, following the author's Space preprocessing.

Regression adapters require endpoint-specific inverse scaling and are deliberately
not exposed here. Official model and adapter revisions are recorded in results.
"""
from functools import lru_cache
from threading import Lock

BASE = 'ChemFM/ChemFM-3B'
BASE_REVISION = 'c1464dd3c51643d2d6926a71db98558be97e1ecf'
ADAPTERS = {
    'hERG Channel Blockage': ('admet_herg', '4d41c774f6947d67c725268d82f8b3f99dc41e2d'),
    'Drug Mutagenicity': ('admet_ames', 'a1f703a6ab8965ee5bda53c9d189ec50bac7cfaf'),
}
_lock = Lock()


class LocalChemFM:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
        from huggingface_hub import snapshot_download
        if not torch.cuda.is_available():
            raise RuntimeError('Local ChemFM requires a working CUDA device')
        base_path = snapshot_download(BASE, revision=BASE_REVISION,
                                      allow_patterns=['*.json', '*.safetensors'])
        tokenizer_path = snapshot_download('ChemFM/admet_herg',
            revision=ADAPTERS['hERG Channel Blockage'][1], allow_patterns=['*.json', '*.safetensors'])
        # ADMET fine-tuning extended the base vocabulary from 320 to 392 tokens.
        # Explicit tokenizer class avoids auto-loading adapter config as a base model.
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(base_path,
            num_labels=1, dtype=torch.float32, device_map='cuda:0')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.loaded = set()
        self.adapter_embeddings = {}
        self.active_adapter = None

    def predict(self, smiles, property_name, batch_size=8):
        import torch
        from rdkit import Chem
        if property_name not in ADAPTERS:
            raise ValueError('Local backend supports only: ' + ', '.join(ADAPTERS))
        name, revision = ADAPTERS[property_name]
        sources = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError('Invalid SMILES')
            sources.append('<molstart>' + Chem.MolToSmiles(mol) + '<eos>')
        if name not in self.loaded:
            from huggingface_hub import snapshot_download
            from safetensors import safe_open
            from pathlib import Path
            adapter_path = snapshot_download('ChemFM/' + name, revision=revision,
                                             allow_patterns=['*.json', '*.safetensors'])
            with safe_open(str(Path(adapter_path) / 'adapter_model.safetensors'), framework='numpy') as weights:
                if 'base_model.model.score.weight' not in weights.keys():
                    raise ValueError('Adapter lacks the trained classification head')
            from peft import PeftModel
            if not self.loaded:
                self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=name)
            else:
                self.model.load_adapter(adapter_path, adapter_name=name)
            # Fail closed if newer loader APIs silently leave a random task head.
            with safe_open(str(Path(adapter_path) / 'adapter_model.safetensors'), framework='pt') as weights:
                expected = weights.get_tensor('base_model.model.score.weight')
                actual = self.model.base_model.model.score.modules_to_save[name].weight.detach().cpu()
                if not torch.equal(actual, expected):
                    raise ValueError('Loaded classifier differs from the trained adapter head')
                expected_embedding = weights.get_tensor('base_model.model.model.embed_tokens.weight')
                if not torch.equal(self.model.get_input_embeddings().weight.detach().cpu(), expected_embedding):
                    raise ValueError('Loaded token embeddings differ from the adapter checkpoint')
                self.adapter_embeddings[name] = expected_embedding.clone()
            self.loaded.add(name)
        # Embeddings are shared base weights, not PEFT modules_to_save. Restore
        # them on every switch so a later endpoint cannot contaminate an earlier one.
        with torch.no_grad():
            self.model.get_input_embeddings().weight.copy_(self.adapter_embeddings[name])
        self.model.set_adapter(name)
        self.active_adapter = name
        self.model.eval()
        probabilities = []
        for start in range(0, len(sources), batch_size):
            batch = self.tokenizer(sources[start:start+batch_size], padding=True,
                return_tensors='pt', add_special_tokens=False, truncation=False)
            if batch['input_ids'].shape[1] > 512:
                raise ValueError('SMILES exceeds ChemFM 512-token limit; refusing silent truncation')
            batch = batch.to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits
                probabilities.extend(torch.sigmoid(logits).float().cpu().flatten().tolist())
        return probabilities


@lru_cache(maxsize=1)
def get_model():
    return LocalChemFM()


def predict_local(smiles, property_name):
    if property_name not in ADAPTERS or not smiles:
        raise ValueError('Provide molecules and a supported local ChemFM endpoint')
    with _lock:
        predictions = get_model().predict(smiles, property_name)
    name, revision = ADAPTERS[property_name]
    return {'ok': True, 'property': property_name,
            'results': [{'smiles': s, 'prediction': p, 'unit': 'probability'} for s, p in zip(smiles, predictions)],
            'source': {'base_model': BASE, 'base_revision': BASE_REVISION,
                       'adapter': 'ChemFM/' + name, 'adapter_revision': revision},
            'unit': 'probability', 'validated': False}
