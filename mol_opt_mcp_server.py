"""DrugAssist MCP adapter; weights are loaded on the first tool call."""
import asyncio
import json
import os
from functools import lru_cache
from threading import Lock
from mcp.server.fastmcp import FastMCP

_inference_lock = Lock()

@lru_cache(maxsize=1)
def get_drugassist():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    model_path = os.environ.get("DRUGASSIST_MODEL_PATH")
    if not model_path:
        model_path = hf_hub_download(
            repo_id="blazerye/DrugAssist-7B",
            filename="DrugAssist-7B-4bit.gguf",
            revision="83337f83d30caca6c1dae77dccf8f3f13e119cb7",
            token=os.environ.get("HF_TOKEN") or None,
        )
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4,
                 n_gpu_layers=int(os.environ.get('DRUGASSIST_GPU_LAYERS', '-1')),
                 seed=int(os.environ.get('DRUGASSIST_SEED', '42')), verbose=False)

def _optimize(prompt: str):
    with _inference_lock:
        return get_drugassist().create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0.2,
            response_format={'type': 'json_object', 'schema': {
                'type': 'object', 'properties': {'optimized_smiles': {'type': 'string'}},
                'required': ['optimized_smiles'], 'additionalProperties': False}},
        )["choices"][0]["message"]

mcp = FastMCP("MolOptServer")

@mcp.tool(name="molecule_optimizer")
async def molecule_optimizer(smiles: str, properties: str, action: str):
    if action not in {"increase", "decrease"}:
        return json.dumps({"error": "action must be increase or decrease"})
    if not smiles.strip() or not properties.strip():
        return json.dumps({"error": "smiles and properties must be non-empty"})
    prompt = (
        f"I have a molecule with the SMILES notation {smiles}. "
        f"Suggest modifications to {action} its {properties} value while maintaining its core structure. "
        'Return only a JSON object with the key "optimized_smiles" containing one SMILES string.'
    )
    try:
        message = await asyncio.to_thread(_optimize, prompt)
        result = json.loads(message["content"])
        from rdkit import Chem
        if not isinstance(result.get('optimized_smiles'), str) or Chem.MolFromSmiles(result['optimized_smiles']) is None:
            raise ValueError('Optimizer returned chemically invalid SMILES')
        from contracts import molecule
        molecule(result["optimized_smiles"], "validation", "/optimized_smiles")
        return json.dumps({"optimized_smiles": result["optimized_smiles"], "original_smiles": smiles})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
