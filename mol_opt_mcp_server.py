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
            token=os.environ.get("HF_TOKEN") or None,
        )
    return Llama(model_path=model_path)

def _optimize(prompt: str):
    with _inference_lock:
        return get_drugassist().create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
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
        f"Suggest modifications to {action} its {properties} value while maintaining its core structure."
    )
    try:
        message = await asyncio.to_thread(_optimize, prompt)
        return json.dumps({"message": message})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
