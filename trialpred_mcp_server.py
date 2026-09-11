"""Optional MediTab inference. Requires a task-specific checkpoint supplied by the user."""
import json
import os
from functools import lru_cache
from threading import Lock
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TrialPredictionServer")
_lock = Lock()

@lru_cache(maxsize=1)
def get_meditab(checkpoint):
    from MediTab.meditab.bert import BertTabClassifier, BertTabTokenizer
    model = BertTabClassifier.from_pretrained(checkpoint)
    tokenizer = BertTabTokenizer.from_pretrained(checkpoint)
    model.eval()
    return tokenizer, model

@mcp.tool(name="predict_trial_success")
def predict_trial_success(trial_text: str) -> str:
    checkpoint = os.environ.get("MEDITAB_MODEL_PATH", "").strip()
    if not checkpoint:
        return json.dumps({"error": "MEDITAB_MODEL_PATH is not configured. A task-specific checkpoint is required; no probability was generated."})
    if not trial_text.strip():
        return json.dumps({"error": "trial_text must be non-empty"})
    try:
        import torch
        with _lock:
            tok, model = get_meditab(checkpoint)
            inputs = tok([trial_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                if logits.numel() != 1:
                    raise ValueError("Expected a binary single-logit checkpoint with success as the positive label")
                probability = float(torch.sigmoid(logits).detach().cpu().item())
                if not 0 <= probability <= 1:
                    raise ValueError("Model returned a non-finite probability")
        return json.dumps({"success_probability": probability, "checkpoint": checkpoint,
                           "validated": False, "note": "Model estimate; checkpoint calibration has not been verified by DrugForge."})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
