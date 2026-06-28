from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess, sys, urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 11

REQUIRED_BINARIES = [
    ("vina", "AutoDock Vina executor"),
    ("obabel", "Open Babel converter"),
    ("java", "Java runtime (needed by P2Rank)"),
]

P2RANK_LAUNCHERS = ["pranker", "pranker.bat", "prank", "prank.bat"]

CORE_PKGS = ["torch", "transformers", "rdkit", "autogen", "autogen_agentchat", "autogen_core", "autogen_ext",
             "openai", "mcp", "llama_cpp"]

MCP_SERVERS = [
    "druggen_mcp_server.py",
    "docking_mcp_server.py",
    "chemical_properties_mcp_server.py",
    "admet_prediction_mcp_server.py",
    "mol_opt_mcp_server.py",
    "name2smiles_mcp_server.py",
    "patient_matching_mcp_server.py",
    "trialgen_mcp_server.py",
    "trialpred_mcp_server.py",
]

def ok(x): return "\033[92mOK\033[0m"
def fail(x): return "\033[91mFAIL\033[0m"
def warn(x): return "\033[93mWARN\033[0m"

def which(name): return shutil.which(name) or ""

def check_python():
    v = sys.version_info
    good = (v.major == REQUIRED_PYTHON_MAJOR and v.minor == REQUIRED_PYTHON_MINOR)
    return {"name":"python_version","ok":good,"required":True,"found":f"{v.major}.{v.minor}.{v.micro}",
            "expected":f"{REQUIRED_PYTHON_MAJOR}.{REQUIRED_PYTHON_MINOR}.x"}

def check_os():
    return {"name":"os","ok":True,"required":False,"found":platform.platform()}

def check_bins():
    out=[]
    for b,desc in REQUIRED_BINARIES:
        p=which(b)
        out.append({"name":f"bin:{b}","ok":bool(p),"required":True,"desc":desc,"found":p or "NOT FOUND"})
    return out

def check_p2rank():
    res=[]
    path=os.environ.get("P2RANK_PATH","").strip()
    if not path:
        res.append({"name":"env:P2RANK_PATH","ok":False,"required":True,"found":"NOT SET"})
        return res
    p=Path(path)
    res.append({"name":"env:P2RANK_PATH","ok":p.exists(),"required":True,"found":str(p.resolve())})
    launcher_ok=any((p/x).exists() for x in P2RANK_LAUNCHERS)
    res.append({"name":"p2rank_launcher","ok":launcher_ok,"required":True,
                "found":", ".join(str(p/x) for x in P2RANK_LAUNCHERS)})
    return res

def check_gpu():
    info={"name":"gpu","ok":True,"required":False,"found":"N/A"}
    try:
        import torch
        info["found"]="CUDA available: "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA (CPU/MPS mode)"
    except Exception as e:
        info["ok"]=False; info["found"]=f"PyTorch not importable ({e})"
    return info

def check_core_pkgs():
    res=[]
    for name in CORE_PKGS:
        try:
            mod = __import__(name)
            ver = getattr(mod,"__version__","unknown")
            res.append({"name":f"py:{name}","ok":True,"required":True,"found":ver})
        except Exception as e:
            res.append({"name":f"py:{name}","ok":False,"required":True,"found":f"IMPORT ERROR: {e}"})
    return res

def check_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return {"name":"env:DEEPSEEK_API_KEY","ok":False,"required":True,"found":"NOT SET"}
    return {"name":"env:DEEPSEEK_API_KEY","ok":True,"required":True,"found":f"{key[:8]}..."}

def check_mcp_files():
    missing = [s for s in MCP_SERVERS if not (PROJECT_DIR / s).exists()]
    if missing:
        return {"name":"mcp_server_files","ok":False,"required":True,
                "found":f"缺失: {', '.join(missing)}"}
    return {"name":"mcp_server_files","ok":True,"required":True,
            "found":f"{len(MCP_SERVERS)} 个文件均存在"}

def check_deepseek_api():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return {"name":"deepseek_api","ok":False,"required":True,"found":"跳过（API key 未设置）"}
    try:
        req = urllib.request.Request(
            "https://api.deepseek.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            connected = r.status == 200
    except Exception as e:
        return {"name":"deepseek_api","ok":False,"required":True,"found":str(e)}
    return {"name":"deepseek_api","ok":connected,"required":True,
            "found":"连接正常" if connected else "连接失败"}

def summarize(items):
    return all(x.get("ok") for x in items if x.get("required"))

def pretty(items):
    print("\n=== Prompt-to-Pill Environment Check ===\n")
    for c in items:
        status = ok("") if c["ok"] else (fail("") if c.get("required") else warn(""))
        req = "(required)" if c.get("required") else "(optional)"
        desc = f" — {c['desc']}" if c.get("desc") else ""
        print(f"[{status}] {c['name']} {req}{desc}")
        if c.get("found"): print(f"        found: {c['found']}")
    print("")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json",action="store_true")
    args=ap.parse_args()

    checks=[]
    checks.append(check_python())
    checks.append(check_os())
    checks.extend(check_bins())
    checks.extend(check_p2rank())
    checks.extend(check_core_pkgs())
    checks.append(check_gpu())
    checks.append(check_api_key())
    checks.append(check_mcp_files())
    checks.append(check_deepseek_api())

    ok_all = summarize(checks)
    if args.json:
        print(json.dumps({"ok":ok_all,"checks":checks},indent=2))
    else:
        pretty(checks)
        print("Required checks passed." if ok_all else "One or more required checks failed.")
    sys.exit(0 if ok_all else 1)

if __name__=="__main__":
    main()
