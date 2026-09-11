"""Reproducible real-tool checks, separate from the fixture-based workflow demo."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
VINA_SOURCE = 'https://raw.githubusercontent.com/ccsb-scripps/AutoDock-Vina/v1.2.7/example/basic_docking/'
VINA_HASHES = {
    '1iep_receptor.pdbqt': 'f13cf3b36f61d87c3b58983e0b8ecf1c3456a685eb86dfe9ccfb139c7bdc2586',
    '1iep_ligand.pdbqt': '15fb35648d8c18c70317842f3a0631b73a19429c710a037ab07310084d579bb8',
    '1iep_ligand.sdf': '051b8742c32adc05c07fb486a4e7c9327f84e131cee33ac4e6a568d07553eb38',
    '1iep_receptorH.pdb': '5f6aee6029f9a2a2c2be32d4eb948ae70808690573e1b69a0850cdffd7048ca7',
}


def fetch(url, path, expected_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if expected_sha256 and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256:
        return {'url': url, 'sha256': expected_sha256, 'file': path.name}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    if expected_sha256 and hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError('Downloaded input differs from the recorded reference hash')
    path.write_bytes(body)
    return {'url': url, 'sha256': hashlib.sha256(body).hexdigest(), 'file': path.name}


def pose_rmsd(reference, pose):
    """Symmetry-aware heavy-atom RMSD in the *unchanged* receptor frame."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    ref, probe = Chem.RemoveHs(reference), Chem.RemoveHs(pose)
    if ref.GetNumAtoms() != probe.GetNumAtoms():
        raise ValueError('Reference and pose have different heavy-atom counts')
    return float(rdMolAlign.CalcRMS(probe, ref))


def redocking(out, seed=42, pocket='known'):
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate
    import docking_module as dm
    out.mkdir(parents=True, exist_ok=True)
    sources = [fetch(VINA_SOURCE + name, out / Path(name).name, VINA_HASHES[Path(name).name]) for name in (
        'solution/1iep_receptor.pdbqt', 'solution/1iep_ligand.pdbqt',
        'data/1iep_ligand.sdf', 'data/1iep_receptorH.pdb')]
    # Published tutorial box; this branch does not evaluate pocket prediction.
    center, size = [15.190, 53.903, 16.917], [20, 20, 20]
    if pocket == 'p2rank':
        pockets = dm.run_p2rank_list(str(out / '1iep_receptorH.pdb'), '1IEP',
                                    out_root=str(out / 'p2rank'))
        if not pockets:
            raise RuntimeError('P2Rank returned no pockets')
        top = pockets[0]
        center = [top[k] for k in ('cx', 'cy', 'cz')]
        size = [top['size']] * 3
    cmd = ['vina', '--receptor', '1iep_receptor.pdbqt', '--ligand', '1iep_ligand.pdbqt',
           '--exhaustiveness', '32', '--seed', str(seed), '--cpu', '2',
           '--num_modes', '9', '--out', 'poses.pdbqt']
    for axis, c, s in zip('xyz', center, size):
        cmd += ['--center_' + axis, str(c), '--size_' + axis, str(s)]
    started = time.monotonic()
    cp = subprocess.run(cmd, cwd=out, text=True, capture_output=True, timeout=600)
    (out / 'vina.log').write_text(cp.stdout + cp.stderr)
    cp.check_returncode()
    # Meeko restores bond orders using the input SMILES atom mapping.
    poses = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule.from_file(str(out / 'poses.pdbqt'), skip_typing=True))
    ref = Chem.SDMolSupplier(str(out / '1iep_ligand.sdf'), removeHs=True)[0]
    if ref is None or len(poses) != 1 or poses[0] is None:
        raise ValueError('Cannot reconstruct reference or docked ligand')
    pose = Chem.Mol(poses[0])
    first = Chem.Conformer(pose.GetConformer(0))
    pose.RemoveAllConformers()
    pose.AddConformer(first, assignId=True)
    rmsd = pose_rmsd(ref, pose)
    result = {'kind': 'experimental_pose_redocking', 'case': '1IEP', 'pocket': pocket,
              'seed': seed, 'exhaustiveness': 32, 'center': center, 'size': size,
              'top1_heavy_atom_rmsd_angstrom': rmsd, 'top1_below_2_angstrom': rmsd < 2,
              'vina_score_kcal_mol': dm.vina_parse_top_score(str(out / 'vina.log')),
              'elapsed_seconds': time.monotonic() - started, 'tools': dm.tool_versions(),
              'sources': sources, 'command': cmd,
              'scope': 'Single public tutorial case; not a binding affinity or drug efficacy benchmark.'}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    return result


async def agent_smoke(out):
    """One bounded real DeepSeek + MCP + RDKit tool-use check; no fixture tools."""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from workflow_runtime import usage_from_messages, content_text
    from langchain_core.messages import ToolMessage
    model = os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')
    llm = ChatOpenAI(model=model, api_key=os.environ['DEEPSEEK_API_KEY'],
                     base_url='https://api.deepseek.com/v1', max_tokens=1024, max_retries=0,
                     timeout=90, extra_body={'thinking': {'type': 'disabled'}})
    client = MultiServerMCPClient({'chemical': {'command': sys.executable,
        'args': [str(ROOT / 'chemical_properties_mcp_sever.py')], 'transport': 'stdio'}})
    async with client.session('chemical') as session:
        tools = [t for t in await load_mcp_tools(session) if t.name == 'rdkit_physchem_batch']
        agent = create_agent(llm, tools, system_prompt='Call the provided tool exactly once for the requested molecules, then briefly summarize its actual output.')
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content':
            'Calculate RDKit properties for aspirin CC(=O)Oc1ccccc1C(=O)O and ethanol CCO.'}]},
            config={'recursion_limit': 6})
    messages = result['messages']
    evidence = [content_text(m.content) for m in messages if isinstance(m, ToolMessage)]
    if len(evidence) != 1:
        raise RuntimeError('Expected exactly one actual RDKit tool result')
    usage = usage_from_messages(messages)
    from workflow_runtime import contains_error
    from contracts import decode_payload
    successful = all(m.status != 'error' for m in messages if isinstance(m, ToolMessage))
    successful = successful and all(not contains_error(decode_payload(e)) for e in evidence)
    report = {'kind': 'live_llm_mcp_rdkit_smoke', 'model': model, 'tool_outputs': evidence,
              'status': 'completed' if successful else 'failed',
              'llm_calls': usage, 'summary': content_text(messages[-1].content),
              'scope': 'Real API and real RDKit; not a full drug-discovery run.'}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=['redocking', 'agent'])
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'science')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pocket', choices=['known', 'p2rank'], default='known')
    args = parser.parse_args()
    tools = ROOT / '.tools'
    os.environ['PATH'] = str(tools / 'bin') + os.pathsep + os.environ['PATH']
    if (tools / 'java25' / 'bin' / 'java').exists():
        os.environ['JAVA_HOME'] = str(tools / 'java25')
    if (tools / 'p2rank_2.5.1').exists():
        os.environ.setdefault('P2RANK_PATH', str(tools / 'p2rank_2.5.1'))
    out = args.output.resolve()
    result = redocking(out, args.seed, args.pocket) if args.check == 'redocking' else asyncio.run(agent_smoke(out))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
