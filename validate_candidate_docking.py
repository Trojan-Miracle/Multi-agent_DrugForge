"""Dock the actual DrugGen rule-passing candidates against their requested target."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import hashlib
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKER = '''import json,sys
from pathlib import Path
import docking_module as dm
score,center,size=dm.dock_with_vina(sys.argv[1],sys.argv[2],k=1,threads=2)
Path('result.json').write_text(json.dumps({'score_kcal_mol':score,'center':center,'size':size,'tools':dm.tool_versions()}))
'''


def run(source, output):
    data = json.loads(source.read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    env['PATH'] = str(ROOT / '.tools/bin') + os.pathsep + env['PATH']
    env['JAVA_HOME'] = str(ROOT / '.tools/java25')
    env['P2RANK_PATH'] = str(ROOT / '.tools/p2rank_2.5.1')
    report = {'kind': 'real_generated_candidate_docking', 'target': data['target'],
              'scope': 'Rule-prefiltered DrugGen candidates; top predicted pocket, Vina exhaustiveness=4. Scores are not experimental affinities.',
              'cases': []}
    for i, lead in enumerate(data['selection']['leads'][:3]):
        folder = output / f'candidate-{i+1}'
        folder.mkdir(exist_ok=True)
        started = time.monotonic()
        entry = {'smiles': lead['canonical_smiles']}
        try:
            with subprocess.Popen([sys.executable, '-c', WORKER, data['target'], entry['smiles']],
                    cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, start_new_session=True) as process:
                try:
                    stdout, stderr = process.communicate(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                    (folder / 'execution.log').write_text(stdout + stderr)
                    raise
                (folder / 'execution.log').write_text(stdout + stderr)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)
            entry.update(status='completed', result=json.loads((folder / 'result.json').read_text()))
            entry['structure_inputs'] = [
                {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in folder.glob('*.pdb') if len(path.stem) == 4]
        except subprocess.TimeoutExpired:
            entry.update(status='failed', error='Candidate docking exceeded 300 seconds')
        except subprocess.CalledProcessError as exc:
            entry.update(status='failed', error=exc.stderr[-2000:])
        entry['seconds'] = time.monotonic()-started
        report['cases'].append(entry)
        (output / 'result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('runs/science/druggen/result.json'))
    p.add_argument('--output', type=Path, default=Path('runs/science/candidate-docking'))
    args = p.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
