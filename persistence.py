"""Local durable checkpoints and an OS-released SQLite run lease."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

GRAPH_VERSION = 2

@contextmanager
def run_lease(path):
    """Only one process may resume a run; SQLite releases the lease after a crash."""
    connection = sqlite3.connect(str(path) + '.lease', timeout=0)
    try:
        try:
            connection.execute('CREATE TABLE IF NOT EXISTS lease (id INTEGER)')
            connection.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError as exc:
            raise RuntimeError('This run is already open in another process') from exc
        yield
    finally:
        connection.close()

def checkpoint_path(run):
    return run.path.with_suffix('.sqlite')

def validate_resume(run, mode):
    metadata = run.data.get('configuration', {})
    if metadata.get('graph_version') != GRAPH_VERSION:
        raise ValueError('Checkpoint version differs; start a new run (legacy logs cannot be resumed)')
    if metadata.get('mode') != mode:
        raise ValueError('Cannot resume a run in a different execution mode')
    if not checkpoint_path(run).is_file():
        raise ValueError('No durable checkpoint exists for this run')

def collect_attempts(snapshot):
    attempts = dict(snapshot.values.get('attempts', {}))
    for task in snapshot.tasks:
        nested = getattr(task, 'state', None)
        if hasattr(nested, 'values'):
            attempts.update(collect_attempts(nested))
    return attempts

def waiting_for_approval(snapshot):
    """Static interrupts have no error. Failed nodes must retry, not request approval."""
    for task in snapshot.tasks:
        if task.error:
            continue
        nested = getattr(task, 'state', None)
        if hasattr(nested, 'values') and waiting_for_approval(nested):
            return True
    return 'mol_opt_agent' in snapshot.next and not any(t.error for t in snapshot.tasks)
