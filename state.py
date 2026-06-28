"""
state.py  ─  运行状态持久化

每次运行生成唯一 run_id，消息实时追加写入 runs/{run_id}.json。
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


class RunState:
    def __init__(self, task: str, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.path = RUNS_DIR / f"{self.run_id}.json"
        self._data = {
            "run_id":      self.run_id,
            "task":        task,
            "started_at":  datetime.now().isoformat(),
            "finished_at": None,
            "status":      "running",
            "messages":    [],
        }
        self._flush()

    def append(self, agent: str, msg_type: str, content: str):
        self._data["messages"].append({
            "id":      len(self._data["messages"]),
            "agent":   agent,
            "type":    msg_type,
            "content": content,
        })
        self._flush()

    def done(self):
        self._data["status"] = "done"
        self._data["finished_at"] = datetime.now().isoformat()
        self._flush()

    def failed(self, reason: str = ""):
        self._data["status"] = "failed"
        self._data["finished_at"] = datetime.now().isoformat()
        if reason:
            self._data["error"] = reason
        self._flush()

    @property
    def messages(self) -> list:
        return self._data["messages"]

    def _flush(self):
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def list_runs(cls) -> list[dict]:
        runs = []
        for f in sorted(RUNS_DIR.glob("*.json"), reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                runs.append({
                    "run_id":     d["run_id"],
                    "task":       d.get("task", "")[:60],
                    "status":     d.get("status"),
                    "started_at": d.get("started_at"),
                    "messages":   len(d.get("messages", [])),
                })
            except Exception:
                pass
        return runs

    @classmethod
    def load(cls, run_id: str) -> dict:
        return json.loads((RUNS_DIR / f"{run_id}.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", metavar="RUN_ID", default=None)
    args = ap.parse_args()

    if args.show:
        data = RunState.load(args.show)
        print(f"\nRun: {data['run_id']}  [{data['status']}]")
        print(f"Task: {data['task']}\n")
        for m in data["messages"]:
            print(f"  [{m['agent']}][{m['type']}] {m['content'][:120].replace(chr(10),' ')}")
    else:
        runs = RunState.list_runs()
        if not runs:
            print("没有历史运行记录")
        else:
            print(f"\n{'run_id':<30} {'status':<8} {'msgs':>4}  task")
            print("-" * 80)
            for r in runs:
                print(f"{r['run_id']:<30} {r['status']:<8} {r['messages']:>4}  {r['task']}")
