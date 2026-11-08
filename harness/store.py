"""SQLite persistence: run snapshots, approvals, and the retained event log.

Separate from core so the routes stay transport-level; swap this module if the
persistence story ever outgrows a single-file database. Self-contained: takes a
Path, no harness singletons.

Resource-constrained box, two tricks on the hottest table (runs is re-snapshotted
on every mutation, ~2x/sec with a 4-agent mesh):
- WAL journal + synchronous=NORMAL: commits are WAL appends, not fsynced
  journal rebuilds (PRAGMAs persist in the DB file, set once here).
- the nodes column is zlib-compressed: the snapshot is the fat part (every node's
  full result — search dumps, page text, the assembled paper) and JSON compresses
  5-10x. Legacy rows (plain JSON) still load: plain is tried first."""
import json
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any

from .models import Node

EVENT_CAP = 500          # per run default; policy.event_cap overrides (research tool, not Splunk)


def _nodes_blob(raw) -> dict:
    """Decode the nodes column: plain JSON (legacy DBs, test fixtures) first,
    zlib-compressed second."""
    if isinstance(raw, str):
        raw = raw.encode()
    try:
        return json.loads(raw)
    except ValueError:  # includes UnicodeDecodeError: it's a zlib stream
        return json.loads(zlib.decompress(raw))


class Store:
    def __init__(self, path: Path, event_cap: int = EVENT_CAP):
        self.event_cap = max(50, int(event_cap))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL, intent TEXT,"
            "  nodes TEXT, stats TEXT);"
            "CREATE TABLE IF NOT EXISTS approvals (id TEXT PRIMARY KEY, run_id TEXT, node_id TEXT,"
            "  tool TEXT, args TEXT, status TEXT, result TEXT, created REAL);"
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  run_id TEXT, ts REAL, level TEXT, kind TEXT, sender TEXT, receiver TEXT,"
            "  msg TEXT, payload TEXT);"
        )
        try:  # migration for DBs created before approval kinds existed
            self.db.execute("ALTER TABLE approvals ADD COLUMN kind TEXT DEFAULT 'exec'")
        except sqlite3.OperationalError:
            pass
        try:  # migration for DBs created before the staleness epoch existed
            self.db.execute("ALTER TABLE approvals ADD COLUMN attempts INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        for col, ddl in (("phase", "ALTER TABLE runs ADD COLUMN phase TEXT DEFAULT 'executing'"),
                         ("constraints", "ALTER TABLE runs ADD COLUMN constraints TEXT DEFAULT '[]'"),
                         ("prompts", "ALTER TABLE runs ADD COLUMN prompts TEXT DEFAULT '{}'"),
                         ("params", "ALTER TABLE runs ADD COLUMN params TEXT DEFAULT '{}'")):
            try:  # migration for DBs created before the plan-checkpoint fields
                self.db.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self.db.commit()

    def save_run(self, run: dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                        (run["id"], run["created"], run["intent"],
                         zlib.compress(json.dumps(
                             {k: v.model_dump() for k, v in run["nodes"].items()}).encode()),
                         json.dumps(run["stats"]),
                         run.get("phase", "executing"),
                         json.dumps(run.get("constraints", [])),
                         json.dumps(run.get("prompts", {})),
                         json.dumps(run.get("params", {}))))
        self.db.commit()

    def load_runs(self) -> dict[str, dict[str, Any]]:
        out = {}
        for rid, created, intent, nodes, stats, phase, cons, prompts, params \
                in self.db.execute("SELECT * FROM runs"):
            out[rid] = {"id": rid, "created": created, "intent": intent,
                        "phase": phase or "executing", "constraints": json.loads(cons or "[]"),
                        "prompts": json.loads(prompts or "{}"),
                        "params": json.loads(params or "{}"),
                        "nodes": {k: Node(**v) for k, v in _nodes_blob(nodes).items()},
                        "stats": json.loads(stats)}
        return out

    def delete_run(self, rid: str) -> None:
        self.db.execute("DELETE FROM runs WHERE id=?", (rid,))
        self.db.execute("DELETE FROM approvals WHERE run_id=?", (rid,))
        self.db.execute("DELETE FROM events WHERE run_id=?", (rid,))
        self.db.commit()

    def log(self, run_id: str, level: str, kind: str, sender: str, receiver: str,
            msg: str, payload: Any = None) -> dict[str, Any]:
        cur = self.db.execute(
            "INSERT INTO events (run_id, ts, level, kind, sender, receiver, msg, payload)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run_id, time.time(), level, kind, sender, receiver, msg,
             json.dumps(payload, default=str)))
        self.db.execute(  # retention cap
            "DELETE FROM events WHERE run_id=? AND id NOT IN"
            " (SELECT id FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?)",
            (run_id, run_id, self.event_cap))
        self.db.commit()
        return {"id": cur.lastrowid, "run_id": run_id, "ts": time.time(), "level": level,
                "kind": kind, "sender": sender, "receiver": receiver, "msg": msg,
                "payload": payload}

    def events(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return [{"id": r[0], "run_id": r[1], "ts": r[2], "level": r[3], "kind": r[4],
                 "sender": r[5], "receiver": r[6], "msg": r[7],
                 "payload": json.loads(r[8])} for r in self.db.execute(
                    "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id",
                    (run_id, after_id))]

    def save_approval(self, a: dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (a["id"], a["run_id"], a["node_id"], a.get("tool", ""),
                         json.dumps(a["args"]), a["status"],
                         json.dumps(a.get("result")), a["created"], a.get("kind", "exec"),
                         a.get("attempts", 0)))
        self.db.commit()

    def approvals(self, status: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM approvals ORDER BY created DESC", ()
        if status:
            q, args = "SELECT * FROM approvals WHERE status=? ORDER BY created DESC", (status,)
        return [self._ap(r) for r in self.db.execute(q, args)]

    def approval(self, aid: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone()
        return self._ap(row) if row else None

    @staticmethod
    def _ap(r) -> dict[str, Any]:
        return {"id": r[0], "run_id": r[1], "node_id": r[2], "tool": r[3],
                "args": json.loads(r[4]), "status": r[5], "result": json.loads(r[6]),
                "created": r[7], "kind": r[8] if len(r) > 8 else "exec",
                "attempts": r[9] if len(r) > 9 else 0}
