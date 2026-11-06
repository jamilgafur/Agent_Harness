"""Core: DAG store + validation, approvals, checkpointing (SQLite), exec mediator,
agent registry (heartbeat/position/phase), persisted event log (paper III-B),
claim leases, tool registry, WebSocket feed, static UI.

Main loop shapes:
  POST /runs {intent|nodes}            -> run enters the claimable set
  POST /claim                          -> atomic ready-leaf handout (Alg-2 routing, locality)
  POST /runs/{id}/nodes/{nid}/decompose-> validated children (approval-parked or applied)
  POST /runs/{id}/exec                 -> sandbox tool call (approval-parked or executed)
  POST /runs/{id}/nodes/{nid}/result   -> DONE (settles parents) / FAILED (repair loop, then escalate)
Approvals hold runs steady; the budget governor (max_llm_calls/max_seconds) terminates them.
Approvals gate the UI, they are NOT an auth boundary: bind to localhost in compose if
anything untrusted shares the LAN. Single process; asyncio.Lock makes mutations atomic."""
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import load
from .graph import apply_patch, descendants, ready_nodes, validate
from .models import Patch, Node, NodeState, NodeType

CFG = load()
POLICY: Dict[str, Any] = CFG["policy"]
TOOLS: Dict[str, Dict[str, Any]] = {
    "shell": {"desc": "run a shell command in the shared sandbox workspace",
              "required": ["cmd"]},
    "read_file": {"desc": "read a file from the workspace", "required": ["path"]},
    "write_file": {"desc": "write a file in the workspace", "required": ["path", "content"]},
    "list_dir": {"desc": "list a directory in the workspace", "required": ["path"]},
    "web_search": {"desc": "search the web (SearXNG); returns [{title,url,snippet}]",
                   "required": ["query"]},
    "http_fetch": {"desc": "fetch a URL and return readable text",
                   "required": ["url"]},
    "llm": {"desc": "virtual (agent-side): the LLM writes an answer using its deps' "
                    "results as material; returns {text}. No args.",
            "required": []},
}
EVENT_CAP = 500          # per run; research tool, not Splunk
AGENT_STALE_S = 15       # heartbeat older than this -> agent shown dead
DB = Path(CFG["data_dir"]) / "harness.db"
app = FastAPI(title="harness-core")
_lock = asyncio.Lock()
_ws: List[WebSocket] = []
# agent_id -> liveness/position/phase; in-memory on purpose (dead agents die with core)
AGENTS: Dict[str, Dict[str, Any]] = {}


def tool_arg_errors(nodes: Dict[str, Node]) -> List[str]:
    """Args schema check: small local models invent arg names; this turns it into
    a validation error they can repair."""
    return [f"{n.id}: tool {n.tool!r} missing required arg {k!r}"
            for n in nodes.values()
            if n.tool and n.tool in TOOLS
            for k in TOOLS[n.tool]["required"] if k not in n.args]


# ---------- store: JSON snapshots + events/mailbox log ----------
class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
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
        for col, ddl in (("phase", "ALTER TABLE runs ADD COLUMN phase TEXT DEFAULT 'executing'"),
                         ("constraints", "ALTER TABLE runs ADD COLUMN constraints TEXT DEFAULT '[]'")):
            try:  # migration for DBs created before the plan-checkpoint fields
                self.db.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self.db.commit()

    def save_run(self, run: Dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
                        (run["id"], run["created"], run["intent"],
                         json.dumps({k: v.model_dump() for k, v in run["nodes"].items()}),
                         json.dumps(run["stats"]),
                         run.get("phase", "executing"),
                         json.dumps(run.get("constraints", []))))
        self.db.commit()

    def load_runs(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for rid, created, intent, nodes, stats, phase, cons in self.db.execute("SELECT * FROM runs"):
            out[rid] = {"id": rid, "created": created, "intent": intent,
                        "phase": phase or "executing", "constraints": json.loads(cons or "[]"),
                        "nodes": {k: Node(**v) for k, v in json.loads(nodes).items()},
                        "stats": json.loads(stats)}
        return out

    def delete_run(self, rid: str) -> None:
        self.db.execute("DELETE FROM runs WHERE id=?", (rid,))
        self.db.execute("DELETE FROM approvals WHERE run_id=?", (rid,))
        self.db.execute("DELETE FROM events WHERE run_id=?", (rid,))
        self.db.commit()

    def log(self, run_id: str, level: str, kind: str, sender: str, receiver: str,
            msg: str, payload: Any = None) -> Dict[str, Any]:
        cur = self.db.execute(
            "INSERT INTO events (run_id, ts, level, kind, sender, receiver, msg, payload)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run_id, time.time(), level, kind, sender, receiver, msg,
             json.dumps(payload, default=str)))
        self.db.execute(  # retention cap
            "DELETE FROM events WHERE run_id=? AND id NOT IN"
            " (SELECT id FROM events WHERE run_id=? ORDER BY id DESC LIMIT ?)",
            (run_id, run_id, EVENT_CAP))
        self.db.commit()
        return {"id": cur.lastrowid, "run_id": run_id, "ts": time.time(), "level": level,
                "kind": kind, "sender": sender, "receiver": receiver, "msg": msg,
                "payload": payload}

    def events(self, run_id: str, after_id: int = 0) -> List[Dict[str, Any]]:
        return [{"id": r[0], "run_id": r[1], "ts": r[2], "level": r[3], "kind": r[4],
                 "sender": r[5], "receiver": r[6], "msg": r[7],
                 "payload": json.loads(r[8])} for r in self.db.execute(
                    "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id",
                    (run_id, after_id))]

    def save_approval(self, a: Dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO approvals VALUES (?,?,?,?,?,?,?,?,?)",
                        (a["id"], a["run_id"], a["node_id"], a.get("tool", ""),
                         json.dumps(a["args"]), a["status"],
                         json.dumps(a.get("result")), a["created"], a.get("kind", "exec")))
        self.db.commit()

    def approvals(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM approvals ORDER BY created DESC", ()
        if status:
            q, args = "SELECT * FROM approvals WHERE status=? ORDER BY created DESC", (status,)
        return [self._ap(r) for r in self.db.execute(q, args)]

    def approval(self, aid: str) -> Optional[Dict[str, Any]]:
        row = self.db.execute("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone()
        return self._ap(row) if row else None

    @staticmethod
    def _ap(r) -> Dict[str, Any]:
        return {"id": r[0], "run_id": r[1], "node_id": r[2], "tool": r[3],
                "args": json.loads(r[4]), "status": r[5], "result": json.loads(r[6]),
                "created": r[7], "kind": r[8] if len(r) > 8 else "exec"}


store = Store(DB)
RUNS: Dict[str, Dict[str, Any]] = store.load_runs()


# ---------- helpers ----------
def _active(run: Dict[str, Any]) -> bool:
    return any(n.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                           NodeState.WAITING) for n in run["nodes"].values())


def _ancestors(run: Dict[str, Any], nid: str) -> set:
    out, stack = set(), list(run["nodes"][nid].deps)
    while stack:
        cur = stack.pop()
        if cur not in out and cur in run["nodes"]:
            out.add(cur)
            stack.extend(run["nodes"][cur].deps)
    return out


def _subtree(run: Dict[str, Any], nid: str) -> List[str]:
    """nid plus every node below it via PARENT links (decomposition tree),
    which is the relation replan deletes along — dep edges ignore parents."""
    out, frontier = [], [nid]
    while frontier:
        cur = frontier.pop()
        kids = [n.id for n in run["nodes"].values() if n.parent == cur]
        out += kids
        frontier += kids
    return out


def _plan_ready(run: Dict[str, Any]) -> bool:
    """Planning checkpoint reached: every abstract node plannable NOW has been
    decomposed down to atomic tool leaves. Abstract nodes gated behind not-yet-run
    tool deps can't be worked during planning (their context arrives with the
    results) — they refine after /go and must NOT block the checkpoint."""
    if run.get("phase") != "planning":
        return False
    nodes = run["nodes"]
    for n in nodes.values():
        if n.tool or n.state not in (NodeState.PENDING, NodeState.RUNNING,
                                     NodeState.REPAIRING):
            continue
        gated = any(nodes[d].tool for d in n.deps if d in nodes)
        if not gated:
            return False
    return True


def _pending_approvals(run_id: str) -> set:
    return {a["node_id"] for a in store.approvals("pending") if a["run_id"] == run_id}


async def _broadcast(msg: Dict[str, Any]) -> None:
    dead = []
    for ws in _ws:
        try:
            await ws.send_text(json.dumps(msg, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws.remove(ws)


async def _event(run_id: str, level: str, kind: str, sender: str, receiver: str,
                 msg: str, payload: Any = None) -> None:
    e = store.log(run_id, level, kind, sender, receiver, msg, payload)
    await _broadcast({"type": "log", "event": e})


async def _state(run_id: str) -> None:
    run = RUNS[run_id]
    await _broadcast({"type": "state", "run_id": run_id,
                      "nodes": {k: v.model_dump(mode="json") for k, v in run["nodes"].items()},
                      "stats": run["stats"], "active": _active(run),
                      "phase": run.get("phase", "executing"),
                      "constraints": run.get("constraints", []),
                      "plan_ready": _plan_ready(run),
                      "pending": sorted(_pending_approvals(run_id))})


def _fail_cascade(run: Dict[str, Any], nid: str, hint: str, state: NodeState) -> None:
    """Terminal failure: node -> state, descendants -> BLOCKED, parent inherits."""
    node = run["nodes"][nid]
    node.state, node.hint = state, hint
    for d in descendants(run["nodes"], nid):
        dn = run["nodes"][d]
        if dn.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                        NodeState.WAITING):
            dn.state, dn.hint = NodeState.BLOCKED, f"upstream {nid} {state.value.lower()}"
    if node.parent and node.parent in run["nodes"]:
        p = run["nodes"][node.parent]
        if p.state != NodeState.DONE:
            _fail_cascade(run, node.parent, f"child {nid} {state.value.lower()}: {hint}", state)


def _settle(run: Dict[str, Any], nid: str) -> None:
    """After DONE: complete parents whose children are all DONE (bottom-up).
    The parent takes its writer child's text (the sink = most deps), so the train
    of thought climbs: raw results stay on leaves, summaries rise level by level,
    and the head node ends holding the final answer itself."""
    node = run["nodes"][nid]
    if node.parent and node.parent in run["nodes"]:
        p = run["nodes"][node.parent]
        kids = [n for n in run["nodes"].values() if n.parent == p.id]
        if kids and all(k.state == NodeState.DONE for k in kids):
            p.state = NodeState.DONE
            writer = max((k for k in kids
                          if isinstance(k.result, dict) and k.result.get("text")),
                         key=lambda k: len(k.deps), default=None)
            p.result = ({"text": writer.result["text"], "from": writer.id}
                        if writer else {k.id: k.result for k in kids})
            _settle(run, p.id)


async def _exec_tool(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Core mediates every tool call to the sandbox. Static tools have fixed
    endpoints; anything else is a dynamic tool from /workspace/tools (P2)."""
    if tool == "llm":   # virtual, agent-side only — never sandboxed
        return {"rc": -7, "stderr": "llm is agent-side; not executable via the sandbox"}
    path = {"shell": "/shell", "read_file": "/read_file", "write_file": "/write_file",
            "list_dir": "/list_dir", "web_search": "/web_search",
            "http_fetch": "/http_fetch"}.get(tool, "/run_tool")
    payload = args if path != "/run_tool" else {"name": tool, "args": args}
    try:
        async with httpx.AsyncClient(timeout=70) as c:
            r = await c.post(CFG["sandbox_url"] + path, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        return {"rc": -3, "stdout": "",
                "stderr": f"sandbox {e.response.status_code}: {e.response.text[:300]}"}
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return {"rc": -4, "stdout": "", "stderr": f"sandbox unreachable: {type(e).__name__}"}


def _budget_breach(run: Dict[str, Any]) -> str:
    """Budget governor: returns a reason when the run exceeded its budgets."""
    if run["stats"]["llm_calls"] >= POLICY["max_llm_calls"]:
        return f"max_llm_calls={POLICY['max_llm_calls']} reached"
    if time.time() - run["created"] >= POLICY["max_seconds"]:
        return f"max_seconds={POLICY['max_seconds']} reached"
    return ""


def _terminate(run: Dict[str, Any], reason: str) -> None:
    """Governor stop: every live node ESCALATED -> run leaves the claimable set."""
    for n in run["nodes"].values():
        if n.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                       NodeState.WAITING):
            n.state, n.hint = NodeState.ESCALATED, f"budget governor: {reason}"
            n.claimed_at = 0.0


def _requeue_stale(run: Dict[str, Any]) -> None:
    """Lease sweep: RUNNING nodes past the lease (no pending approval) go back to PENDING."""
    lease = POLICY["claim_lease_seconds"]
    pend = _pending_approvals(run["id"])
    for n in run["nodes"].values():
        if (n.state == NodeState.RUNNING and n.claimed_at
                and time.time() - n.claimed_at > lease and n.id not in pend):
            n.state, n.hint = NodeState.PENDING, "lease expired (agent died?)"
            n.claimed_at = 0.0
            store.log(run["id"], "warn", "lease", "core", "-",
                      f"{n.id}: lease expired, requeued")


# ---------- request bodies ----------
class RunIn(BaseModel):
    intent: str = ""
    nodes: List[Node] = []


class ClaimIn(BaseModel):
    agent_id: str
    max_nodes: int = 1     # parallel ready-set: claim up to this many leaves at once


class ResultIn(BaseModel):
    state: str               # DONE | FAILED
    result: Optional[Any] = None
    hint: str = ""
    llm_calls: int = 0
    agent_id: str = ""


class ChildIn(BaseModel):
    intent: str
    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    deps: List[Any] = []    # sibling indices (agent) or sibling ids (approval round-trip)
    success_predicate: str = ""


class DecomposeIn(BaseModel):
    children: List[ChildIn]


class RedirectIn(BaseModel):
    agent_id: str
    reason: str = ""


class ExecIn(BaseModel):
    node_id: str
    tool: str
    args: Dict[str, Any] = {}


class DecisionIn(BaseModel):
    decision: str            # approve | reject
    args: Optional[Dict[str, Any]] = None   # exec: edited args; patch: {"children": [...]}


class HeartbeatIn(BaseModel):
    agent_id: str
    run_id: str = ""
    node_id: str = ""
    phase: str = "idle"


def _build_children(nodes: Dict[str, Node], nid: str,
                    raw: List[Dict[str, Any]]) -> Tuple[List[Node], List[str]]:
    """Validate + instantiate decomposition children. Errors here feed the LLM repair loop."""
    if nid not in nodes:
        return [], [f"unknown node {nid!r}"]
    parent = nodes[nid]
    if parent.state not in (NodeState.RUNNING, NodeState.PENDING, NodeState.REPAIRING):
        return [], [f"cannot decompose {nid} in state {parent.state.value}"]
    if not raw:
        return [], ["decompose needs at least one child"]
    try:
        parsed = [ChildIn(**c) for c in raw]
    except Exception as e:
        return [], [f"child schema error: {e}"]
    errors: List[str] = []
    children: List[Node] = []
    for i, ch in enumerate(parsed):
        resolved: List[str] = []
        for j in ch.deps:
            if isinstance(j, int):
                if not 0 <= j < i:
                    errors.append(f"child {i}: dep index {j} must point to an earlier sibling")
                else:
                    resolved.append(f"{nid}.{j + 1}")
            elif isinstance(j, str) and any(c.id == j for c in children):
                resolved.append(j)          # parked approval form: already-resolved id
            else:
                errors.append(f"child {i}: dep {j!r} must be an earlier sibling index or id")
        # small models write "tool": "null" (string) — treat null-ish as abstract
        tool = None if (ch.tool or "").strip().lower() in ("null", "none", "") else ch.tool
        children.append(Node(
            id=f"{nid}.{i + 1}", intent=ch.intent, tool=tool, args=ch.args if tool else {},
            deps=list(parent.deps) + resolved,
            success_predicate=ch.success_predicate, depth=parent.depth + 1, parent=nid))
    errors += tool_arg_errors({c.id: c for c in children})
    return ([] if errors else children), errors


def _apply_decompose(run: Dict[str, Any], nid: str, children: List[Node]) -> List[str]:
    """Validate as a patch and apply; parent -> WAITING. Caller holds the lock."""
    if run.get("phase") == "planning":
        # plan-first: an abstract child must be plannable NOW, from neighbor
        # context — result-gated planning (deps on tools) waits for /go,
        # so strip tool deps off abstract children
        for c in children:
            if c.tool:
                continue
            c.deps = [d for d in c.deps
                      if not next((x.tool for x in children if x.id == d),
                                  run["nodes"].get(d) and run["nodes"][d].tool)]
    patch = Patch(ops=[{"op": "add_node", "node": c.model_dump()} for c in children],
                  reason=f"decompose {nid}")
    new_nodes, errs = apply_patch(run["nodes"], patch, TOOLS, POLICY["max_depth"],
                                  POLICY["max_nodes"], POLICY["auto_approve"])
    if errs:
        return errs
    new_nodes[nid].state = NodeState.WAITING
    run["nodes"] = new_nodes
    run["stats"]["patches"] += 1
    store.save_run(run)
    return []


def _agents_public() -> Dict[str, Any]:
    now = time.time()
    return {aid: {**a, "age": round(now - a["last_seen"], 1),
                  "alive": now - a["last_seen"] < AGENT_STALE_S}
            for aid, a in AGENTS.items()}
# ---------- endpoints: infra ----------
@app.get("/health")
async def health() -> Dict[str, Any]:
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            ollama_ok = (await c.get(CFG["ollama_url"] + "/api/tags")).status_code == 200
    except Exception:
        pass
    return {"ok": True, "ollama": ollama_ok, "runs": len(RUNS), "agents": len(AGENTS)}


@app.get("/tools")
def tools() -> Dict[str, Any]:
    return {"tools": TOOLS, "auto_approve": POLICY["auto_approve"]}


@app.post("/tools/sync")
async def tools_sync() -> Dict[str, Any]:
    """Pull dynamic tools (P2) from the sandbox into the registry. Repeatable:
    re-sync after dropping or editing a .py in /workspace/tools."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(CFG["sandbox_url"] + "/discover_tools")
            r.raise_for_status()
            found = r.json().get("tools", {})
    except Exception as e:
        raise HTTPException(502, f"sandbox: {type(e).__name__}")
    for name in [k for k, v in TOOLS.items() if v.get("dynamic") and k not in found]:
        del TOOLS[name]                # hot-reload removes vanished tools too
    for name, meta in found.items():
        TOOLS[name] = {"desc": meta.get("desc", f"dynamic tool ({meta.get('file')})"),
                       "required": meta.get("required", []), "dynamic": True}
    return {"synced": sorted(found), "tools": TOOLS}


@app.get("/policy")
def get_policy() -> Dict[str, Any]:
    return POLICY


@app.get("/autosize")
async def autosize() -> Dict[str, Any]:
    """Autosizer (P3): suggest max_agents from measured LLM latency.
    shortcut: fixed latency bands; a real autosizer also watches queue depth
    and VRAM — upgrade if the mesh ever shares the GPU with other services."""
    t0 = time.time()
    ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            ok = (await c.get(CFG["ollama_url"] + "/api/tags")).status_code == 200
    except Exception:
        pass
    ms = round((time.time() - t0) * 1000)
    suggested = 1 if not ok else (8 if ms < 200 else 4 if ms < 600 else 2 if ms < 1500 else 1)
    return {"ollama_ok": ok, "ping_ms": ms if ok else None,
            "suggested_max_agents": suggested, "current": POLICY["max_agents"]}


@app.post("/policy")
async def set_policy(body: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"max_agents", "approve_dag_edits", "claim_lease_seconds",
               "confirm_actions", "auto_approve"}
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(400, f"not editable at runtime: {sorted(unknown)}")
    POLICY.update(body)
    await _broadcast({"type": "policy", "policy": POLICY})
    return POLICY


@app.post("/heartbeat")
async def heartbeat(hb: HeartbeatIn) -> Dict[str, Any]:
    AGENTS[hb.agent_id] = {"last_seen": time.time(), "run_id": hb.run_id,
                           "node_id": hb.node_id, "phase": hb.phase}
    await _broadcast({"type": "agents", "agents": _agents_public()})
    return {"ok": True}


@app.get("/agents")
def list_agents() -> Dict[str, Any]:
    return _agents_public()


# ---------- endpoints: runs ----------
@app.post("/runs")
async def create_run(body: RunIn) -> Dict[str, Any]:
    if body.nodes:
        nodes = {n.id: n for n in body.nodes}
        if len(nodes) != len(body.nodes):
            raise HTTPException(400, ["duplicate node ids"])
        errs = validate(nodes, TOOLS, POLICY["max_depth"], POLICY["max_nodes"],
                        POLICY["auto_approve"])
        errs += tool_arg_errors(nodes)
        if errs:
            raise HTTPException(400, errs)
    elif body.intent.strip():
        nodes = {"root": Node(id="root", intent=body.intent.strip())}
    else:
        raise HTTPException(400, ["provide either intent or nodes (template DAG)"])
    # intent runs are LLM-decomposed: plan first (checkpoint), user-approved /go
    # starts the bottom-up execution. Template DAGs are pre-planned: run at once.
    run = {"id": uuid.uuid4().hex[:8], "created": time.time(),
           "intent": body.intent or f"template ({len(nodes)} nodes)", "nodes": nodes,
           "phase": "executing" if body.nodes else "planning", "constraints": [],
           "stats": {"llm_calls": 0, "execs": 0, "approvals": 0, "patches": 0}}
    async with _lock:
        RUNS[run["id"]] = run
        store.save_run(run)
    store.log(run["id"], "info", "run", "user", "core", f"run created: {run['intent'][:80]}")
    await _broadcast({"type": "run", "run_id": run["id"], "intent": run["intent"]})
    await _state(run["id"])
    return {"run_id": run["id"]}


@app.get("/runs")
def list_runs() -> List[Dict[str, Any]]:
    return [{"id": r["id"], "intent": r["intent"], "created": r["created"],
             "active": _active(r), "phase": r.get("phase", "executing"),
             "constraints": r.get("constraints", []), "plan_ready": _plan_ready(r),
             "states": {s.value: sum(1 for n in r["nodes"].values() if n.state == s)
                        for s in NodeState}} for r in sorted(RUNS.values(), key=lambda x: -x["created"])]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    return {"id": run["id"], "intent": run["intent"], "active": _active(run),
            "phase": run.get("phase", "executing"),
            "constraints": run.get("constraints", []), "plan_ready": _plan_ready(run),
            "nodes": {k: v.model_dump(mode="json") for k, v in run["nodes"].items()},
            "stats": run["stats"], "policy": POLICY,
            "pending": sorted(_pending_approvals(run_id))}


class ConstraintIn(BaseModel):
    constraints: List[str]


class AddedNode(BaseModel):
    intent: str
    parent: str = "root"


class NodeUpdate(BaseModel):
    intent: str


@app.post("/runs/{run_id}/nodes")
async def add_node(run_id: str, body: AddedNode) -> Dict[str, Any]:
    """User adds a branch at the planning checkpoint. It enters ABSTRACT (no tool):
    an agent's LLM claims it and decomposes it against its neighbor context —
    the user's rough idea becomes a fleshed-out sub-tree."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    if run.get("phase") != "planning":
        raise HTTPException(409, "tree edits only at the planning checkpoint")
    intent = " ".join(body.intent.split())[:300]
    if not intent:
        raise HTTPException(400, "empty intent")
    if body.parent not in run["nodes"]:
        raise HTTPException(404, f"unknown parent node {body.parent!r}")
    if len(run["nodes"]) >= POLICY["max_nodes"]:
        raise HTTPException(400, f"node cap reached ({POLICY['max_nodes']})")
    async with _lock:
        i = 1
        while f"{body.parent}.{i}" in run["nodes"]:
            i += 1
        nid = f"{body.parent}.{i}"
        p = run["nodes"][body.parent]
        run["nodes"][nid] = Node(id=nid, intent=intent, deps=list(p.deps),
                                 depth=p.depth + 1, parent=body.parent)
        # keep the convergence invariant: the parent's sink (writer with most
        # deps) must consume EVERY sibling, user-added ones included
        sibs = [n for n in run["nodes"].values() if n.parent == body.parent]
        sink = max((n for n in sibs if n.tool == "llm" and n.deps),
                   key=lambda n: len(n.deps), default=None)
        if sink and nid not in sink.deps:
            sink.deps = [n.id for n in sibs if n.id != sink.id]
        store.save_run(run)
    await _event(run_id, "info", "plan", "user", "core",
                 f"branch added by user: {nid} — {intent[:60]}")
    await _state(run_id)
    return {"ok": True, "id": nid}


@app.patch("/runs/{run_id}/nodes/{nid}")
async def update_node(run_id: str, nid: str, body: NodeUpdate) -> Dict[str, Any]:
    """User rewrites a node's instruction while the plan is still open."""
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    if run.get("phase") != "planning":
        raise HTTPException(409, "tree edits only at the planning checkpoint")
    n = run["nodes"][nid]
    if n.state not in (NodeState.PENDING, NodeState.WAITING):
        raise HTTPException(409, f"cannot edit {nid} in state {n.state.value}")
    intent = " ".join(body.intent.split())[:300]
    if not intent:
        raise HTTPException(400, "empty intent")
    n.intent = intent
    store.save_run(run)
    await _event(run_id, "info", "plan", "user", "core",
                 f"node {nid} reworded: {intent[:60]}")
    await _state(run_id)
    return {"ok": True}


@app.put("/runs/{run_id}/constraints")
async def set_constraints(run_id: str, body: ConstraintIn) -> Dict[str, Any]:
    """User plan feedback (the 'ignore methodology / discuss X' box): stored on the
    run and injected into every decomposition and synthesis prompt from then on."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    clean = []
    for c in body.constraints:
        c = " ".join(str(c).split())[:300]
        if c and c not in clean:
            clean.append(c)
    if len(clean) > 12:
        raise HTTPException(400, "too many constraints (max 12)")
    run["constraints"] = clean
    store.save_run(run)
    store.log(run_id, "info", "plan", "user", "core",
              "constraints set: " + ("; ".join(clean) if clean else "(none)"))
    await _state(run_id)
    return {"ok": True, "constraints": clean}


@app.post("/runs/{run_id}/go")
async def go(run_id: str) -> Dict[str, Any]:
    """User approved the plan: execution starts bottom-up from the leaves."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    if run.get("phase") != "planning":
        raise HTTPException(409, "run is not at the planning checkpoint")
    run["phase"] = "executing"
    store.save_run(run)
    store.log(run_id, "info", "plan", "user", "core",
              f"plan approved ({len(run['nodes'])} nodes, "
              f"{'ready' if _plan_ready(run) else 'partial'}) — execution starts")
    await _state(run_id)
    return {"ok": True}


@app.post("/runs/{run_id}/nodes/{nid}/replan")
async def replan(run_id: str, nid: str) -> Dict[str, Any]:
    """'Break this down further': during planning, turn a node back into an
    abstract planner task (deleting its subtree) so an agent re-decomposes it
    against the current constraints."""
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    if run.get("phase") != "planning":
        raise HTTPException(409, "replan only at the planning checkpoint "
                                 "(after /go, use fix & retry instead)")
    async with _lock:
        kids = _subtree(run, nid)
        if any(run["nodes"][d].state == NodeState.RUNNING for d in kids):
            raise HTTPException(409, "a planner is working inside this subtree — wait")
        n = run["nodes"][nid]
        if n.state not in (NodeState.PENDING, NodeState.WAITING):
            raise HTTPException(409, f"cannot replan {nid} in state {n.state.value}")
        for d in kids:
            del run["nodes"][d]
        n.state, n.tool, n.args, n.result, n.hint = NodeState.PENDING, None, {}, None, ""
        # same rule as _apply_decompose: plannable NOW (tool deps wait for /go)
        n.deps = [d for d in n.deps if d in run["nodes"] and not run["nodes"][d].tool]
        store.save_run(run)
    await _event(run_id, "info", "plan", "user", "core",
                 f"replan: {nid} goes back to the planner ({len(kids)} descendants removed)")
    await _state(run_id)
    return {"ok": True}


@app.get("/runs/{run_id}/events")
def get_events(run_id: str, after_id: int = 0) -> List[Dict[str, Any]]:
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    return store.events(run_id, after_id)


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str, force: bool = False) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    if _active(run):
        if not force:
            raise HTTPException(409, "run is still active; pass force=1 to stop and delete")
        _terminate(run, "deleted by user")   # stop the claim/heartbeat dance before removal
    async with _lock:
        del RUNS[run_id]
        store.delete_run(run_id)
    await _broadcast({"type": "run_deleted", "run_id": run_id})
    return {"ok": True}


# ---------- endpoints: agent loop ----------
@app.post("/claim")
async def claim(body: ClaimIn) -> Dict[str, Any]:
    terminated: List[str] = []
    async with _lock:
        candidates = []  # (run, node) ready across active runs
        for run in sorted(RUNS.values(), key=lambda x: x["created"]):
            if not _active(run):
                continue
            _requeue_stale(run)
            breach = _budget_breach(run)
            if breach:
                _terminate(run, breach)
                store.save_run(run)
                store.log(run["id"], "error", "budget", "core", "-",
                          f"run terminated: {breach}")
                terminated.append(run["id"])
                continue
            cands = ready_nodes(run["nodes"])
            if run.get("phase") == "planning":
                # plan-first: only abstract nodes (further decomposition) are
                # claimable; tool work waits for the user's /go
                cands = [n for n in cands if not n.tool]
            candidates += [(run, n) for n in cands]
        if candidates:
            # Algorithm 2 routing: exclude nodes where this agent is in the
            # visited set (it redirected them away); if nothing remains, ⊥
            fresh = [c for c in candidates if body.agent_id not in c[1].routed_past]
            if fresh:
                candidates = fresh
            else:
                candidates = []
        if not candidates:
            picked: List[Tuple[Dict[str, Any], Node]] = []
        else:
            # locality: prefer siblings of the node this agent last worked on (answer 7:
            # agents grow their branch then move up); else deepest ready node
            first = None
            last = AGENTS.get(body.agent_id, {})
            last_run = RUNS.get(last.get("run_id", ""))
            if last_run and last.get("node_id") in last_run["nodes"]:
                last_parent = last_run["nodes"][last["node_id"]].parent
                own = [c for c in candidates if c[0]["id"] == last_run["id"]
                       and c[1].parent and c[1].parent == last_parent]
                first = max(own, key=lambda c: c[1].depth) if own else None
            if first is None:
                first = max(candidates, key=lambda c: c[1].depth)
            # parallel ready-set: fill remaining slots from the same run, deepest first
            want = max(1, min(body.max_nodes, 8))
            pool = [c for c in candidates if c[0]["id"] == first[0]["id"] and c != first]
            pool.sort(key=lambda c: -c[1].depth)
            picked = [first] + pool[: want - 1]
        if picked:
            run, first = picked[0][0], picked[0][1]
            nodes: List[Node] = []
            for _, n in picked:
                n.state = NodeState.RUNNING
                n.attempts += 1
                n.claimed_at = time.time()
                nodes.append(n)
                store.log(run["id"], "info", "task", "core", body.agent_id,
                          f"Task: {body.agent_id} claimed {n.id}",
                          {"node": n.id, "type": "Task"})
            AGENTS[body.agent_id] = {"last_seen": time.time(), "run_id": run["id"],
                                     "node_id": first.id, "phase": "claimed"}
            store.save_run(run)
            inputs = {d: run["nodes"][d].result for d in first.deps}
            siblings = [{"id": s.id, "state": s.state.value}
                        for s in run["nodes"].values()
                        if s.parent == first.parent and s.id != first.id]
            parent = run["nodes"].get(first.parent) if first.parent else None
            context = {"parent_intent": parent.intent if parent else "",
                       "run_intent": run["intent"],
                       "siblings": [{"id": s.id, "intent": s.intent, "tool": s.tool}
                                    for s in run["nodes"].values()
                                    if s.parent == first.parent and s.id != first.id]}
    for rid in terminated:
        await _state(rid)
    if not picked:
        return {"node": None}
    await _state(run["id"])
    await _broadcast({"type": "agents", "agents": _agents_public()})
    return {"run_id": run["id"], "node": first.model_dump(mode="json"),
            "nodes": [n.model_dump(mode="json") for n in nodes],
            "inputs": inputs, "siblings": siblings, "tools": TOOLS,
            "constraints": run.get("constraints", []), "context": context,
            "max_depth": POLICY["max_depth"]}


@app.post("/runs/{run_id}/nodes/{nid}/redirect")
async def redirect(run_id: str, nid: str, body: RedirectIn) -> Dict[str, Any]:
    """Agent handoff: release the node with a reason; another agent picks it up."""
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    async with _lock:
        n = run["nodes"][nid]
        if n.state != NodeState.RUNNING:
            raise HTTPException(409, f"cannot redirect {nid} in state {n.state.value}")
        n.state = NodeState.PENDING
        n.claimed_at = 0.0
        n.hint = f"redirected by {body.agent_id}: {body.reason}"[:300]
        if body.agent_id not in n.routed_past:
            n.routed_past.append(body.agent_id)   # Alg-2: mark agent as visited
        store.save_run(run)
    await _event(run_id, "warn", "redirect", body.agent_id, "core",
                 f"Redirect: {nid} — {body.reason or 'no reason'}",
                 {"node": nid, "type": "Redirect", "reason": body.reason})
    await _state(run_id)
    return {"ok": True}


@app.post("/runs/{run_id}/nodes/{nid}/result")
async def report(run_id: str, nid: str, body: ResultIn) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    async with _lock:
        run["stats"]["llm_calls"] += body.llm_calls
        n = run["nodes"][nid]
        if n.state != NodeState.RUNNING:
            raise HTTPException(409, f"stale report: {nid} is {n.state.value}, not RUNNING")
        n.claimed_at = 0.0
        if body.state == "DONE":
            n.state, n.result, n.hint = NodeState.DONE, body.result, ""
            _settle(run, nid)
        elif n.type == NodeType.LOOP:
            if n.attempts < n.max_iterations:
                # shortcut: loop = requeue-per-iteration, not DAG unrolling; upgrade
                # when a loop body needs to be a multi-node subgraph (P-later)
                n.state, n.result = NodeState.PENDING, body.result
                n.hint = (body.hint[:200] if body.hint else
                          f"condition unmet — iteration {n.attempts}/{n.max_iterations}")
                store.log(run["id"], "info", "loop", body.agent_id or "-", "core",
                          f"{nid}: iteration {n.attempts}/{n.max_iterations}, condition unmet")
            else:
                _fail_cascade(run, nid,
                              body.hint or f"loop: condition unmet after "
                                           f"{n.attempts} iterations",
                              NodeState.ESCALATED)
        else:
            if n.attempts < POLICY["max_repairs"]:
                n.state, n.hint = NodeState.REPAIRING, body.hint
            else:
                _fail_cascade(run, nid, body.hint or f"failed after {n.attempts} attempts",
                              NodeState.ESCALATED)
        breach = _budget_breach(run)
        if breach:
            _terminate(run, breach)
            store.log(run["id"], "error", "budget", "core", "-",
                      f"run terminated: {breach}")
        store.save_run(run)
    await _event(run_id, "info" if body.state == "DONE" else "warn",
                 "result", body.agent_id or "-", "core",
                 f"Result: {nid} {body.state}" + (f" — {body.hint[:80]}" if body.hint else ""),
                 {"node": nid, "type": "Result", "state": body.state})
    await _state(run_id)
    return {"ok": True}
@app.post("/runs/{run_id}/nodes/{nid}/decompose")
async def decompose(run_id: str, nid: str, body: DecomposeIn) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    breach = _budget_breach(run)
    if breach:
        async with _lock:
            _terminate(run, breach)
            store.save_run(run)
        await _event(run_id, "error", "budget", "core", "-", f"run terminated: {breach}")
        await _state(run_id)
        raise HTTPException(429, f"budget governor: {breach}")
    children, errs = _build_children(run["nodes"], nid,
                                     [c.model_dump() for c in body.children])
    if errs:
        raise HTTPException(400, errs)
    # validate as a graph patch even when approvals are on: a bad decomposition
    # must bounce into the agent's repair loop now, not into the user's dialog later
    check = Patch(ops=[{"op": "add_node", "node": c.model_dump()} for c in children],
                  reason=f"decompose {nid}")
    _, verrs = apply_patch(run["nodes"], check, TOOLS, POLICY["max_depth"],
                           POLICY["max_nodes"], POLICY["auto_approve"])
    if verrs:
        raise HTTPException(400, verrs)
    if POLICY["approve_dag_edits"]:
        a = {"id": uuid.uuid4().hex[:8], "run_id": run_id, "node_id": nid, "kind": "patch",
             "tool": "", "args": {"children": [c.model_dump(mode="json") for c in children]},
             "status": "pending", "result": None, "created": time.time()}
        async with _lock:
            run["stats"]["approvals"] += 1
            store.save_approval(a)
        await _broadcast({"type": "approval", "approval": a})
        return {"ok": False, "status": "pending_approval", "exec_id": a["id"],
                "children": [c.id for c in children]}
    async with _lock:
        errs = _apply_decompose(run, nid, children)
        if errs:
            raise HTTPException(400, errs)
    await _event(run_id, "info", "patch", "-", "core",
                 f"Patch applied: decompose {nid} -> {[c.id for c in children]}")
    await _state(run_id)
    return {"ok": True, "children": [c.id for c in children]}


@app.post("/runs/{run_id}/patch")
async def patch_run(run_id: str, patch: Patch) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    async with _lock:
        new_nodes, errs = apply_patch(run["nodes"], patch, TOOLS, POLICY["max_depth"],
                                      POLICY["max_nodes"], POLICY["auto_approve"])
        if errs:
            raise HTTPException(400, errs)
        for n in new_nodes.values():  # user fix: re-open BLOCKED nodes whose ancestors cleared
            if n.state == NodeState.BLOCKED and not any(
                    new_nodes[a].state in (NodeState.ESCALATED, NodeState.FAILED)
                    for a in _ancestors({**run, "nodes": new_nodes}, n.id)):
                n.state, n.hint = NodeState.PENDING, ""
        run["nodes"] = new_nodes
        run["stats"]["patches"] += 1
        store.save_run(run)
    await _event(run_id, "info", "patch", "user", "core",
                 f"user patch: {patch.reason or 'manual'}")
    await _state(run_id)
    return {"ok": True}


@app.post("/runs/{run_id}/exec")
async def exec_tool(run_id: str, body: ExecIn) -> Dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or body.node_id not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    if body.tool not in TOOLS:
        raise HTTPException(400, f"unknown tool {body.tool!r}")
    needs_approval = POLICY["confirm_actions"] and body.tool not in POLICY["auto_approve"]
    a = {"id": uuid.uuid4().hex[:8], "run_id": run_id, "node_id": body.node_id,
         "kind": "exec", "tool": body.tool, "args": body.args,
         "status": "pending" if needs_approval else "done",
         "result": None, "created": time.time()}
    if not needs_approval:   # execute outside the lock: shell may run up to 60s
        a["result"] = await _exec_tool(body.tool, body.args)
    async with _lock:
        run["stats"]["execs"] += 1
        if needs_approval:
            run["stats"]["approvals"] += 1
        store.save_approval(a)
        store.log(run_id, "info", "exec", "-", "core",
                  f"exec {body.tool} on {body.node_id}"
                  + (" (auto-approved)" if not needs_approval else " (awaiting approval)"))
        store.save_run(run)
    if needs_approval:
        await _broadcast({"type": "approval", "approval": a})
    return {"exec_id": a["id"], "status": a["status"], "result": a["result"]}


@app.get("/exec/{exec_id}")
def exec_status(exec_id: str) -> Dict[str, Any]:
    a = store.approval(exec_id)
    if not a:
        raise HTTPException(404, "unknown exec")
    return {"exec_id": exec_id, "status": a["status"], "result": a["result"],
            "kind": a["kind"]}


@app.get("/approvals")
def approvals(status: str = "pending") -> List[Dict[str, Any]]:
    return store.approvals(status)


@app.post("/approvals/{aid}")
async def decide(aid: str, body: DecisionIn) -> Dict[str, Any]:
    a = store.approval(aid)
    if not a:
        raise HTTPException(404, "unknown approval")
    if a["status"] != "pending":
        raise HTTPException(409, f"already {a['status']}")
    run = RUNS.get(a["run_id"])
    approved = body.decision == "approve"
    if a["kind"] == "patch":
        edited = (body.args or {}).get("children", a["args"]["children"])
        if approved:
            if run is None:
                raise HTTPException(404, "run deleted")
            children, errs = _build_children(run["nodes"], a["node_id"], edited)
            if errs:
                raise HTTPException(400, errs)
            async with _lock:
                errs = _apply_decompose(run, a["node_id"], children)
                if errs:
                    raise HTTPException(400, errs)
                a["status"] = "done"
                a["result"] = {"applied": True, "children": [c.id for c in children]}
                store.save_approval(a)
            await _event(a["run_id"], "info", "approval", "user", "core",
                         f"patch approved for {a['node_id']} "
                         f"(edited: {'yes' if edited != a['args']['children'] else 'no'})")
            await _state(a["run_id"])
        else:  # reject -> node REPAIRING with hint; agent re-decomposes with it
            async with _lock:
                if run and a["node_id"] in run["nodes"]:
                    n = run["nodes"][a["node_id"]]
                    if n.state == NodeState.RUNNING:
                        n.state = NodeState.REPAIRING
                        n.hint = "user rejected decomposition — propose different children"
                        n.claimed_at = 0.0
                        store.save_run(run)
                a["status"] = "rejected"
                store.save_approval(a)
            await _event(a["run_id"], "warn", "approval", "user", "core",
                         f"patch rejected for {a['node_id']}")
            await _state(a["run_id"])
        await _broadcast({"type": "approval", "approval": a})
        return {"exec_id": aid, "status": a["status"], "result": a["result"]}
    # kind == exec
    if approved:
        args = body.args if body.args is not None else a["args"]
        a["args"] = args
        a["result"] = await _exec_tool(a["tool"], args)   # outside lock: shell may run 60s
        a["status"] = "done"
    else:
        a["status"] = "rejected"
    async with _lock:
        store.save_approval(a)
    await _event(a["run_id"], "info" if approved else "warn", "approval", "user", "core",
                 f"exec {a['tool']} {'approved' if approved else 'REJECTED'} on {a['node_id']}")
    await _broadcast({"type": "approval", "approval": a})
    return {"exec_id": aid, "status": a["status"], "result": a["result"]}


# ---------- metrics (P5: paper §V definitions) ----------
@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """latency (run start→last event), TSR (intents with all nodes DONE),
    FRT (failed node → its DONE report), delegation depth (max node depth),
    KG-consistency proxy (first-pass success: DONE nodes at attempts==1)."""
    per_run: Dict[str, Any] = {}
    frt_samples: List[int] = []
    done_total = done_first_pass = 0
    for run in RUNS.values():
        evs = store.events(run["id"])
        latency_ms = round(((evs[-1]["ts"] if evs else time.time()) - run["created"]) * 1000)
        nodes = list(run["nodes"].values())
        all_done = bool(nodes) and all(n.state == NodeState.DONE for n in nodes)
        # FRT: pair each FAILED result with the next DONE result on the same node
        fails: Dict[str, List[float]] = {}
        frts: List[int] = []
        for e in evs:
            p = e.get("payload") or {}
            if e["kind"] == "result" and p.get("node"):
                nid = p["node"]
                if p.get("state") == "FAILED":
                    fails.setdefault(nid, []).append(e["ts"])
                elif p.get("state") == "DONE" and fails.get(nid):
                    frts.append(round((e["ts"] - fails[nid][-1]) * 1000))
        frt_ms = round(sum(frts) / len(frts)) if frts else None
        frt_samples += frts
        d = max((n.depth for n in nodes), default=0)
        done = [n for n in nodes if n.state == NodeState.DONE]
        fp = sum(1 for n in done if n.attempts <= 1)
        done_total += len(done)
        done_first_pass += fp
        per_run[run["id"]] = {"latency_ms": latency_ms, "all_done": all_done,
                              "active": _active(run), "frt_ms": frt_ms,
                              "max_depth": d, "nodes": len(nodes),
                              "llm_calls": run["stats"]["llm_calls"]}
    finished = [v for v in per_run.values() if not v["active"]]
    tsr = (round(100 * sum(1 for v in finished if v["all_done"]) / len(finished))
           if finished else None)
    return {
        "runs": per_run,
        "total": {
            "tsr_pct": tsr,
            "avg_latency_ms": (round(sum(v["latency_ms"] for v in finished) / len(finished))
                               if finished else None),
            "avg_frt_ms": (round(sum(frt_samples) / len(frt_samples))
                           if frt_samples else None),
            "max_delegation_depth": max((v["max_depth"] for v in per_run.values()), default=0),
            "first_pass_pct": (round(100 * done_first_pass / done_total)
                               if done_total else None),
        },
    }


# ---------- workspace proxy (read-only, for the UI files panel) ----------
@app.get("/workspace/files")
async def ws_files(path: str = ".") -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(CFG["sandbox_url"] + "/list_dir", json={"path": path})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(502, f"sandbox: {type(e).__name__}")


@app.get("/workspace/file")
async def ws_file(path: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(CFG["sandbox_url"] + "/read_file", json={"path": path})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:200])
    except Exception as e:
        raise HTTPException(502, f"sandbox: {type(e).__name__}")


# ---------- websocket + UI ----------
@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    _ws.append(websocket)
    for rid in RUNS:
        await _state(rid)
    await websocket.send_text(json.dumps({"type": "agents", "agents": _agents_public()},
                                         default=str))
    await websocket.send_text(json.dumps({"type": "policy", "policy": POLICY}, default=str))
    try:
        while True:
            await websocket.receive_text()   # ignore client msgs; keep alive
    except WebSocketDisconnect:
        if websocket in _ws:
            _ws.remove(websocket)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
