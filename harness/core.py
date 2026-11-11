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
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import load
from .graph import (
    apply_patch,
    descendants,
    lint,
    named_parts,
    ready_nodes,
    resolve_args,
    validate,
)
from .models import (
    AddedNode,
    ChildIn,
    ClaimIn,
    ConfigIn,
    ConstraintIn,
    DEFAULT_MAX_DEPTH,
    DecisionIn,
    DecomposeIn,
    ExecIn,
    GuideIn,
    HeartbeatIn,
    Node,
    NodeState,
    NodeType,
    NodeUpdate,
    Patch,
    RedirectIn,
    ResultIn,
    RunIn,
    SkillIn,
)
from .prompts import DEFAULT_PARAMS, PROMPT_KEYS, resolve_params
from .store import Store

CFG = load()
POLICY: dict[str, Any] = CFG["policy"]
TOOLS: dict[str, dict[str, Any]] = {
    "shell": {"desc": "run a shell command in the shared sandbox workspace",
              "required": ["cmd"]},
    "read_file": {"desc": "read a file from the workspace", "required": ["path"],
                  "check": "content", "ground": {"path": "{path}"}},
    "write_file": {"desc": "write a file in the workspace",
                   "required": ["path", "content"], "check": "bytes"},
    "list_dir": {"desc": "list a directory in the workspace", "required": ["path"],
                 "check": "entries"},
    "web_search": {"desc": "search the web (SearXNG); returns [{title,url,snippet}]",
                   "required": ["query"], "ground": {"query": "{intent}"}},
    "http_fetch": {"desc": "fetch a URL and return readable text",
                   "required": ["url"]},
    "llm": {"desc": "virtual (agent-side): the LLM writes an answer using its deps' "
                    "results as material; returns {text}. No args.",
            "required": []},
}
AGENT_STALE_S = 15       # heartbeat older than this -> agent shown dead
DB = Path(CFG["data_dir"]) / "harness.db"
app = FastAPI(title="harness-core")
_lock = asyncio.Lock()
_ws: list[WebSocket] = []
# agent_id -> liveness/position/phase; in-memory on purpose (dead agents die with core)
AGENTS: dict[str, dict[str, Any]] = {}

# ---------- shared memory buffer: TTL result cache for idempotent read tools ----------
# web_search / http_fetch are read-only against the world: identical args inside the
# TTL window return the memoized result, skipping sandbox+network+egress entirely.
# It is shared across runs and agents — a fact one agent fetched, all agents reuse.
TOOL_CACHE_TTL_S = 900
TOOL_CACHE_MAX = 64
_CACHEABLE_TOOLS = frozenset({"web_search", "http_fetch"})
_TOOL_CACHE: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
_TOOL_CACHE_STATS = {"hits": 0, "misses": 0}


def _tool_cache_key(tool: str, args: dict[str, Any]) -> str:
    return f"{tool}\x00{json.dumps(args, sort_keys=True, default=str)}"


def _tool_cache_get(tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    hit = _TOOL_CACHE.get(_tool_cache_key(tool, args))
    if hit is None:
        _TOOL_CACHE_STATS["misses"] += 1
        return None
    ts, val = hit
    if time.time() - ts > TOOL_CACHE_TTL_S:
        del _TOOL_CACHE[_tool_cache_key(tool, args)]
        _TOOL_CACHE_STATS["misses"] += 1
        return None
    _TOOL_CACHE.move_to_end(_tool_cache_key(tool, args))
    _TOOL_CACHE_STATS["hits"] += 1
    return {**val, "cached": True}   # shallow copy: cached payloads are read-only


def _tool_cache_put(tool: str, args: dict[str, Any], val: dict[str, Any]) -> None:
    # failures (rc != 0) and stderr-bearing sandboxes never enter the cache
    if isinstance(val, dict) and val.get("rc", 0) not in (0, None):
        return
    key = _tool_cache_key(tool, args)
    _TOOL_CACHE[key] = (time.time(), val)
    _TOOL_CACHE.move_to_end(key)
    while len(_TOOL_CACHE) > TOOL_CACHE_MAX:
        _TOOL_CACHE.popitem(last=False)   # LRU eviction


def tool_arg_errors(nodes: dict[str, Node]) -> list[str]:
    """Args schema check: small local models invent arg names; this turns it into
    a validation error they can repair."""
    return [f"{n.id}: tool {n.tool!r} missing required arg {k!r}"
            for n in nodes.values()
            if n.tool and n.tool in TOOLS
            for k in TOOLS[n.tool]["required"] if k not in n.args]


store = Store(DB, event_cap=POLICY.get("event_cap", 500))
RUNS: dict[str, dict[str, Any]] = store.load_runs()


# ---------- helpers ----------
def _active(run: dict[str, Any]) -> bool:
    return any(n.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                           NodeState.WAITING) for n in run["nodes"].values())


def _ancestors(run: dict[str, Any], nid: str) -> set:
    out, stack = set(), list(run["nodes"][nid].deps)
    while stack:
        cur = stack.pop()
        if cur not in out and cur in run["nodes"]:
            out.add(cur)
            stack.extend(run["nodes"][cur].deps)
    return out


def _subtree(run: dict[str, Any], nid: str) -> list[str]:
    """nid plus every node below it via PARENT links (decomposition tree),
    which is the relation replan deletes along — dep edges ignore parents."""
    out, frontier = [], [nid]
    while frontier:
        cur = frontier.pop()
        kids = [n.id for n in run["nodes"].values() if n.parent == cur]
        out += kids
        frontier += kids
    return out


def _plan_ready(run: dict[str, Any]) -> bool:
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


async def _broadcast(msg: dict[str, Any]) -> None:
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
                      "prompts": run.get("prompts", {}),
                      "params": run.get("params", {}),
                      "plan_ready": _plan_ready(run),
                      "warnings": lint(run["nodes"]),
                      "pending": sorted(_pending_approvals(run_id))})


def _writers(run: dict[str, Any]) -> set[str]:
    """Nodes that materialize an answer from others' results: the synthesis
    sink and `$from` write-backs (deliverable writers). They must survive dead
    siblings and run once their deps are terminal — cascade-blocking them turned
    one dead leaf into an answerless head and a never-written deliverable."""
    sink = resolve_params(run.get("params"))["sink_tool"]
    return {n.id for n in run["nodes"].values()
            if n.tool == sink or "$from" in json.dumps(n.args or {})}


def _refine_writer(run: dict[str, Any], nid: str, hint: str) -> None:
    """A terminal failure whose error names a workspace file gets ONE
    error-driven refinement of the chain that AUTHORED that file: the content
    source re-synthesizes with the runtime error in hand, the write re-lands,
    and the failed node re-runs against the fixed file (dep-ordered behind the
    write). Measured live: the generated script failed at run time
    ('ModuleNotFoundError', wrong column names) while its writer sat DONE and
    every runner burned its retries against the broken artifact — the error
    never reached the author. Bounded: one refine per content source, one
    retry per node.
    shortcut: markers live on the run dict (lost on core restart mid-run), and
    literal-content writes are not refined — only `$from` chains; upgrade when
    a live run shows one needing feedback."""
    n = run["nodes"][nid]
    if _budget_breach(run):
        return                       # an out-of-budget run must end, not loop
    # only the EXECUTION ERROR may attribute a failure to a file — the predicate
    # echo names every task path (live T2: 'identify failing tests in
    # bugfix-lab/analyze.py' rewrote a working fix into test code)
    err = hint.split(" | rc=", 1)[1] if " | rc=" in (hint or "") else ""
    if not err:
        return
    chain = [w for w in run["nodes"].values()
             if w.tool == "write_file" and w.id != nid
             and isinstance((w.args or {}).get("content"), dict)
             and "$from" in w.args["content"]
             and str(w.args.get("path") or "")
             and re.search(r"(?<![\w.-])" + re.escape(str(w.args["path"])), err)]
    if not chain:
        return
    w = max(chain, key=lambda x: len(str(x.args.get("path") or "")))
    s = run["nodes"].get(w.args["content"]["$from"])
    if s is None:
        return
    refined = run.setdefault("refined", [])
    retried = run.setdefault("retried", [])
    if s.id not in refined:
        refined.append(s.id)
        s.state, s.attempts = NodeState.PENDING, 0
        s.hint = (f"the file produced from your previous output failed at run "
                  f"time: {hint}")[:300]
        w.state, w.attempts, w.hint = NodeState.PENDING, 0, ""
    if nid in retried:
        return
    retried.append(nid)
    if w.state == NodeState.DONE:
        n.state, n.attempts, n.hint = NodeState.PENDING, 0, ""
    else:
        # re-run AFTER the rewrite, deferred until the write reports DONE. No
        # dep edge: the runner may already FEED the chain (its output is the
        # sink's material) — an edge back closed a cycle and deadlocked the run
        run.setdefault("defer_retry", {}).setdefault(w.id, []).append(nid)


def _fail_cascade(run: dict[str, Any], nid: str, hint: str, state: NodeState) -> None:
    """Terminal failure of ONE node: it -> state, its dep-consumers -> BLOCKED.
    Parents are NOT escalated (a leaf going purple must not sink the whole run);
    the parent just carries a pointer hint, so fix & retry on the failing leaf
    reopens the branch — patch_run lifts BLOCKED nodes whose ancestors cleared."""
    node = run["nodes"][nid]
    node.state, node.hint = state, hint
    writers = _writers(run)
    for d in descendants(run["nodes"], nid):
        dn = run["nodes"][d]
        if dn.id in writers:
            continue   # writers answer from the material that SURVIVED — blocking
                       # them turned one dead leaf into an answerless run
        if dn.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                        NodeState.WAITING):
            dn.state, dn.hint = NodeState.BLOCKED, f"upstream {nid} {state.value.lower()}"
    if node.parent and node.parent in run["nodes"]:
        p = run["nodes"][node.parent]
        if p.state != NodeState.DONE:
            p.hint = f"child {nid} {state.value.lower()}: {hint}"[:300]
    _refine_writer(run, nid, hint)   # before collapse: a refined node is alive,
    _collapse_dead_branch(run, nid)  # its parents must keep waiting for it


def _collapse_dead_branch(run: dict[str, Any], nid: str) -> None:
    """Escalation is terminal: a WAITING parent whose children ALL went terminal
    can never settle (settle needs all-DONE) — escalate it too, bubbling to the
    root, so a branch that ran out of repairs fails the run honestly instead of
    leaving unclaimable WAITING nodes: those hold `_active` true forever (the
    decomposition benchmark had runs spin until the driver gave up on them)."""
    pid = run["nodes"][nid].parent
    terminal = (NodeState.DONE, NodeState.FAILED, NodeState.BLOCKED, NodeState.ESCALATED)
    writers = _writers(run)
    while pid and pid in run["nodes"]:
        p = run["nodes"][pid]
        kids = [n for n in run["nodes"].values() if n.parent == pid]
        if p.state != NodeState.WAITING or not kids \
                or all(k.state == NodeState.DONE for k in kids) \
                or not all(k.state in terminal for k in kids):
            return
        writer = max((k for k in kids
                      if k.state == NodeState.DONE
                      and isinstance(k.result, dict) and k.result.get("text")),
                     key=lambda k: len(k.deps), default=None)
        if writer:
            # partial settle: a sibling died but the writer answered from what
            # survived — the head must hold that text, not go up empty
            p.state = NodeState.DONE
            p.result = {"text": writer.result["text"], "from": writer.id}
            _settle(run, p.id)
            return
        failed = ", ".join(k.id for k in kids if k.state != NodeState.DONE)
        p.state = NodeState.ESCALATED
        p.hint = f"children failed: {failed}"[:300]
        for d in descendants(run["nodes"], p.id):   # dep-consumers of a collapsed
            dn = run["nodes"][d]                    # parent can never run — block
            if dn.id in writers:
                continue                            # writers still answer
            if dn.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                            NodeState.WAITING):     # them too, or they sit claimable
                dn.state, dn.hint = NodeState.BLOCKED, f"upstream {p.id} escalated"[:300]
        pid = p.parent


def _settle(run: dict[str, Any], nid: str) -> None:
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
        elif kids and all(k.state in (NodeState.DONE, NodeState.FAILED,
                                      NodeState.BLOCKED, NodeState.ESCALATED)
                          for k in kids):
            # the branch can't settle (someone terminal-failed) and its LAST kid
            # just finished: _collapse_dead_branch only ran at the failure itself,
            # while siblings were still live — re-check here or the parent stays
            # WAITING forever and holds `active` true
            _collapse_dead_branch(run, nid)


async def _exec_tool(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Core mediates every tool call to the sandbox. Static tools have fixed
    endpoints; anything else is a dynamic tool from /workspace/tools (P2)."""
    if tool == "llm":   # virtual, agent-side only — never sandboxed
        return {"rc": -7, "stderr": "llm is agent-side; not executable via the sandbox"}
    if tool in _CACHEABLE_TOOLS:
        cached = _tool_cache_get(tool, args)
        if cached is not None:
            return cached
    path = {"shell": "/shell", "read_file": "/read_file", "write_file": "/write_file",
            "list_dir": "/list_dir", "web_search": "/web_search",
            "http_fetch": "/http_fetch"}.get(tool, "/run_tool")
    payload = args if path != "/run_tool" else {"name": tool, "args": args}
    try:
        async with httpx.AsyncClient(timeout=70) as c:
            r = await c.post(CFG["sandbox_url"] + path, json=payload)
            r.raise_for_status()
            out = r.json()
            if tool in _CACHEABLE_TOOLS:
                _tool_cache_put(tool, args, out)
            return out
    except httpx.HTTPStatusError as e:
        return {"rc": -3, "stdout": "",
                "stderr": f"sandbox {e.response.status_code}: {e.response.text[:300]}"}
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return {"rc": -4, "stdout": "", "stderr": f"sandbox unreachable: {type(e).__name__}"}


def _budget_breach(run: dict[str, Any]) -> str:
    """Budget governor: returns a reason when the run exceeded its budgets."""
    if run["stats"]["llm_calls"] >= POLICY["max_llm_calls"]:
        return f"max_llm_calls={POLICY['max_llm_calls']} reached"
    if time.time() - run["created"] >= POLICY["max_seconds"]:
        return f"max_seconds={POLICY['max_seconds']} reached"
    return ""


def _terminate(run: dict[str, Any], reason: str) -> None:
    """Governor stop: every live node ESCALATED -> run leaves the claimable set."""
    for n in run["nodes"].values():
        if n.state in (NodeState.PENDING, NodeState.RUNNING, NodeState.REPAIRING,
                       NodeState.WAITING):
            n.state, n.hint = NodeState.ESCALATED, f"budget governor: {reason}"
            n.claimed_at = 0.0


def _requeue_stale(run: dict[str, Any]) -> None:
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


def _known_tools(run: dict[str, Any]) -> dict[str, Any]:
    """TOOLS plus the run's custom synthesis sink name (per-run virtual writer)."""
    sink = resolve_params(run.get("params"))["sink_tool"]
    return TOOLS if sink in TOOLS else {**TOOLS, sink: TOOLS["llm"]}


def _build_children(nodes: dict[str, Node], nid: str,
                    raw: list[dict[str, Any]]) -> tuple[list[Node], list[str]]:
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
    errors: list[str] = []
    children: list[Node] = []
    for i, ch in enumerate(parsed):
        resolved: list[str] = []
        for j in ch.deps:
            if isinstance(j, int):
                if not 0 <= j < i:
                    errors.append(f"child {i}: dep index {j} must point to an earlier sibling")
                else:
                    resolved.append(f"{nid}.{j + 1}")
            elif isinstance(j, str) and (any(c.id == j for c in children) or j in nodes):
                resolved.append(j)          # batch sibling or an existing run node
        # (cross-branch edge: e.g. a conclusion section consumes another branch —
        # the target already exists, so no cycle can route through the new child)
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


def _idx(nid: str) -> int:
    """Creation order within a parent: root.2 < root.10 (numeric tail, not string)."""
    try:
        return int(nid.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return 0


def _apply_decompose(run: dict[str, Any], nid: str, children: list[Node]) -> list[str]:
    """Validate as a patch and apply; parent -> WAITING. Caller holds the lock."""
    parent = run["nodes"].get(nid)
    if parent is not None and parent.parent in run["nodes"]:
        # document-shaped runs (the intent names its pieces) wire their sections:
        # a branch's writer consumes the EARLIER sections' summaries, so it writes
        # with the previous part in hand and sections cannot race each other —
        # section branches sit at any depth (part -> chapter -> section)
        parts = named_parts(run.get("intent") or "")
        if parts:
            sink_name = resolve_params(run.get("params"))["sink_tool"]
            earlier = [s for s in run["nodes"].values()
                       if s.parent == parent.parent and s.id != nid
                       and _idx(s.id) < _idx(nid)]
            for c in children:
                if c.tool == sink_name:
                    c.deps += [s.id for s in earlier if s.id not in c.deps]
    # Tool-gated abstract children KEEP their tool deps: during planning the claim
    # gate holds them (they refine with real results after /go), and _plan_ready
    # exempts them from the checkpoint. Stripping them here would delete the
    # information edge from the stored graph — the audit trail must not lie.
    patch = Patch(ops=[{"op": "add_node", "node": c.model_dump()} for c in children],
                  reason=f"decompose {nid}")
    new_nodes, errs = apply_patch(run["nodes"], patch, _known_tools(run),
                                  POLICY["max_depth"], POLICY["max_nodes"],
                                  POLICY["auto_approve"])
    if errs:
        return errs
    new_nodes[nid].state = NodeState.WAITING
    run["nodes"] = new_nodes
    run["stats"]["patches"] += 1
    store.save_run(run)
    return []


def _agents_public() -> dict[str, Any]:
    now = time.time()
    return {aid: {**a, "age": round(now - a["last_seen"], 1),
                  "alive": now - a["last_seen"] < AGENT_STALE_S}
            for aid, a in AGENTS.items()}
# ---------- endpoints: infra ----------
@app.get("/health")
async def health() -> dict[str, Any]:
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            ollama_ok = (await c.get(CFG["ollama_url"] + "/api/tags")).status_code == 200
    except Exception:
        pass
    return {"ok": True, "ollama": ollama_ok, "runs": len(RUNS), "agents": len(AGENTS),
            "tool_cache": {"entries": len(_TOOL_CACHE), **_TOOL_CACHE_STATS}}


@app.get("/tools")
def tools() -> dict[str, Any]:
    return {"tools": TOOLS, "auto_approve": POLICY["auto_approve"]}


@app.post("/tools/sync")
async def tools_sync() -> dict[str, Any]:
    """Pull dynamic tools (P2) from the sandbox into the registry. Repeatable:
    re-sync after dropping or editing a .py in /workspace/tools."""
    return await _sync_tools()


async def _sandbox_get(path: str) -> dict[str, Any]:
    """Sandbox HTTP seam (tests stub this): GET and return JSON."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(CFG["sandbox_url"] + path)
        r.raise_for_status()
        return r.json()


async def _sandbox_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Sandbox HTTP seam (tests stub this): POST and return JSON or HTTP 4xx."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(CFG["sandbox_url"] + path, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code,
                                f"sandbox: {r.json().get('detail', r.text)[:200]}")
        return r.json()


async def _sync_tools() -> dict[str, Any]:
    """Pull dynamic tools (P2) from the sandbox into TOOLS; hot-reload removes
    vanished tools too. Shared by /tools/sync and /skills."""
    try:
        found = (await _sandbox_get("/discover_tools")).get("tools", {})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"sandbox: {type(e).__name__}")
    for name in [k for k, v in TOOLS.items() if v.get("dynamic") and k not in found]:
        del TOOLS[name]                # hot-reload removes vanished tools too
    for name, meta in found.items():
        ent = {"desc": meta.get("desc", f"dynamic tool ({meta.get('file')})"),
               "required": meta.get("required", []), "dynamic": True,
               "unvetted": name not in TOOLS}   # new since startup: first use shows a card
        for extra in ("check", "ground"):
            if meta.get(extra):
                ent[extra] = meta[extra]
        TOOLS[name] = ent
    return {"synced": sorted(found), "tools": TOOLS}


_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


@app.post("/skills")
async def skills_create(body: SkillIn) -> dict[str, Any]:
    """The AI (or the UI) registers a NEW tool at runtime: a declarative spec
    (name, desc, required args, shell template) becomes a tool the planner can
    pick from the very next claim. A new tool's first use shows the 'unvetted'
    approval card unless it is auto-approved, keeping the human gate."""
    if not _SKILL_NAME.match(body.name):
        raise HTTPException(400, f"bad skill name {body.name!r} (lowercase, digits, _)")
    if body.name in TOOLS and not TOOLS[body.name].get("dynamic"):
        raise HTTPException(409, f"{body.name!r} is a built-in tool")
    await _sandbox_post("/create_skill", body.model_dump())
    await _sync_tools()
    if body.name not in TOOLS:
        raise HTTPException(502, "sandbox registered the file but discovery lost it")
    store.log("-", "info", "skill", "user", "core",
              f"skill registered: {body.name} ({body.desc[:120]})")
    return {"created": body.name, "tool": TOOLS[body.name]}


@app.get("/policy")
def get_policy() -> dict[str, Any]:
    return POLICY


@app.get("/autosize")
async def autosize() -> dict[str, Any]:
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
async def set_policy(body: dict[str, Any]) -> dict[str, Any]:
    """Runtime knobs. Numeric budgets land via the UI settings panel, so they are
    validated + clamped here: a typo in the UI must not misbehave the governor."""
    editable_bool = {"confirm_actions", "approve_dag_edits"}
    editable_list = {"auto_approve"}
    bounds = {"max_agents": (1, 32), "max_nodes": (8, 2000),
              "max_depth": (2, DEFAULT_MAX_DEPTH),  # config's depth is legal to set back
              "max_repairs": (1, 10), "max_llm_calls": (10, 100000),
              "max_seconds": (60, 86400), "claim_lease_seconds": (10, 3600)}
    unknown = set(body) - editable_bool - editable_list - set(bounds)
    if unknown:
        raise HTTPException(400, f"not editable at runtime: {sorted(unknown)}")
    clean: dict[str, Any] = {}
    for k, v in body.items():
        if k in bounds:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise HTTPException(400, f"{k} must be a number")
            lo, hi = bounds[k]
            clean[k] = int(min(hi, max(lo, v)))
        elif k in editable_bool:
            if not isinstance(v, bool):
                raise HTTPException(400, f"{k} must be a boolean")
            clean[k] = v
        else:   # auto_approve
            if not isinstance(v, list) or not all(isinstance(t, str) for t in v):
                raise HTTPException(400, "auto_approve must be a list of tool names")
            clean[k] = [t for t in v if t in TOOLS]
    POLICY.update(clean)
    await _broadcast({"type": "policy", "policy": POLICY})
    return POLICY


@app.get("/tools/cache")
def tool_cache_stats() -> dict[str, Any]:
    """Shared-memory buffer observability: what's memoized and how much work it saved."""
    now = time.time()
    return {"entries": [{"tool": k.split("\x00", 1)[0],
                         "args": json.loads(k.split("\x00", 1)[1]),
                         "age_s": round(now - ts, 1)}
                        for k, (ts, _) in _TOOL_CACHE.items()],
            "ttl_s": TOOL_CACHE_TTL_S, "max_entries": TOOL_CACHE_MAX,
            **_TOOL_CACHE_STATS}


@app.delete("/tools/cache")
def tool_cache_clear() -> dict[str, Any]:
    n = len(_TOOL_CACHE)
    _TOOL_CACHE.clear()
    return {"cleared": n}


@app.post("/heartbeat")
async def heartbeat(hb: HeartbeatIn) -> dict[str, Any]:
    AGENTS[hb.agent_id] = {"last_seen": time.time(), "run_id": hb.run_id,
                           "node_id": hb.node_id, "phase": hb.phase}
    await _broadcast({"type": "agents", "agents": _agents_public()})
    return {"ok": True}


@app.get("/agents")
def list_agents() -> dict[str, Any]:
    return _agents_public()


# ---------- endpoints: runs ----------
@app.post("/runs")
async def create_run(body: RunIn) -> dict[str, Any]:
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
           "prompts": {}, "params": {},
           "stats": {"llm_calls": 0, "execs": 0, "approvals": 0, "patches": 0}}
    async with _lock:
        RUNS[run["id"]] = run
        store.save_run(run)
    store.log(run["id"], "info", "run", "user", "core", f"run created: {run['intent'][:80]}")
    await _broadcast({"type": "run", "run_id": run["id"], "intent": run["intent"]})
    await _state(run["id"])
    return {"run_id": run["id"]}


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    return [{"id": r["id"], "intent": r["intent"], "created": r["created"],
             "active": _active(r), "phase": r.get("phase", "executing"),
             "constraints": r.get("constraints", []), "plan_ready": _plan_ready(r),
             "prompts": r.get("prompts", {}), "params": r.get("params", {}),
             "states": {s.value: sum(1 for n in r["nodes"].values() if n.state == s)
                        for s in NodeState}} for r in sorted(RUNS.values(), key=lambda x: -x["created"])]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    return {"id": run["id"], "intent": run["intent"], "active": _active(run),
            "phase": run.get("phase", "executing"),
            "constraints": run.get("constraints", []), "plan_ready": _plan_ready(run),
            "prompts": run.get("prompts", {}), "params": run.get("params", {}),
            "nodes": {k: v.model_dump(mode="json") for k, v in run["nodes"].items()},
            "stats": run["stats"], "policy": POLICY, "warnings": lint(run["nodes"]),
            "pending": sorted(_pending_approvals(run_id))}


@app.get("/runs/{run_id}/snapshot")
def snapshot(run_id: str) -> dict[str, Any]:
    """Save-as-template: the run's DAG reset to a fresh POST /runs body — reuse a
    plan that worked (drug checklist, onboarding flow, paper skeleton) as a
    template DAG without re-decomposing."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    nodes = []
    for n in run["nodes"].values():
        c = n.model_copy(deep=True)
        c.state, c.result, c.attempts, c.hint = NodeState.PENDING, None, 0, ""
        c.claimed_at, c.routed_past, c.llm_calls = 0.0, [], 0
        nodes.append(c.model_dump(mode="json"))
    return {"intent": run["intent"], "nodes": nodes}


@app.get("/runs/{run_id}/knowledge")
def knowledge(run_id: str) -> dict[str, Any]:
    """The run's accumulated knowledge tree: every node holding text, nested by
    branch — leaves are raw evidence, inner nodes are the summaries that rose.
    This is what patch planning and the planner's 'Known material' context read."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    nodes = run["nodes"]
    kids: dict[str, list[Node]] = {}
    for n in nodes.values():
        if n.parent:
            kids.setdefault(n.parent, []).append(n)

    def build(nid: str) -> dict[str, Any] | None:
        ch = [b for c in sorted(kids.get(nid, []), key=lambda x: x.id)
              for b in [build(c.id)] if b is not None]
        n = nodes[nid]
        text = n.result.get("text", "") if isinstance(n.result, dict) else ""
        if not text and not ch:
            return None
        return {"id": nid, "intent": n.intent, "chars": len(text),
                "text": text[:400], "children": ch}

    return {"knowledge": [b for nid in nodes if not nodes[nid].parent
                          for b in [build(nid)] if b is not None]}


@app.post("/runs/{run_id}/guide")
async def set_guide(run_id: str, body: GuideIn) -> dict[str, Any]:
    """An agent's auto-generated per-task guide lands as a 'guide:' constraint —
    visible and editable in the plan bar like any user constraint. Idempotent:
    parallel agents racing the root split write it once."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    text = " ".join(body.text.split())[:400]
    if not text:
        raise HTTPException(400, "empty guide")
    added = False
    async with _lock:
        cons = run.setdefault("constraints", [])
        if not any(str(c).startswith("guide:") for c in cons):
            cons.append(f"guide: {text}")
            added = True
            store.save_run(run)
    if added:
        await _event(run_id, "info", "plan", "agent", "core",
                     f"task guide generated: {text[:70]}")
        await _state(run_id)
    return {"ok": True, "constraints": run["constraints"]}


@app.put("/runs/{run_id}/config")
async def set_config(run_id: str, body: ConfigIn) -> dict[str, Any]:
    """Per-run prompt + parameter overrides (the 'adapt it to your domain' knob).
    Blank/omitted keys keep the built-in defaults; the whole maps are replaced.
    Validated/clamped through the same resolver the agents use (prompts.py)."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    if run.get("phase") != "planning":
        raise HTTPException(409, "config edits only at the planning checkpoint")
    unknown_p = set(body.params) - set(DEFAULT_PARAMS)
    if unknown_p:
        raise HTTPException(400, f"unknown params: {sorted(unknown_p)}")
    resolved = resolve_params(body.params)
    params = {k: resolved[k] for k in body.params}
    prompts = {k: v[:4000] for k, v in body.prompts.items()
               if k in PROMPT_KEYS and isinstance(v, str) and v.strip()}
    run["prompts"], run["params"] = prompts, params
    store.save_run(run)
    store.log(run_id, "info", "plan", "user", "core",
              f"config set: prompts={sorted(prompts) or 'defaults'}, params={params or 'defaults'}")
    await _state(run_id)
    return {"ok": True, "prompts": prompts, "params": params}


@app.post("/runs/{run_id}/nodes")
async def add_node(run_id: str, body: AddedNode) -> dict[str, Any]:
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
        sink_name = resolve_params(run.get("params"))["sink_tool"]
        sink = max((n for n in sibs if n.tool == sink_name and n.deps),
                   key=lambda n: len(n.deps), default=None)
        if sink and nid not in sink.deps:
            sink.deps = [n.id for n in sibs if n.id != sink.id]
        store.save_run(run)
    await _event(run_id, "info", "plan", "user", "core",
                 f"branch added by user: {nid} — {intent[:60]}")
    await _state(run_id)
    return {"ok": True, "id": nid}


@app.patch("/runs/{run_id}/nodes/{nid}")
async def update_node(run_id: str, nid: str, body: NodeUpdate) -> dict[str, Any]:
    """User rewrites a node's instruction while the plan is still open."""
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    if run.get("phase") != "planning":
        raise HTTPException(409, "tree edits only at the planning checkpoint")
    n = run["nodes"][nid]
    if n.state not in (NodeState.PENDING, NodeState.WAITING):
        raise HTTPException(409, f"cannot edit {nid} in state {n.state.value}")
    if not body.intent.strip() and body.args is None:
        raise HTTPException(400, "nothing to update (provide intent and/or args)")
    if body.args is not None:
        if not isinstance(body.args, dict):
            raise HTTPException(400, "args must be an object")
        if n.tool and n.tool in TOOLS:
            missing = [k for k in TOOLS[n.tool]["required"] if k not in body.args]
            if missing:
                raise HTTPException(400, f"{n.tool} still needs args: {missing}")
        n.args = body.args
    if body.intent.strip():
        intent = " ".join(body.intent.split())[:300]
        n.intent = intent
    store.save_run(run)
    await _event(run_id, "info", "plan", "user", "core",
                 f"node {nid} edited by user")
    await _state(run_id)
    return {"ok": True}


@app.put("/runs/{run_id}/constraints")
async def set_constraints(run_id: str, body: ConstraintIn) -> dict[str, Any]:
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
async def go(run_id: str) -> dict[str, Any]:
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
async def replan(run_id: str, nid: str) -> dict[str, Any]:
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
def get_events(run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    return store.events(run_id, after_id)


@app.delete("/runs/{run_id}")
async def delete_run(run_id: str, force: bool = False) -> dict[str, Any]:
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
async def claim(body: ClaimIn) -> dict[str, Any]:
    terminated: list[str] = []
    async with _lock:
        candidates = []  # (run, node) ready across active runs
        for run in sorted(RUNS.values(), key=lambda x: x["created"]):
            if not _active(run):
                continue
            if not any(n.state in (NodeState.PENDING, NodeState.REPAIRING, NodeState.RUNNING)
                       for n in run["nodes"].values()):
                continue   # settled run (no work pending, running, or requeuable): skip scans
            _requeue_stale(run)
            breach = _budget_breach(run)
            if breach:
                _terminate(run, breach)
                store.save_run(run)
                store.log(run["id"], "error", "budget", "core", "-",
                          f"run terminated: {breach}")
                terminated.append(run["id"])
                continue
            cands = ready_nodes(run["nodes"], resilient=_writers(run))
            if run.get("phase") == "planning":
                # plan-first: only abstract nodes are claimable, and planning runs on
                # structure, not results — deps WITHOUT a tool are free context, so the
                # DONE-gate applies only to tool deps. (Gating abstracts on abstract
                # deps deadlocks a deps-carrying abstract node: its deps settle after
                # /go, but /go waits for plan_ready — the 9ea37d2e root.4 corner.)
                cands = [n for n in run["nodes"].values()
                         if not n.tool
                         and n.state in (NodeState.PENDING, NodeState.REPAIRING)
                         and all(run["nodes"][d].state == NodeState.DONE
                                 for d in n.deps
                                 if d in run["nodes"] and run["nodes"][d].tool)]
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
            picked: list[tuple[dict[str, Any], Node]] = []
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
            picked = [first]   # one node per claim: a claimed-but-unhandled node would
        # sit RUNNING until its lease expired; parallelism is --scale agent=N
        if picked:
            run, first = picked[0][0], picked[0][1]
            first.state = NodeState.RUNNING
            first.attempts += 1
            first.claimed_at = time.time()
            store.log(run["id"], "info", "task", "core", body.agent_id,
                      f"Task: {body.agent_id} claimed {first.id}",
                      {"node": first.id, "type": "Task"})
            AGENTS[body.agent_id] = {"last_seen": time.time(), "run_id": run["id"],
                                     "node_id": first.id, "phase": "claimed"}
            store.save_run(run)
            nodes: list[Node] = [first]
            inputs = {d: run["nodes"][d].result for d in first.deps}
            siblings = [{"id": s.id, "state": s.state.value}
                        for s in run["nodes"].values()
                        if s.parent == first.parent and s.id != first.id]
            parent = run["nodes"].get(first.parent) if first.parent else None
            known = [{"id": n.id, "intent": n.intent[:60],
                      "chars": len(str(n.result.get("text", "")))}
                     for n in sorted(run["nodes"].values(), key=lambda x: x.id)
                     if isinstance(n.result, dict) and n.result.get("text")][:20]
            context = {"parent_intent": parent.intent if parent else "",
                       "run_intent": run["intent"],
                       "known": known,
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
            "prompts": run.get("prompts", {}), "params": run.get("params", {}),
            "max_depth": POLICY["max_depth"]}


@app.post("/runs/{run_id}/nodes/{nid}/redirect")
async def redirect(run_id: str, nid: str, body: RedirectIn) -> dict[str, Any]:
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
async def report(run_id: str, nid: str, body: ResultIn) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or nid not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    async with _lock:
        n = run["nodes"][nid]
        if n.state != NodeState.RUNNING:
            # checked BEFORE the spend lands: a stale report must not eat the
            # run's llm_calls budget (it 409s — nothing was credited)
            raise HTTPException(409, f"stale report: {nid} is {n.state.value}, not RUNNING")
        run["stats"]["llm_calls"] += body.llm_calls
        n.llm_calls += body.llm_calls
        n.claimed_at = 0.0
        if body.state == "DONE":
            n.state, n.result, n.hint = NodeState.DONE, body.result, ""
            _settle(run, nid)
            for d in run.get("defer_retry", {}).pop(nid, []):
                dn = run["nodes"].get(d)
                if dn is not None and dn.state != NodeState.PENDING:
                    # error-driven refinement: the artifact is fixed — the node
                    # that failed against it gets its re-run now
                    dn.state, dn.attempts, dn.hint = NodeState.PENDING, 0, ""
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
async def decompose(run_id: str, nid: str, body: DecomposeIn) -> dict[str, Any]:
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
    _, verrs = apply_patch(run["nodes"], check, _known_tools(run), POLICY["max_depth"],
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
async def patch_run(run_id: str, patch: Patch) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "unknown run")
    async with _lock:
        new_nodes, errs = apply_patch(run["nodes"], patch, TOOLS, POLICY["max_depth"],
                                      POLICY["max_nodes"], POLICY["auto_approve"])
        errs += tool_arg_errors(new_nodes) if not errs else []
        if errs:
            raise HTTPException(400, errs)
        for n in new_nodes.values():  # user fix: re-open BLOCKED nodes whose ancestors cleared
            if n.state == NodeState.BLOCKED and not any(
                    new_nodes[a].state in (NodeState.ESCALATED, NodeState.FAILED)
                    for a in _ancestors({**run, "nodes": new_nodes}, n.id)):
                n.state, n.hint = NodeState.PENDING, ""
        run["nodes"] = new_nodes
        if time.time() - run["created"] >= POLICY["max_seconds"]:
            # user patch on an out-of-budget (already budget-terminated) run:
            # re-arm the wall clock — max_seconds counts from `created`, so the
            # next claim tick re-kills a resumed run before it moves (9ea37d2e:
            # "user patch" → terminated again 1s later, twice). A human directly
            # fixing the run IS the supervision the budget stands in for.
            run["created"] = time.time()
            store.log(run_id, "info", "patch", "user", "core",
                      "budget re-armed after user patch")
        run["stats"]["patches"] += 1
        store.save_run(run)
    await _event(run_id, "info", "patch", "user", "core",
                 f"user patch: {patch.reason or 'manual'}")
    await _state(run_id)
    return {"ok": True}


@app.post("/runs/{run_id}/exec")
async def exec_tool(run_id: str, body: ExecIn) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run or body.node_id not in run["nodes"]:
        raise HTTPException(404, "unknown run or node")
    if body.tool not in TOOLS:
        raise HTTPException(400, f"unknown tool {body.tool!r}")
    node = run["nodes"][body.node_id]
    # $from bindings resolve against dep results HERE (server-side), so what the
    # sandbox runs — and what the approval card shows — is the final data
    args, err = resolve_args(run["nodes"], Node(**{**node.model_dump(), "args": body.args}))
    if err:
        raise HTTPException(400, err)
    # unvetted = dropped+synced after startup: a human must see it once, even if
    # auto_approve names it (a planted file must not widen the approved surface silently)
    needs_approval = POLICY["confirm_actions"] and (
        body.tool not in POLICY["auto_approve"] or bool(TOOLS[body.tool].get("unvetted")))
    a = {"id": uuid.uuid4().hex[:8], "run_id": run_id, "node_id": body.node_id,
         "kind": "exec", "tool": body.tool, "args": args, "attempts": node.attempts,
         "status": "pending" if needs_approval else "done",
         "result": None, "created": time.time()}
    if not needs_approval:   # execute outside the lock: shell may run up to 60s
        a["result"] = await _exec_tool(body.tool, args)
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
def exec_status(exec_id: str) -> dict[str, Any]:
    a = store.approval(exec_id)
    if not a:
        raise HTTPException(404, "unknown exec")
    return {"exec_id": exec_id, "status": a["status"], "result": a["result"],
            "kind": a["kind"]}


@app.get("/approvals")
def approvals(status: str = "pending") -> list[dict[str, Any]]:
    return store.approvals(status)


@app.post("/approvals/{aid}")
async def decide(aid: str, body: DecisionIn) -> dict[str, Any]:
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
        # staleness gate: the agent gives up on an unanswered card (APPROVAL_TIMEOUT),
        # reports, and the node can be re-claimed and re-run. A late approval must
        # NOT replay the tool (mutation replay) — the attempts epoch pins the card
        # to the claim that parked it, and terminal states block it outright.
        node = run["nodes"].get(a["node_id"]) if run else None
        if (node is None or a.get("attempts", node.attempts) != node.attempts
                or node.state in (NodeState.DONE, NodeState.BLOCKED, NodeState.ESCALATED)):
            a["status"] = "stale"
            async with _lock:
                store.save_approval(a)
            await _event(a["run_id"], "warn", "approval", "user", "core",
                         f"stale exec card for {a['node_id']} ignored (node moved on)")
            await _broadcast({"type": "approval", "approval": a})
            return {"exec_id": aid, "status": "stale", "result": None}
        if a["tool"] in TOOLS:   # first vetted use: a human just approved it by name
            TOOLS[a["tool"]].pop("unvetted", None)
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
def metrics() -> dict[str, Any]:
    """latency (run start→last event), TSR (intents with all nodes DONE),
    FRT (failed node → its DONE report), delegation depth (max node depth),
    KG-consistency proxy (first-pass success: DONE nodes at attempts==1)."""
    per_run: dict[str, Any] = {}
    frt_samples: list[int] = []
    done_total = done_first_pass = 0
    for run in RUNS.values():
        evs = store.events(run["id"])
        latency_ms = round(((evs[-1]["ts"] if evs else time.time()) - run["created"]) * 1000)
        nodes = list(run["nodes"].values())
        all_done = bool(nodes) and all(n.state == NodeState.DONE for n in nodes)
        # FRT: pair each FAILED result with the next DONE result on the same node
        fails: dict[str, list[float]] = {}
        frts: list[int] = []
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
async def ws_files(path: str = ".") -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(CFG["sandbox_url"] + "/list_dir", json={"path": path})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(502, f"sandbox: {type(e).__name__}")


@app.get("/workspace/file")
async def ws_file(path: str) -> dict[str, Any]:
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
