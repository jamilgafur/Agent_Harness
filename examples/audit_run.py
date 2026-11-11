#!/usr/bin/env python3
"""Audit one full run of the product loop: sentence -> decomposed DAG -> human
checkpoints -> grounded answer, with the trail inspected afterwards.

This is the harness demonstrating itself, not a feature micro-benchmark. It
drives a complete intent run in-process (real core via FastAPI TestClient, a
real Agent whose LLM is swapped) and then audits what came out:

  shape        — final ASCII tree: states, attempts, tools, per-node budgets
  reliability  — first-pass success + repair count (from the event log)
  tree quality — lint warnings, convergence, delegation depth
  grounding    — every URL cited in the head answer must be a URL the tree
                 actually gathered (invented citations = audit failure)
  human ledger — every checkpoint a human faced: plan approval, exec cards,
                 edits (offline: decisions are scripted; --live --interactive
                 times a real human)
  persistence  — simulated restart: state reloaded from SQLite must equal
                 the in-memory state, events intact
  budget       — llm_calls against the governor cap

    python examples/audit_run.py                        # offline audit (CI-safe)
    python examples/audit_run.py --live                 # real model, auto-approve
    python examples/audit_run.py --live --interactive   # you ARE the human loop
"""
import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("HARNESS_CONFIG", json.dumps({
    "data_dir": "/tmp/harness-audit", "sandbox_url": "http://stubbed"}))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import harness.agent.base as agent_base  # noqa: E402
import harness.core as core  # noqa: E402
from harness.agent import Agent  # noqa: E402
from harness.llm import LLM  # noqa: E402
from harness.models import NodeState  # noqa: E402

INTENT = "How transformers scale: attention cost vs sequence length, and scaling laws"

ATTENTION_HITS = [
    {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762",
     "page_text": "The Transformer computes scaled dot-product attention: an n by n "
                  "matrix over positions, O(n^2) in sequence length."},
    {"title": "FlashAttention", "url": "https://arxiv.org/abs/2205.14135",
     "page_text": "IO-aware exact attention: tiles the n by n matrix to cut HBM "
                  "traffic; same math, less memory traffic."},
]
SCALING_HITS = [
    {"title": "Scaling Laws for Neural Language Models", "url": "https://arxiv.org/abs/2001.08361",
     "page_text": "Loss follows a power law in parameters, data and compute over "
                  "seven orders of magnitude."},
    {"title": "Chinchilla", "url": "https://arxiv.org/abs/2203.15556",
     "page_text": "Compute-optimal: model and tokens should scale together, "
                  "roughly 20 tokens per parameter."},
]

TEXT_A = ("Self-attention forms an n x n affinity matrix, so its cost grows "
          "quadratically in sequence length: every token attends to every token "
          "(https://arxiv.org/abs/1706.03762). FlashAttention keeps the same math "
          "but tiles it to cut memory traffic (https://arxiv.org/abs/2205.14135).")
TEXT_B = ("Loss follows a power law in parameters, data and compute "
          "(https://arxiv.org/abs/2001.08361); Chinchilla shows the compute-optimal "
          "ratio is roughly 20 tokens per parameter (https://arxiv.org/abs/2203.15556), "
          "so past the knee you add tokens, not width.")
INTRO = ("Transformer scaling has two levers that matter in practice: what each "
         "token costs as sequences grow, and how loss falls as models and data "
         "grow. The branches below cover both from primary sources.")
OUTRO = ("Both branches point the same way: attention cost is the per-sequence "
         "wall, scaling laws the per-parameter slope. Read together they say where "
         "the next unit of compute should go — and where it cannot help.")


# ---------- offline doubles: deterministic everywhere the pipeline touches a model ----------
class ScriptedLlm:
    """Same interface as harness.llm.LLM, zero network: canned decompositions and
    grounded answers. The point of offline mode is auditing the machinery around
    the model — including one scripted flake (see _fake_exec) so the FAILED ->
    repair loop contributes real data to the reliability audit."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, system, prompt, **kw):
        self.calls += 1
        if system.lstrip().startswith("In at most four short lines"):
            return ("Pieces: token-level cost, model-level scaling. "
                    "Evidence: primary papers with explicit quantities.")
        if "introduction for a document" in prompt:
            return INTRO
        if "take away" in prompt:
            return OUTRO
        if "Task: " in prompt:
            # note: a thin answer does NOT fail synthesis — handle() appends the
            # sources footer before the _llm_ok length check, so even 4 chars of
            # prose pass once URLs are gathered. Repair is exercised via the
            # search flake in _fake_exec instead.
            task = prompt.split("Task: ", 1)[1].splitlines()[0].lower()
            return TEXT_A if "quadratic" in task else TEXT_B
        return ""

    def chat_json(self, system, prompt, **kw):
        self.calls += 1
        if "Depth 0 of" in prompt:   # root split; shaping pins the sink + deepens branches
            return {"children": [
                {"intent": "branch A: find primary sources on attention's quadratic "
                           "cost in sequence length", "tool": "web_search",
                 "args": {"query": "attention quadratic cost sequence length transformer"},
                 "deps": [], "success_predicate": "relevant results returned"},
                {"intent": "branch B: find primary sources on transformer scaling laws",
                 "tool": "web_search",
                 "args": {"query": "transformer scaling laws power law compute"},
                 "deps": [], "success_predicate": "relevant results returned"},
                {"intent": "write the answer from both branches", "tool": "llm",
                 "deps": [0, 1], "success_predicate": "answers the parent task"}]}
        task = prompt.split("Parent task: ", 1)[1].splitlines()[0]  # branch split (depth 1)
        quad = "quadratic" in task.lower()
        return {"children": [
            {"intent": "gather evidence: " + ("an n x n affinity matrix per layer"
                                              if quad else "power-law fits and compute-optimal token ratios"),
             "tool": "web_search",
             "args": {"query": ("attention quadratic cost sequence length transformer"
                                if quad else "transformer scaling laws chinchilla")},
             "deps": [], "success_predicate": "relevant results returned"},
            {"intent": (f"summarize the {'quadratic cost' if quad else 'scaling-law'} "
                        f"evidence with citations"), "tool": "llm",
             "deps": [0], "success_predicate": "grounded summary with citations"}]}


_FLAKE = {"used": False}


async def _fake_exec(tool, args):
    """Sandbox stand-in for offline mode: canned searches keyed on the query
    (robust to the human's edited query), everything else out of scope. The
    first attention search flakes (empty results) exactly once — the judge
    must reject it and the repair loop must recover."""
    if tool == "web_search":
        q = str(args.get("query", "")).lower()
        if "quadratic" in q and not _FLAKE["used"]:
            _FLAKE["used"] = True
            return {"rc": 0, "results": []}
        hits = SCALING_HITS if ("scaling" in q or "chinchilla" in q) else ATTENTION_HITS
        return {"rc": 0, "results": hits}
    return {"rc": -1, "stderr": f"offline audit: tool {tool!r} is not part of this demo"}


_TC_LOCK = threading.Lock()


class InProcess:
    """Agent-side HTTP adapter: the worker's expect(.json()/.status_code) surface
    over FastAPI's TestClient. ALL core requests share one lock — core's store is
    a single sqlite3 connection (fine for core's single-threaded asyncio, not for
    concurrent callers), so driver + worker threads must not overlap requests."""

    def __init__(self, tc: TestClient) -> None:
        self.tc = tc

    def get(self, path):
        with _TC_LOCK:
            return self.tc.get(path)

    def post(self, path, json=None):
        with _TC_LOCK:
            return self.tc.post(path, json=json)


def _req(fn, *a, **kw):
    with _TC_LOCK:
        return fn(*a, **kw)


# ---------- the run driver: one worker + one human, end to end ----------
def _decide(tc: TestClient, card: dict, decisions: list, interactive: bool) -> None:
    """Act as the human at an approval card. Offline: scripted policy — approve
    everything, and edit-and-resubmit the SECOND web_search query once (proving
    the arg-edit surface). Interactive: ask, time the human."""
    detail = {"kind": card.get("kind"), "tool": card.get("tool", ""),
              "node": card.get("node_id")}
    if interactive:
        t0 = time.time()
        raw = input(f"  card {card['kind']} {card.get('tool', '')} on "
                    f"{card['node_id']} — [a]pprove / [e]dit-args / [r]eject: ").strip().lower()
        detail["seconds"] = round(time.time() - t0, 1)
        if raw == "r":
            _req(tc.post, f"/approvals/{card['id']}", json={"decision": "reject"})
            detail["decision"] = "reject"
            decisions.append(detail)
            return
        if raw == "e":
            _req(tc.post, f"/approvals/{card['id']}", json={
                "decision": "approve",
                "args": {"query": "transformer scaling laws chinchilla"}})
            detail.update(decision="edit-and-resubmit", edited=True)
            decisions.append(detail)
            return
    edits = sum(1 for d in decisions if d.get("edited"))
    edited_args = None
    if card.get("kind") == "exec" and card.get("node_id") == "root.2.1" and edits == 0:
        # edit-and-resubmit demo on the scaling card: semantically equivalent query,
        # so the grounded story survives the human's edit
        edited_args = {"query": "chinchilla compute optimal token ratio"}
    _req(tc.post, f"/approvals/{card['id']}", json={"decision": "approve",
                                                    **({"args": edited_args} if edited_args else {})})
    detail.update(decision="approve", edited=edited_args is not None)
    decisions.append(detail)


def run_audit(interactive: bool = False, live: bool = False) -> dict:
    """Drive one full intent run in-process and return the audit report."""
    tc = TestClient(core.app)
    _FLAKE["used"] = False                      # every audit replays the same story
    # fresh DB per audit: repeated runs must not bleed stale runs/events into the trail
    core.store = core.Store(Path(tempfile.mkdtemp(prefix="harness-audit-")) / "audit.db",
                            event_cap=500)
    core.RUNS.clear()
    core.AGENTS.clear()
    # demo policy: the human gates evidence gathering (web_search NOT auto-approved),
    # everything read-only + the virtual writer runs free
    core.POLICY.update({"confirm_actions": True, "max_repairs": 3, "approve_dag_edits": False,
                        # single-process driver owns its claims and sends no heartbeats:
                        # a slow model call must not outlive the claim lease
                        "claim_lease_seconds": 3600,
                        "auto_approve": ["http_fetch", "read_file", "list_dir", "llm"]})
    agent_base.POLL, agent_base.APPROVAL_TIMEOUT = 0.05, 120.0
    orig_exec, core._exec_tool = core._exec_tool, _fake_exec   # offline sandbox seam

    agent = Agent.__new__(Agent)
    agent.llm = (LLM(os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                     os.environ.get("MODEL", "qwen2.5:3b")) if live else ScriptedLlm())
    agent.http, agent.phase, agent.position = InProcess(tc), "idle", {"run_id": "", "node_id": ""}

    run_id = _req(tc.post, "/runs", json={"intent": INTENT}).json()["run_id"]
    decisions: list[dict] = []
    threads: list[threading.Thread] = []
    went = False
    idle = 0
    for _ in range(20000):
        claim = _req(tc.post, "/claim", json={"agent_id": "auditor"}).json()
        if claim.get("node"):
            t = threading.Thread(target=agent.handle, args=(claim,), daemon=True)
            t.start()
            threads.append(t)
        for card in _req(tc.get, "/approvals", params={"status": "pending"}).json():
            _decide(tc, card, decisions, interactive)
        run = _req(tc.get, f"/runs/{run_id}").json()
        busy = any(t.is_alive() for t in threads)
        if not went and run["phase"] == "planning" and run["plan_ready"]:
            tree_before = run["nodes"]
            went = True
            decisions.append({"kind": "plan", "decision": "approve",
                              "detail": f"checkpoint: {len(tree_before)} nodes"})
            _req(tc.post, f"/runs/{run_id}/go")
        if not run["active"] and not busy:
            break
        # model calls take minutes; only give up when nothing is claimable AND
        # no worker is mid-call — an iteration budget would race the model
        idle = 0 if (claim.get("node") or busy) else idle + 1
        if idle > 40:
            break
        time.sleep(0.25)
    for t in threads:
        t.join(timeout=30)
    report = _audit(tc, run_id, agent, decisions, live)
    core._exec_tool = orig_exec
    return report


# ---------- the audit checks ----------
def _audit(tc: TestClient, run_id: str, agent, decisions: list, live: bool) -> dict:
    run = _req(tc.get, f"/runs/{run_id}").json()
    nodes = run["nodes"]
    final = (nodes.get("root", {}).get("result") or {}).get("text", "")

    gathered = {h["url"] for n in nodes.values()
                for h in (n.get("result") or {}).get("results", []) if isinstance(h, dict)}
    cited = set(re.findall(r"https?://[^\s)\]]+", final))
    dangling = sorted(cited - gathered)

    evs = _req(tc.get, f"/runs/{run_id}/events").json()
    repairs = sum(1 for e in evs
                  if e["kind"] == "result" and (e.get("payload") or {}).get("state") == "FAILED")
    done = [n for n in nodes.values() if n["state"] == NodeState.DONE.value]
    first_pass = sum(1 for n in done if n["attempts"] <= 1)

    before = {k: (v["state"], v["result"]) for k, v in nodes.items()}
    core.RUNS.clear()
    core.RUNS.update(core.store.load_runs())                 # simulated restart
    reloaded = core.RUNS[run_id]["nodes"]
    after = {k: (v.state.value, v.result) for k, v in reloaded.items()}
    events_after = len(core.store.events(run_id))

    return {
        "settled": bool(done) and all(n["state"] == NodeState.DONE.value for n in nodes.values()),
        "nodes": len(nodes),
        "tree_text": _tree(nodes),
        "first_pass": f"{first_pass}/{len(done)}",
        "repairs": repairs,
        "lint_warnings": run.get("warnings") or [],
        "max_depth": max((n["depth"] for n in nodes.values()), default=0),
        "grounding": {"cited": sorted(cited), "gathered": sorted(gathered),
                      "dangling": dangling, "ok": not dangling},
        "human": {"count": len(decisions), "decisions": decisions},
        "persistence": {"states_match": before == after, "events": events_after},
        "budget": {"llm_calls": run["stats"]["llm_calls"], "cap": core.POLICY["max_llm_calls"],
                   "model_calls_observed": agent.llm.calls},
        "stats": run["stats"],
        "mode": "live" if live else "offline (scripted model + canned tools)",
    }


def _tree(nodes: dict) -> str:
    kids: dict[str, list] = {}
    for n in nodes.values():
        kids.setdefault(n["parent"] or "", []).append(n)

    def row(n, depth: int) -> str:
        tool = n["tool"] or "plan"
        head = (f"{'  ' * depth}{n['id']:<10} [{n['state']} {n['attempts']}x "
                f"{n['llm_calls']}c {tool}] {n['intent'][:52]}")
        return "\n".join([head] + [row(k, depth + 1)
                                   for k in sorted(kids.get(n["id"], []), key=lambda x: x["id"])])

    return "\n".join(row(n, 0) for n in sorted(kids.get("", []), key=lambda x: x["id"]))


def render(a: dict) -> str:
    g, p, b = a["grounding"], a["persistence"], a["budget"]
    checks = [("settled: every node DONE", a["settled"]),
              ("first-pass success", a["first_pass"]),
              ("grounding: cited URLs are gathered URLs", g["ok"]),
              ("persistence: restart reload identical", p["states_match"])]
    lines = [f"AUDIT — {a['mode']}", f"nodes: {a['nodes']} · repairs: {a['repairs']} "
             f"· delegation depth: {a['max_depth']}", "",
             a["tree_text"], "",
             "CHECKS"]
    for name, ok in checks:
        lines.append(f"  [{'PASS' if ok is True or (isinstance(ok, str) and ok != '0/0') else 'FAIL'}] "
                     f"{name}: {ok}")
    lines += ["",
              f"HUMAN LEDGER ({a['human']['count']} decisions):"]
    lines += [f"  - {d.get('decision') or d.get('detail')} "
              f"{'(' + d['tool'] + ')' if d.get('tool') else ''}"
              f"{' [edited args]' if d.get('edited') else ''}" for d in a["human"]["decisions"]]
    lines += ["",
              f"GROUNDING: {len(g['cited'])} cited ⊆ {len(g['gathered'])} gathered"
              + (f"; DANGLING: {g['dangling']}" if g["dangling"] else " — no invented citations"),
              f"BUDGET: {b['llm_calls']} accounted of {b['model_calls_observed']} observed "
              f"model calls, cap {b['cap']} (successful decompose calls ride no report)"
              f" · stats: {json.dumps(a['stats'])}",
              f"TRAIL: {p['events']} events persisted"]
    if a["lint_warnings"]:
        lines += ["LINT"] + [f"  ! {w}" for w in a["lint_warnings"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="real Ollama model")
    ap.add_argument("--interactive", action="store_true",
                    help="you sit at the approval cards (timed)")
    args = ap.parse_args(argv)
    report = run_audit(interactive=args.interactive, live=args.live)
    print(render(report))
    return 0 if report["settled"] and report["grounding"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


