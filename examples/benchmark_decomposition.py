#!/usr/bin/env python3
"""Flat vs tree: does decomposition actually solve multi-part problems better?

THE experiment for the product claim. Same model, same tasks, same evidence —
the only variable is structure:

  flat   — one chat call must answer every part of a multi-part question
  tree   — the full harness pipeline: decompose -> branch evidence -> branch
           writers -> head synthesis (real Agent + real model; tools canned)

The suite uses INVENTED facts in a canned corpus each arm receives (flat gets
it inline, tree gets it through canned tool results — whatever retrieval tool
its plan picks): memory cannot contaminate
the comparison, and scoring is a deterministic per-part substring check — the
eval never asks an LLM to grade itself.

Metrics per arm: part coverage (which arm DROPS parts — the known flat-model
failure), tasks fully correct, calls, wall time. The tree arm is additionally
audited (nodes, depth, grounding) with the same checks as examples/audit_run.py.

  python examples/benchmark_decomposition.py                    # qwen2.5:3b @ localhost
  MODEL=llama3.1:8b python examples/benchmark_decomposition.py  # another model
  python examples/benchmark_decomposition.py --limit 2          # smoke run
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("HARNESS_CONFIG", json.dumps({
    "data_dir": "/tmp/harness-decomp-bench", "sandbox_url": "http://stubbed"}))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import harness.agent.base as agent_base  # noqa: E402
import harness.core as core  # noqa: E402
from harness.agent import Agent  # noqa: E402
from harness.llm import LLM  # noqa: E402

# ---------- the suite: invented facts, four extractable parts each ----------
# Each corpus carries every answer; the question names its parts. Scoring is
# substring presence per part — deterministic, and useless to memorization.
SUITE: list[dict] = [
    {"q": "Report four facts about the Meridian telescope: the year it was built, "
          "the city it stands in, its designer, and its mirror diameter in meters.",
     "corpus": "The Meridian telescope was built in 1876 in the city of Valdara. Its "
               "designer Ivo Brandt specified a mirror diameter of 1.2 meters, unusually "
               "wide for the era. The dome was raised the following spring.",
     "parts": {"year built": "1876", "city": "Valdara", "designer": "Brandt",
               "mirror diameter": "1.2"}},
    {"q": "Give the four registered names of the Caldrein river: its source lake, "
          "the capital city on its banks, the estuary town, and its length in km.",
     "corpus": "The Caldrein rises at Lake Fenn, flows past the capital city of Ormstad, "
               "and reaches the sea at the estuary town of Kelp Bay. Its total length "
               "measured in the 1902 survey was 412 km.",
     "parts": {"source lake": "Fenn", "capital": "Ormstad", "estuary": "Kelp Bay",
               "length": "412"}},
    {"q": "Summarize the four data points from the Vess lab report: the compound id, "
          "the yield percent, the temperature in Celsius at which it stabilized, "
          "and the lead researcher's surname.",
     "corpus": "Compound KX-9 reached a yield of 73 percent when the reaction mixture "
               "stabilized at 84 degrees Celsius. Lead researcher Dr. Aiko Serrin signed "
               "the report; her team repeated the run twice with identical results.",
     "parts": {"compound id": "KX-9", "yield": "73", "temperature": "84",
               "lead researcher": "Serrin"}},
    {"q": "List the four awards the paper 'Sparse Lattices' won: the Tovin prize year, "
          "the Halder medal year, the journal that published it, and the page count.",
     "corpus": "'Sparse Lattices' won the Tovin prize in 2019 and the Halder medal in "
               "2021. It appeared in the Journal of Discrete Structures across 28 pages, "
               "including its appendix of constructions.",
     "parts": {"Tovin prize": "2019", "Halder medal": "2021",
               "journal": "Discrete Structures", "page count": "28"}},
    {"q": "State the four census figures for the town of Brindle: population, the "
          "number of schools, the mayor's surname, and the year the bridge opened.",
     "corpus": "Brindle's 2018 census recorded a population of 14,300 with nine schools. "
               "Mayor Sula Wexford oversaw the opening of the Wexford bridge in 2016, the "
               "town's first crossing.",
     "parts": {"population": "14,300", "schools": "nine", "mayor": "Wexford",
               "bridge opened": "2016"}},
    {"q": "Report the four specifications of the Ornis-4 drone: wingspan in meters, "
          "battery life in minutes, the manufacturer's home city, and the certification year.",
     "corpus": "The Ornis-4 drone has a wingspan of 3.4 meters and a battery life of 47 "
               "minutes. Its manufacturer, Polwerk Avionics of Drenthe, secured type "
               "certification in 2023.",
     "parts": {"wingspan": "3.4", "battery life": "47", "manufacturer city": "Drenthe",
               "certification year": "2023"}},
    {"q": "Give the four details of the tariff dispute: the two countries involved, "
          "the disputed good, the tariff percent, and the arbitration year.",
     "corpus": "The 2020 arbitration settled the tariff dispute between Velmark and "
               "Ostrand over imported slate tile, reducing the tariff to 18 percent over "
               "three years.",
     "parts": {"country one": "Velmark", "country two": "Ostrand",
               "disputed good": "slate", "tariff percent": "18"}},
    {"q": "State the four ingredients in the Ferringh recipe: the flour type, the "
          "cheese type, the herb, and the baking temperature in Celsius.",
     "corpus": "The Ferringh recipe mixes spelt flour with brined yzella cheese and the "
               "herb tarragon, baked at 210 degrees Celsius for forty minutes until the "
               "crust blisters.",
     "parts": {"flour": "spelt", "cheese": "yzella", "herb": "tarragon",
               "baking temperature": "210"}},
    # ---------- distractor-heavy: facts sit among irrelevant sentences ----------
    {"q": "Report four specifications of the Marchetto Dam: its height in meters, "
          "the reservoir name, the number of turbines, and the year it opened.",
     "corpus": "Villagers still hold the midsummer lantern fair along the river every "
                "June, a tradition older than the dam itself. The Marchetto Dam raises "
                "the water 46 meters into the Renno reservoir, named for the flooded "
                "village of Renno. Engineering trucks queued for weeks during the 1988 "
                "consolidation works. Its power hall houses seven Francis turbines, and "
                "the dam opened ceremonially in 1993, three years behind schedule. The "
                "lake's ferry runs twice daily except in winter storms.",
     "parts": {"height": "46", "reservoir": "Renno", "turbines": "seven",
               "opened": "1993"}},
    {"q": "State the four facts of the Tavolara rail line: its track gauge in meters, "
          "its length in km, the year the first train ran, and its operating company.",
     "corpus": "A competing tramway proposal along the coastal road was voted down "
                "three times before the railway won approval. The Tavolara rail line "
                "lays a 1.6-meter track gauge over 64 km of switchbacks between the "
                "plateau towns. Postcards of the first journey still sell at the "
                "terminus gift shop. The first train ran in 1911 with a brass band at "
                "every crossing. Since 1976 the line has been operated by Ferrovia "
                "Nordess, a worker-owned company that restored two steam engines for "
                "holiday runs.",
     "parts": {"gauge": "1.6", "length": "64", "first train": "1911",
               "operator": "Nordess"}},
    {"q": "Give the four registry details of the Larkspur clinical study: the number "
          "of patients, the follow-up period in months, the biomarker named, and the "
          "lead hospital.",
     "corpus": "Before Larkspur, two smaller registries failed to recruit their "
                "quotas. The Larkspur study enrolled 842 patients and followed each for "
                "thirty-six months, tracking the biomarker syndecan-2 in quarterly "
                "blood draws. Recruitment ads ran on evening radio in three provinces. "
                "The lead hospital was St. Aldric's Medical Centre in Bovelier, whose "
                "trial office published the method appendix separately. A sister study "
                "on inflammatory markers begins enrollment next spring.",
     "parts": {"patients": "842", "follow-up": "thirty-six", "biomarker": "syndecan-2",
               "hospital": "Aldric"}},
    {"q": "Report the four registry facts of the Kessert lighthouse: its tower height "
          "in meters, its light range in nautical miles, the keeper's surname who "
          "automated it, and the decade it was automated.",
     "corpus": "Fishermen on the Brel reef still curse the fog that hides the point "
                "twice a winter. The Kessert lighthouse stands 18 meters above the "
                "granite point, casting its light 21 nautical miles over the strait. "
                "Keeper Juhan Eskhar, who served almost until death, oversaw its "
                "automation in the 1980s, when the lamp room was rewired and the "
                "clockwork retired. The keeper's cottage now rents to weekend visitors "
                "from the mainland.",
     "parts": {"height": "18", "range": "21", "keeper": "Eskhar", "automated": "1980"}},
]

FLAT_SYS = ("You are a careful analyst. Answer EVERY part of the question using ONLY "
            "the provided material. Number each part.")


# ---------- scoring: deterministic, per part ----------
def score(answer: str, parts: dict[str, str]) -> dict:
    """One score per named part: the expected substring present in the answer
    (case-insensitive). The eval never asks an LLM to grade itself."""
    low = (answer or "").lower()
    hits = {name: exp.lower() in low for name, exp in parts.items()}
    return {"covered": sum(hits.values()), "total": len(parts), "hits": hits}


# ---------- flat arm: one call, everything at once ----------
def run_flat(llm, task: dict) -> dict:
    ask = (f"Material:\n{task['corpus']}\n\nQuestion: {task['q']}")
    t0 = time.time()
    answer = llm.chat(FLAT_SYS, ask, json_mode=False, model=getattr(llm, "model", ""),
                      temperature=0.2).strip()
    return {"answer": answer, "seconds": round(time.time() - t0, 1),
            "calls": 1, **score(answer, task["parts"])}


# ---------- tree arm: the real pipeline ----------
_CORPUS = {"text": ""}
_FILES: dict[str, str] = {}


def _run_shell(cmd: str) -> dict:
    """Run the model's command for real against a workspace holding the canned
    corpus under every filename the command mentions."""
    with tempfile.TemporaryDirectory() as td:
        for n in set(re.findall(r"[\w./-]+\.\w+", cmd)):
            (Path(td) / Path(n).name).write_text(_FILES.get(Path(n).name, _CORPUS["text"]))
        try:
            r = subprocess.run(cmd, shell=True, cwd=td, capture_output=True,
                               text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {"rc": -1, "stdout": "", "stderr": "timeout"}
    return {"rc": r.returncode, "stdout": r.stdout[:4000], "stderr": r.stderr[:1000]}


async def _canned_exec(tool, args):
    """Sandbox seam: the canned corpus is the world's only evidence. Serve it
    through WHICHEVER retrieval tool the model's plan picks (web_search hits,
    read_file, shell over a corpus file), shape-identical to harness/sandbox.py
    results — tool-choice luck is not the variable being measured."""
    name = Path(args.get("path", "")).name
    if tool == "web_search":
        return {"rc": 0, "results": [{"title": "source document", "url": "https://canned.local/doc",
                                      "page_text": _CORPUS["text"]}]}
    if tool == "read_file":
        return {"content": _FILES.get(name, _CORPUS["text"])}
    if tool == "write_file":
        _FILES[name] = args.get("content", "")
        return {"bytes": len(_FILES[name])}
    if tool == "shell":
        return _run_shell(args.get("cmd", ""))
    return {"rc": -1, "stdout": "", "stderr": f"benchmark: tool {tool!r} not in scope"}


class TreeAgent(Agent):
    """Real Agent, real model, canned sandbox. The scripted decompose plan comes
    from the model itself — that is what is being measured."""

    def _ensure_guide(self, claim, pa):   # skip the guide nicety: not the variable
        pass


def run_tree(llm, task: dict, tc: TestClient, run_id: str) -> dict:
    """Drive one full harness run for the task and return outcome + audit stats."""
    _CORPUS["text"] = task["corpus"]
    _FILES.clear()
    agent = TreeAgent.__new__(TreeAgent)
    agent.llm, agent.http = llm, _InProc(tc)
    agent.phase, agent.position = "idle", {"run_id": "", "node_id": ""}
    threads: list[threading.Thread] = []
    went = False
    idle = 0
    for _ in range(20000):
        claim = _req(tc.post, "/claim", json={"agent_id": "bench"}).json()
        if claim.get("node"):
            t = threading.Thread(target=agent.handle, args=(claim,), daemon=True)
            t.start()
            threads.append(t)
        run = _req(tc.get, f"/runs/{run_id}").json()
        busy = any(t.is_alive() for t in threads)
        if not went and run["phase"] == "planning" and run["plan_ready"]:
            went = True
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
    run = _req(tc.get, f"/runs/{run_id}").json()
    answer = (run["nodes"].get("root", {}).get("result") or {}).get("text", "")
    gathered = {h["url"] for n in run["nodes"].values()
                for h in (n.get("result") or {}).get("results", []) if isinstance(h, dict)}
    cited = set(re.findall(r"https?://[^\s)\]]+", answer))
    return {"answer": answer, "nodes": len(run["nodes"]),
            "max_depth": max((n["depth"] for n in run["nodes"].values()), default=0),
            "settled": all(n["state"] == "DONE" for n in run["nodes"].values()),
            "grounded": cited <= gathered,
            **score(answer, task["parts"])}


class _InProc:
    """All core requests share ONE lock: core's store is a single sqlite3
    connection (fine for core's single-threaded asyncio, not for concurrent
    callers), so driver + worker threads must not overlap requests."""

    def __init__(self, tc):
        self.tc = tc

    def get(self, path):
        with _TC_LOCK:
            return self.tc.get(path)

    def post(self, path, json=None):
        with _TC_LOCK:
            return self.tc.post(path, json=json)


_TC_LOCK = threading.Lock()


def _req(fn, *a, **kw):
    with _TC_LOCK:
        return fn(*a, **kw)


def fresh_run(tc: TestClient, task: dict) -> str:
    return _req(tc.post, "/runs", json={"intent": task["q"]}).json()["run_id"]


# ---------- report ----------
def report(rows: list[dict], model: str) -> str:
    """Markdown: flat vs tree on coverage/accuracy/cost, plus the tree's audit stats."""
    def agg(arm: str) -> str:
        rs = [r for r in rows if r["arm"] == arm]
        cov = sum(r["covered"] for r in rs)
        tot = sum(r["total"] for r in rs)
        full = sum(1 for r in rs if r["covered"] == r["total"])
        calls = sum(r["calls"] for r in rs) / len(rs)
        secs = sum(r["seconds"] for r in rs) / len(rs)
        return (f"| {arm} | {cov}/{tot} ({100 * cov // tot}%) | {full}/{len(rs)} | "
                f"{calls:.1f} | {secs:.1f} |")

    lines = [f"Model: {model}", "",
             "| arm | parts covered | tasks fully correct | calls/task | s/task |",
             "|---|---|---|---|---|", agg("flat"), agg("tree")]
    tree = [r for r in rows if r["arm"] == "tree"]
    if tree:
        nodes = sum(r["nodes"] for r in tree) / len(tree)
        depth = max(r["max_depth"] for r in tree)
        lines += ["", f"tree shape: {nodes:.1f} nodes/task, delegation depth {depth}, "
                      f"settled {sum(r['settled'] for r in tree)}/{len(tree)}, "
                      f"grounded {sum(r['grounded'] for r in tree)}/{len(tree)}"]
    # per-task matrix: coverage conflates pipeline reliability (did the run
    # settle at all) with content quality (settled runs dropping parts) — split
    flat_by = {r["task"]: r for r in rows if r["arm"] == "flat"}
    tree_by = {r["task"]: r for r in rows if r["arm"] == "tree"}
    common = sorted(set(flat_by) & set(tree_by))
    lines += ["", "| task | flat | tree | settled | winner |", "|---|---|---|---|---|"]
    for i in common:
        f, t = flat_by[i], tree_by[i]
        win = ("tree" if t["covered"] > f["covered"]
               else "flat" if t["covered"] < f["covered"] else "tie") \
            if t["settled"] else "unsettled"
        lines.append(f"| {i} | {f['covered']}/{f['total']} | {t['covered']}/{t['total']} | "
                     f"{'yes' if t['settled'] else 'no'} | {win} |")
    st = [i for i in common if tree_by[i]["settled"]]
    if st:
        fcov = sum(flat_by[i]["covered"] for i in st)
        tcov = sum(tree_by[i]["covered"] for i in st)
        tot = sum(flat_by[i]["total"] for i in st)
        lines += ["", f"settled tasks only ({len(st)}/{len(common)}): flat {fcov}/{tot} "
                      f"vs tree {tcov}/{tot} — content quality, reliability removed"]
    return "\n".join(lines)


def run_suite(make_llm, tasks: list[dict], tc: TestClient, core_module) -> list[dict]:
    rows = []
    for i, task in enumerate(tasks):
        llm = make_llm()
        flat = run_flat(llm, task)
        rows.append({"arm": "flat", "task": i, **flat})
        core_module.RUNS.clear()
        run_id = fresh_run(tc, task)
        tree_llm = make_llm()
        t0 = time.time()
        tree = run_tree(tree_llm, task, tc, run_id)
        tree["seconds"] = round(time.time() - t0, 1)
        tree["calls"] = tree_llm.calls
        rows.append({"arm": "tree", "task": i, **tree})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "qwen2.5:3b"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="drop the first N tasks")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="", help="raw results JSON path")
    args = ap.parse_args(argv)
    tasks = SUITE[args.skip:]
    if args.limit:
        tasks = tasks[: args.limit]

    tc = TestClient(core.app)
    core.store = core.Store(Path(tempfile.mkdtemp(prefix="harness-decomp-")) / "bench.db",
                            event_cap=500)
    core.RUNS.clear()
    core.POLICY.update({"confirm_actions": True, "max_repairs": 3,
                        "approve_dag_edits": False,
                        # single-process driver owns its claims and sends no
                        # heartbeats: without this, a slow model call outlives
                        # the lease and core requeues the node mid-call
                        "claim_lease_seconds": 3600,
                        "auto_approve": ["web_search", "http_fetch", "read_file",
                                         "list_dir", "llm", "shell", "write_file"]})
    agent_base.POLL, agent_base.APPROVAL_TIMEOUT = 0.05, 60.0
    orig_exec, core._exec_tool = core._exec_tool, _canned_exec
    try:
        make_llm = lambda: LLM(args.url, args.model, timeout=args.timeout)  # noqa: E731
        rows = run_suite(make_llm, tasks, tc, core)
    finally:
        core._exec_tool = orig_exec
    print(report(rows, args.model))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()


