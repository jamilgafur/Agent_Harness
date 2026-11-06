"""Agent worker: claim → execute (approval-gated, via core) → judge → report,
or decompose abstract nodes via LLM with a validate-and-resubmit repair loop.

Decomposition is shape-pinned deterministically after the model speaks, in this order:
  chat_json -> cap 4 (slot for sink) -> _force_synthesis (flatten gather chains,
  ground lone/all-llm plans, one convergence sink) -> _deepen_branches (root-level
  research plans become abstract-branch trees). The model proposes; these decide.

Communicates per paper III-B: Task (claim), Result (report), Redirect (handoff),
Heartbeat (phase + position, keeps the lease alive during long LLM calls).

Debug a stuck worker live: `docker compose kill -s USR1 agent` -> traceback in logs.

Runnable with zero LLM calls when nodes carry auto-approved tools and no
success_predicate (template DAGs) — that path is the offline acceptance test.
"""
import faulthandler
import json
import os
import re
import signal
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import load
from .llm import LLM

# debug: `docker compose kill -s USR1 agent` dumps a live traceback to the logs
signal.signal(signal.SIGUSR1, lambda *_: faulthandler.dump_traceback())

CFG = load()
POLICY: Dict[str, Any] = CFG["policy"]
AGENT_ID = os.environ.get("AGENT_ID") or f"agent-{socket.gethostname()[-4:]}"  # unique per scaled container
POLL = float(os.environ.get("POLL_INTERVAL", "2"))
HEARTBEAT = float(os.environ.get("HEARTBEAT_SECONDS", POLICY.get("heartbeat_seconds", 5)))
CORE = CFG["core_url"]

DECOMPOSE_SYS = """You decompose one task into 2-4 smaller subtasks for a DAG.
Reply with JSON only:
{"children":[{"intent":"...","tool":"<one of the Available tools, or null>","args":{...},
"deps":[0],"success_predicate":"..."}]}
Rules: deps are indices of earlier siblings in YOUR list (not of the parent).
tool=null means the subtask needs further decomposition. Ground work in tools when possible.
args must contain exactly the required keys for the chosen tool.
Every child with a tool needs a success_predicate (a checkable outcome).
Use tool "llm" (no args) for synthesis/analysis/answering: it receives all its deps'
results as material and WRITES the answer — prefer it over write_file for prose.
For knowledge/research questions the ONLY acceptable shape is gather-branches then synthesize.
Gather facts with tools first — llm children may ONLY synthesize material their deps produced,
never facts from their own memory. Example for "tell me about types of neural network pruning":
{"children":[
 {"intent":"research structured pruning of neural networks","tool":"web_search",
  "args":{"query":"structured pruning neural networks"},"success_predicate":"relevant hits returned"},
 {"intent":"research unstructured pruning of neural networks","tool":"web_search",
  "args":{"query":"unstructured pruning neural networks"},"success_predicate":"relevant hits returned"},
 {"intent":"write the answer about pruning types","tool":"llm","deps":[0,1],
  "success_predicate":"answers the parent question using the gathered material"}]}
Keep subtopic terms from the parent task in every search query (do not drop qualifiers).
For research/\"tell me\"/paper tasks the synthesis child MUST be the LAST child and its deps
MUST list EVERY other child (all gather branches converge into it); nothing may depend on
the synthesis child. Independent gather children MUST have empty deps —
chain a child onto another ONLY when it truly needs that child's output.
Large/broad tasks (papers, surveys, \"main approaches to X\"): FIRST split into 2-3 abstract
branch children (tool=null, one per subtopic/section) plus ONE llm child dep-ing on ALL of
them — branch children will decompose further by these same rules (search leaves, an llm
summary per branch), answers flowing up the tree.
If the prompt lists neighboring branches, split work that COMPLEMENTS them — never duplicate
a search or subtopic a neighbor already owns (neighbor context is what keeps branches independent)."""

SYNTH_SYS = """You write grounded, specific answers for the user based ONLY on the material
provided from other DAG branches. Match the depth the task asks for: a paper,
report or survey gets full structured sections; a "tell me about" gets a compact brief.
KEEP THE CONCRETE SPECIFICS — numbers, names, mechanisms, dates. Specifics are what
make an answer useful; vague paraphrases ("it improves efficiency") are failure.
Ground every claim in a page that was actually READ (page_text present in the
material) and cite that page's URL in parentheses when you use it.
If the material is thin or missing, say what is missing at the end instead of inventing."""

JUDGE_SYS = """You judge whether a tool execution satisfied the success predicate.
Reply with JSON only: {"ok": true|false, "why": "short reason"}"""


def _material(inputs: Dict[str, Any], limit: int = 6500) -> str:
    """Render dep results into readable material for the synthesis prompt.
    Generous limit: the head must see full branch summaries, not clips."""
    parts = []
    for nid, res in inputs.items():
        if res is None:
            continue
        if isinstance(res, dict) and isinstance(res.get("results"), list):
            body = "\n".join(
                f"- {x.get('title','')}: {x.get('url','')}\n  "
                + ((x.get("page_text") or x.get("snippet") or "")[:700])
                for x in res["results"][:8])
        elif isinstance(res, dict) and "text" in res:
            body = str(res["text"])
        elif isinstance(res, dict) and "content" in res:
            body = str(res["content"])
        elif isinstance(res, dict) and any(isinstance(x, dict) and x.get("text")
                                           for x in res.values()):
            # aggregate parent result: prefer the children's synthesis texts —
            # that's the train of thought climbing up, not the raw leaves
            body = "\n\n".join(f"── {k} ──\n{x['text']}"
                               for k, x in res.items()
                               if isinstance(x, dict) and x.get("text"))
        else:
            body = json.dumps(res, default=str)[:600]
        parts.append(f"[from {nid}]\n{body}")
    return ("\n\n".join(parts))[:limit] or "(no material was gathered — say so in the answer)"


def _llm_ok(text: str) -> bool:
    """Deterministic acceptance for synthesis: present and substantive.
    LLM-judging prose against model-written predicates was brittle (false 'unmet'
    verdicts escalated whole runs); success_predicate stays as guidance only."""
    return bool(text) and len(text) >= 40


def _sources(inputs: Dict[str, Any], limit: int = 12) -> str:
    """Deterministic citation footer: the URLs the material actually came from —
    raw search hits AND the footers of climbing branch summaries (parents hold
    text, not dumps, so links must also be read back out of the prose)."""
    urls: List[str] = []

    def collect(v: Any) -> None:
        if isinstance(v, dict):
            for h in v.get("results") or []:
                if isinstance(h, dict) and h.get("url"):
                    urls.append(h["url"])
            for x in v.values():
                collect(x)
        elif isinstance(v, str) and "http" in v:
            urls.extend(re.findall(r"https?://[^\s)\]]+", v))

    for v in inputs.values():
        collect(v)
    seen = [u.rstrip(".,") for u in dict.fromkeys(urls)][:limit]
    return "\n\nSources:\n" + "\n".join(f"- {u}" for u in seen) if seen else ""


GATHER_TOOLS = ("web_search", "http_fetch")


def _force_synthesis(children: List[Dict[str, Any]], intent: str) -> List[Dict[str, Any]]:
    """Research-shaped decompositions must converge on ONE llm synthesis sink fed
    by every other child. A 3b model places it wrong ~half the time (or skips
    grounding entirely, answering from memory), so pin the shape deterministically
    in the same spot where we normalize model output."""
    has_gather = any(c.get("tool") in GATHER_TOOLS for c in children)
    has_llm = any(c.get("tool") == "llm" for c in children)
    if not has_gather and not has_llm:
        return children   # pure task shape (write_file/shell) — leave it
    kids = [dict(c) for c in children]
    n = len(kids)
    for i, c in enumerate(kids):  # clamp stale indices; deps arrive as sibling indices
        c["deps"] = [d for d in c.get("deps") or []
                     if isinstance(d, int) and 0 <= d < n and d != i]
    # flatten degenerate prefix chains: gather children deps'd only on earlier
    # gather children are independent (args are static — no data flows between
    # them), so cumulative chaining just serializes execution for no reason
    gidx = {i for i, c in enumerate(kids) if c.get("tool") in GATHER_TOOLS}
    if len(gidx) >= 2 and all(all(d in gidx for d in kids[i]["deps"]) for i in gidx):
        for i in gidx:
            kids[i]["deps"] = []
    if not has_gather:
        llms = [i for i, c in enumerate(kids) if c.get("tool") == "llm"]
        if not llms:
            return kids
        if len(llms) == 1 and n == 1:
            # a lone writer answering from memory: ground it instead
            kids[0].update({"tool": "web_search",
                            "args": {"query": kids[0]["intent"][:300]},
                            "success_predicate": "relevant hits returned"})
            kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                         "tool": "llm", "args": {}, "deps": [0],
                         "success_predicate": "answers the parent task from the gathered material"})
            return kids
        sink = llms[-1]  # last writer always becomes the convergence sink
        if len(llms) >= 2:  # earlier writers become grounding searches
            for i in llms[:-1]:
                kids[i].update({"tool": "web_search",
                                "args": {"query": kids[i]["intent"][:300]},
                                "success_predicate": "relevant hits returned"})
        for i, c in enumerate(kids):
            if i != sink and sink in c["deps"]:
                c["deps"].remove(sink)
        kids[sink]["deps"] = [i for i in range(n) if i != sink]
        return kids
    syn = next((i for i, c in enumerate(kids) if c.get("tool") == "llm"), None)
    if syn is None:
        kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                     "tool": "llm", "args": {}, "deps": list(range(n)),
                     "success_predicate": "answers the parent task from the gathered material"})
        return kids
    for i, c in enumerate(kids):  # nothing downstream of synthesis may point back at it
        if i != syn and syn in c["deps"]:
            c["deps"].remove(syn)
    kids[syn]["deps"] = [i for i in range(n) if i != syn]
    return kids


def _deepen_branches(children: List[Dict[str, Any]], depth: int) -> List[Dict[str, Any]]:
    """Root-level research plans become real TREES: each gather branch turns into an
    abstract subtask (tool=null) that re-decomposes into its own searches + summary
    (root.N.M ids), syntheses aggregating upward. Gated to depth 0 — branch nodes
    decompose shallowly on their own (search leaves + one summary each)."""
    if depth != 0 or len(children) < 2:
        return children
    llms = [i for i, c in enumerate(children) if c.get("tool") == "llm"]
    gathers = [i for i, c in enumerate(children) if c.get("tool") in GATHER_TOOLS]
    if len(llms) > 1 or len(gathers) != len(children) - len(llms):
        return children   # shell/write_file in the mix — a task plan, leave it
    kids = [dict(c) for c in children]
    for i in gathers:
        kids[i]["tool"], kids[i]["args"], kids[i]["deps"] = None, {}, []
    return kids


class Agent:
    def _upstream(self, run_id: str, node: Dict[str, Any]) -> Dict[str, Any]:
        """Traverse the DAG upward (deps + parents) and gather ancestor results.
        Runs pull the run state once — every read is an API call the core can audit."""
        try:
            nodes = self.http.get(f"/runs/{run_id}").json()["nodes"]
        except Exception:
            return {}
        out: Dict[str, Any] = {}
        todo = list(node.get("deps") or []) + ([node["parent"]] if node.get("parent") else [])
        seen = set(todo)
        while todo:
            nid = todo.pop()
            anc = nodes.get(nid)
            if not anc:
                continue
            if anc.get("state") == "DONE" and anc.get("result") is not None:
                out[nid] = anc["result"]
            ups = list(anc.get("deps") or []) + ([anc["parent"]] if anc.get("parent") else [])
            for up in ups:
                if up and up not in seen:
                    seen.add(up)
                    todo.append(up)
        return out

    def __init__(self) -> None:
        self.llm = LLM(CFG["ollama_url"], CFG["model"])
        self.http = httpx.Client(base_url=CORE, timeout=30)
        self.phase = "idle"
        self.position: Dict[str, str] = {"run_id": "", "node_id": ""}

    # ---------- heartbeat (III-B Heartbeat) ----------
    def _heartbeat_loop(self) -> None:
        while True:
            try:
                self.http.post("/heartbeat", json={"agent_id": AGENT_ID,
                                                   **self.position, "phase": self.phase})
            except Exception:
                pass
            time.sleep(HEARTBEAT)

    def _set(self, phase: str, run_id: str = "", node_id: str = "") -> None:
        self.phase = phase
        self.position = {"run_id": run_id, "node_id": node_id}

    # ---------- LLM-assisted pieces ----------
    def decompose(self, node: Dict[str, Any], inputs: Dict[str, Any],
                  tools: Dict[str, Any], max_depth: int,
                  constraints: Optional[List[str]] = None,
                  context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        spec = "; ".join(f"{name}(args: {', '.join(s['required'])})"
                         for name, s in tools.items())
        # bound dep results: one fat search dump must not blow the model's ctx
        brief = {k: json.dumps(v, default=str)[:600] for k, v in inputs.items()}
        sibs = "; ".join(f"{s['id']}: {str(s['intent'])[:60]}"
                         for s in (context or {}).get("siblings", [])[:8])
        prompt = (f"Parent task: {node['intent']}\nDepth {node['depth']} of {max_depth}.\n"
                  f"User constraints the plan must respect: "
                  f"{'; '.join(constraints or []) or 'none'}\n"
                  f"Overall run goal: {str((context or {}).get('run_intent'))[:160]}\n"
                  + (f"This task sits under branch: "
                     f"{str(context['parent_intent'])[:120]}\n"
                     if context and context.get("parent_intent") else "")
                  + (f"Neighboring branches (do NOT duplicate their work): {sibs}\n"
                     if sibs else "")
                  + f"Available tools with required args: {spec}\n"
                  f"Inputs from dependencies: {brief}\n"
                  f"args keys must match the required names exactly (e.g. write_file takes path and content).\n"
                  f"Hint from previous failure: {node.get('hint') or 'none'}")
        out = self.llm.chat_json(DECOMPOSE_SYS, prompt)
        if not out or not isinstance(out.get("children"), list) or not out["children"]:
            return None
        kids = out["children"][:4]
        # leave a slot for the synthesis sink if the planner forgot one
        if len(kids) == 4 and any(c.get("tool") in GATHER_TOOLS for c in kids) \
                and not any(c.get("tool") == "llm" for c in kids):
            kids = kids[:3]
        kids = _force_synthesis(kids, node["intent"])
        return {"children": _deepen_branches(kids, node.get("depth", 0))}

    def _write_answer(self, run_id: str, node: Dict[str, Any],
                      inputs: Dict[str, Any], material: str,
                      constraints: List[str]) -> str:
        """Write a node's text. The RUN'S HEAD writer (parent == root, several
        branch summaries in) is assembled deterministically: intro + Bottom line
        from the model (small, 3b-safe asks), body = the verified branch summaries
        verbatim as sections — nothing gets dropped in the funnel and the answer
        keeps the tree's coverage. Everyone else: a single synthesis call."""
        pre = (f"User constraints to respect: {'; '.join(constraints)}\n"
               if constraints else "")
        if node.get("parent") == "root":
            rn = self.http.get(f"/runs/{run_id}").json()["nodes"]
            sections: List[Tuple[str, str]] = []
            for d, res in inputs.items():
                if isinstance(res, dict) and res.get("text") and d in rn:
                    t = re.sub(r"\n+Sources:\n(?:- .+\s*)+$", "", res["text"]).strip()
                    if t:
                        sections.append((rn[d]["intent"], t))
            if len(sections) >= 2:
                title = rn.get("root", {}).get("intent") or node["intent"]
                intro = self.llm.chat(
                    SYNTH_SYS,
                    pre + f"Write ONLY a 3-4 sentence introduction for a document "
                          f"titled \"{title}\" — no headings, no lists, prose only.\n"
                          f"Material from the neighboring nodes:\n{material}",
                    json_mode=False).strip()
                outro = self.llm.chat(
                    SYNTH_SYS,
                    pre + f"Write ONLY 4-6 concluding sentences comparing the sections "
                          f"and saying what the reader should take away, for a document "
                          f"titled \"{title}\" — no headings, prose only.\n"
                          f"Material from the neighboring nodes:\n{material}",
                    json_mode=False).strip()
                parts = [f"# {title}"]
                if intro:
                    parts.append(intro)
                parts += [f"## {s}\n\n{t}" for s, t in sections]
                if outro:
                    parts.append(f"## Bottom line\n\n{outro}")
                return "\n\n".join(parts)
        return self.llm.chat(
            SYNTH_SYS,
            pre + f"Task: {node['intent']}\n"
                  f"Material from the neighboring nodes:\n{material}",
            json_mode=False).strip()

    def judge(self, node: Dict[str, Any], exec_result: Dict[str, Any]) -> bool:
        pred = node.get("condition") or node.get("success_predicate", "")
        if node.get("tool") in GATHER_TOOLS:
            # deterministic, same rationale as _llm_ok: page_text makes the payload
            # big enough that a small LLM-judge silently fails good searches
            return bool(exec_result.get("results") or exec_result.get("content"))
        if not pred:
            return exec_result.get("rc", 0) == 0 if node.get("tool") == "shell" else True
        out = self.llm.chat_json(JUDGE_SYS,
                                 f"Task: {node['intent']}\nPredicate: {pred}\n"
                                 f"Execution result: {exec_result}")
        return bool(out and out.get("ok") is True)   # strict: only explicit true passes

    # ---------- approval round-trips ----------
    def _wait(self, exec_id: str) -> Dict[str, Any]:
        """Poll an approval/exec until decided. Result dicts carry rc>=-1;
        rejected-by-user returns rc=-2."""
        while True:
            time.sleep(POLL)
            st = self.http.get(f"/exec/{exec_id}").json()
            if st["status"] == "done":
                return st["result"]
            if st["status"] == "rejected":
                return {"rc": -2, "stderr": "rejected by user"}

    def exec_via_core(self, run_id: str, node: Dict[str, Any]) -> Dict[str, Any]:
        r = self.http.post(f"/runs/{run_id}/exec",
                           json={"node_id": node["id"], "tool": node["tool"],
                                 "args": node.get("args", {})})
        data = r.json()
        if r.status_code != 200:
            return {"rc": -1, "stderr": f"exec rejected: {data}"}
        if data["status"] == "done":
            return data["result"]
        return self._wait(data["exec_id"])

    # ---------- main loop ----------
    def handle(self, claim: Dict[str, Any]) -> None:
        run_id, node, inputs = claim["run_id"], claim["node"], claim["inputs"]
        calls_before = self.llm.calls
        try:
            if node.get("tool"):
                if node["tool"] == "llm":   # virtual: synthesize from neighbor results
                    self._set("writing", run_id, node["id"])
                    # train of thought: each writer reads only its own level's
                    # material (direct deps); branch summaries carry what rose
                    # from below — see _material's summary-preference rule
                    material = _material(inputs)
                    self._set("writing", run_id, node["id"])
                    cons = claim.get("constraints") or []
                    try:
                        text = self._write_answer(run_id, node, inputs, material, cons)
                    except Exception as e:
                        self.report(run_id, node, "FAILED", hint=f"llm call failed: {e}",
                                    before=calls_before)
                        return
                    text += _sources(inputs)
                    res = {"text": text}
                    if _llm_ok(text):
                        self.report(run_id, node, "DONE", result=res, before=calls_before)
                    else:
                        self.report(run_id, node, "FAILED",
                                    hint="synthesis produced an empty/thin answer",
                                    before=calls_before)
                    return
                self._set("executing", run_id, node["id"])
                res = self.exec_via_core(run_id, node)
                if res.get("rc", 0) == -2:      # user rejected: hand off with the reason (III-B Redirect)
                    self.http.post(f"/runs/{run_id}/nodes/{node['id']}/redirect",
                                   json={"agent_id": AGENT_ID,
                                         "reason": res.get("stderr", "rejected")})
                    return
                if res.get("rc", 0) < 0:
                    self.report(run_id, node, "FAILED", hint=res.get("stderr", "exec error"),
                                before=calls_before)
                    return
                self._set("judging", run_id, node["id"])
                if self.judge(node, res):
                    self.report(run_id, node, "DONE", result=res, before=calls_before)
                else:
                    self.report(run_id, node, "FAILED",
                                hint=f"predicate unmet: {node.get('success_predicate')}",
                                before=calls_before)
            else:
                self._set("decomposing", run_id, node["id"])
                # planners travel upward too: full ancestor context, direct deps win
                ctx = {**self._upstream(run_id, node), **inputs}
                sub = self.decompose(node, ctx, claim["tools"], claim["max_depth"],
                                     claim.get("constraints"), claim.get("context"))
                errs: Any = "LLM returned nothing usable" if sub is None else None
                for _ in range(POLICY["max_repairs"]):
                    if errs is None:
                        r = self.http.post(f"/runs/{run_id}/nodes/{node['id']}/decompose",
                                           json=sub)
                        if r.status_code == 200:
                            if r.json().get("status") == "pending_approval":
                                self._set("waiting-approval", run_id, node["id"])
                                out = self._wait(r.json()["exec_id"])
                                if out.get("rc") == -2:   # user rejected patch: re-decompose with the hint baked by core
                                    return
                            return
                        errs = r.json().get("detail", r.text)
                    sub = self.decompose(
                        {**node, "hint": f"previous decomposition rejected: {errs}"},
                        inputs, claim["tools"], claim["max_depth"])
                    errs = None if sub else "LLM returned nothing usable"
                self.report(run_id, node, "FAILED",
                            hint=f"cannot decompose: {errs}", before=calls_before)
        except Exception as e:  # network blip, LLM down, ...
            try:
                self.report(run_id, node, "FAILED", hint=f"agent error: {e}", before=calls_before)
            except Exception:
                pass
        finally:
            self._set("idle")

    def report(self, run_id: str, node: Dict[str, Any], state: str,
               result: Any = None, hint: str = "", before: int = 0) -> None:
        r = self.http.post(f"/runs/{run_id}/nodes/{node['id']}/result",
                           json={"state": state, "result": result, "hint": hint,
                                 "llm_calls": self.llm.calls - before, "agent_id": AGENT_ID})
        if r.status_code == 409:
            print(f"[{AGENT_ID}] stale report for {node['id']} ignored (lease expired)")

    def run(self) -> None:
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print(f"[{AGENT_ID}] polling {CORE} (ollama={CFG['ollama_url']} model={CFG['model']})")
        while True:
            try:
                claim = self.http.post("/claim", json={"agent_id": AGENT_ID}).json()
                if claim.get("node"):
                    print(f"[{AGENT_ID}] claimed {claim['node']['id']}: {claim['node']['intent'][:80]}")
                    self.handle(claim)
                else:
                    self._set("idle")
                    time.sleep(POLL)
            except Exception as e:
                print(f"[{AGENT_ID}] poll error: {e}")
                time.sleep(POLL * 3)


if __name__ == "__main__":
    Agent().run()
