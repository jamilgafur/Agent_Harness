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
from typing import Any

import httpx

from .config import load
from .graph import named_parts
from .llm import LLM
from .prompts import DEFAULT_PARAMS, DEFAULT_PROMPTS, resolve_params, resolve_prompt

# debug: `docker compose kill -s USR1 agent` dumps a live traceback to the logs
signal.signal(signal.SIGUSR1, lambda *_: faulthandler.dump_traceback())

CFG = load()
POLICY: dict[str, Any] = CFG["policy"]
AGENT_ID = os.environ.get("AGENT_ID") or f"agent-{socket.gethostname()[-4:]}"  # unique per scaled container
POLL = float(os.environ.get("POLL_INTERVAL", "2"))
APPROVAL_TIMEOUT = float(os.environ.get("APPROVAL_TIMEOUT", "1800"))  # unanswered card gives up
HEARTBEAT = float(os.environ.get("HEARTBEAT_SECONDS", POLICY.get("heartbeat_seconds", 5)))
CORE = CFG["core_url"]

# Canonical prompt text lives in prompts.py (single source of truth, per-run
# overridable via PUT /runs/{id}/config). Aliases keep the old names below.
DECOMPOSE_SYS = DEFAULT_PROMPTS["decompose"]
SYNTH_SYS = DEFAULT_PROMPTS["synthesize"]
JUDGE_SYS = DEFAULT_PROMPTS["judge"]


def _material(inputs: dict[str, Any], limit: int = 6500) -> str:
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


def _llm_down(exc: BaseException) -> bool:
    """LLM runner unreachable/mid-crash: timeouts, disconnects, server 5xx.
    Distinguishes 'infra paused' from 'this node is bad' — the repair loop must
    not spend a node's attempts waiting out a wedged ollama (300s timeouts)."""
    if isinstance(exc, (httpx.ReadTimeout, httpx.RemoteProtocolError,
                      httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    return type(exc).__name__ == "ResponseError" and getattr(exc, "status_code", 0) >= 500


def _llm_ok(text: str) -> bool:
    """Deterministic acceptance for synthesis: present and substantive.
    LLM-judging prose against model-written predicates was brittle (false 'unmet'
    verdicts escalated whole runs); success_predicate stays as guidance only."""
    return bool(text) and len(text) >= 40


def _sources(inputs: dict[str, Any], limit: int = 12) -> str:
    """Deterministic citation footer: the URLs the material actually came from —
    raw search hits AND the footers of climbing branch summaries (parents hold
    text, not dumps, so links must also be read back out of the prose)."""
    urls: list[str] = []

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


GATHER_TOOLS = tuple(DEFAULT_PARAMS["gather_tools"])
SINK_TOOL = DEFAULT_PARAMS["sink_tool"]

# Structured-output schemas (Ollama `format`): the reply's shape is enforced by the
# platform, so the prompts carry only semantic rules — no prose-described JSON.
CHILDREN_SCHEMA = {
    "type": "object",
    "properties": {"children": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "tool": {"type": ["string", "null"]},
            "args": {"type": "object"},
            "deps": {"type": "array", "items": {"type": ["integer", "string"]}},
            "success_predicate": {"type": "string"}},
        "required": ["intent"]}}},
    "required": ["children"],
}
JUDGE_SCHEMA = {"type": "object",
                "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
                "required": ["ok"]}


def _ground_args(tools: dict[str, Any] | None, tool: str, intent: str) -> dict[str, Any]:
    """Arg template for grounding a bare subtask onto a gather tool — declared in
    the tool's registry entry ("ground"), so non-search gatherers work too."""
    tpl: dict[str, Any] = (tools or {}).get(tool, {}).get("ground") or {"query": "{intent}"}
    return {k: v.format(intent=intent[:300]) for k, v in tpl.items()}


_NULLISH = ("", "null", "none")


def _norm_child(c: dict[str, Any], sink: str) -> dict[str, Any]:
    """Normalize one planner child: null-ish tool strings ("null", "none", missing)
    become None FIRST (they slip past every tool==X check otherwise), and a
    deps-carrying null becomes the writer — only a writer consumes other
    children's output; as an abstract node it deadlocks the run in planning."""
    c = dict(c)
    if str(c.get("tool") or "").strip().lower() in _NULLISH:
        c["tool"] = None
    if not c.get("tool") and c.get("deps"):
        c["tool"] = sink
        c.setdefault("args", {})
        if not c.get("success_predicate"):
            c["success_predicate"] = "answers the parent task from the gathered material"
    return c


def _force_synthesis(children: list[dict[str, Any]], intent: str,
                     gather: tuple = GATHER_TOOLS,
                     sink: str = SINK_TOOL, depth: int = 0,
                     tools: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Research-shaped decompositions must converge on ONE synthesis sink fed
    by every other child. A 3b model places it wrong ~half the time (or skips
    grounding entirely, answering from memory), so pin the shape deterministically
    in the same spot where we normalize model output."""
    children = [_norm_child(c, sink) for c in children]
    if depth >= 2 and children \
            and all(not c.get("tool") or c.get("tool") == sink for c in children):
        # anti-recursion guard: small models recurse 'split into pieces' forever
        # (depth-7 abstract trees). Sections/subsections live at depth 0-1; below
        # that, 'abstract' children ARE the leaves — ground them into real work.
        kids = [dict(c, tool=gather[0],
                     args=_ground_args(tools, gather[0], str(c.get("intent", ""))),
                     deps=[], success_predicate="relevant results returned")
                for c in children]
        kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                     "tool": sink, "args": {}, "deps": list(range(len(kids))),
                     "success_predicate": "answers the parent task from the gathered material"})
        return kids
    has_gather = any(c.get("tool") in gather for c in children)
    has_llm = any(c.get("tool") == sink for c in children)
    if not has_gather and not has_llm:
        if children and all(not c.get("tool") or c.get("tool") == sink
                            for c in children):
            # pure branch tree (e.g. a named-parts document plan): still must
            # converge on a sink or nothing ever rolls up to the parent
            kids = [dict(c) for c in children]
            kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                         "tool": sink, "args": {}, "deps": list(range(len(children))),
                         "success_predicate": "answers the parent task from the gathered material"})
            return kids
        return children   # pure task shape (write_file/shell) — leave it
    kids = [dict(c) for c in children]
    n = len(kids)
    for i, c in enumerate(kids):  # clamp stale indices; deps arrive as sibling indices
        c["deps"] = [d for d in c.get("deps") or []
                     if isinstance(d, int) and 0 <= d < n and d != i]
    # flatten degenerate prefix chains: gather children deps'd only on earlier
    # gather children are independent (args are static — no data flows between
    # them), so cumulative chaining just serializes execution for no reason
    gidx = {i for i, c in enumerate(kids) if c.get("tool") in gather}
    if len(gidx) >= 2 and all(all(d in gidx for d in kids[i]["deps"]) for i in gidx):
        for i in gidx:
            kids[i]["deps"] = []
    if not has_gather:
        llms = [i for i, c in enumerate(kids) if c.get("tool") == sink]
        if not llms:
            return kids
        if len(llms) == 1 and n == 1:
            # a lone writer answering from memory: ground it instead
            kids[0].update({"tool": gather[0],
                            "args": _ground_args(tools, gather[0], kids[0]["intent"]),
                            "success_predicate": "relevant hits returned"})
            kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                         "tool": sink, "args": {}, "deps": [0],
                         "success_predicate": "answers the parent task from the gathered material"})
            return kids
        sink_i = llms[-1]  # last writer always becomes the convergence sink
        if len(llms) >= 2:  # earlier writers become grounding searches
            for i in llms[:-1]:
                kids[i].update({"tool": gather[0],
                                "args": _ground_args(tools, gather[0], kids[i]["intent"]),
                                "success_predicate": "relevant hits returned"})
        for i, c in enumerate(kids):
            if i != sink_i and sink_i in c["deps"]:
                c["deps"].remove(sink_i)
        kids[sink_i]["deps"] = [i for i in range(n) if i != sink_i]
        return kids
    syn = next((i for i, c in enumerate(kids) if c.get("tool") == sink), None)
    if syn is None:
        kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                     "tool": sink, "args": {}, "deps": list(range(n)),
                     "success_predicate": "answers the parent task from the gathered material"})
        return kids
    for i, c in enumerate(kids):  # nothing downstream of synthesis may point back at it
        if i != syn and syn in c["deps"]:
            c["deps"].remove(syn)
    kids[syn]["deps"] = [i for i in range(n) if i != syn]
    return kids


def _deepen_branches(children: list[dict[str, Any]], depth: int,
                     gather: tuple = GATHER_TOOLS,
                     sink: str = SINK_TOOL) -> list[dict[str, Any]]:
    """Root-level research plans become real TREES: each gather branch turns into an
    abstract subtask (tool=null) that re-decomposes into its own searches + summary
    (root.N.M ids), syntheses aggregating upward. Gated to depth 0 — branch nodes
    decompose shallowly on their own (search leaves + one summary each)."""
    if depth != 0 or len(children) < 2:
        return children
    llms = [i for i, c in enumerate(children) if c.get("tool") == sink]
    gathers = [i for i, c in enumerate(children) if c.get("tool") in gather]
    if len(llms) > 1 or len(gathers) != len(children) - len(llms):
        return children   # shell/write_file in the mix — a task plan, leave it
    kids = [dict(c) for c in children]
    for i in gathers:
        kids[i]["tool"], kids[i]["args"], kids[i]["deps"] = None, {}, []
    return kids


class Agent:
    def _upstream(self, run_id: str, node: dict[str, Any]) -> dict[str, Any]:
        """Traverse the DAG upward (deps + parents) and gather ancestor results.
        Runs pull the run state once — every read is an API call the core can audit."""
        try:
            nodes = self.http.get(f"/runs/{run_id}").json()["nodes"]
        except Exception:
            return {}
        out: dict[str, Any] = {}
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
        self.position: dict[str, str] = {"run_id": "", "node_id": ""}

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
    def _ensure_guide(self, claim: dict[str, Any], pa: dict[str, Any]) -> None:
        """Dynamic by default: on the run's root split, the agent writes a per-task
        guide (what the task is made of, what an excellent answer looks like) and
        stores it as a visible+editable constraint — it then rides every decompose
        and synthesize prompt in the run. No static domain text anywhere."""
        cons = claim.get("constraints") or []
        if claim["node"].get("depth", 0) != 0 or \
                any(str(c).startswith("guide:") for c in cons):
            return
        try:
            task = str((claim.get("context") or {}).get("run_intent"))[:300]
            ask = resolve_prompt("guide", claim.get("prompts")).replace("{task}", task)
            txt = " ".join(self.llm.chat(ask, "", json_mode=False, model=pa["model"],
                                         temperature=pa["temperature"]).split())[:400]
            if txt:
                self.http.post(f"/runs/{claim['run_id']}/guide", json={"text": txt})
        except Exception:
            pass   # guide is a nicety; a failed generation must not stall the plan

    def decompose(self, node: dict[str, Any], inputs: dict[str, Any],
                  tools: dict[str, Any], max_depth: int,
                  constraints: list[str] | None = None,
                  context: dict[str, Any] | None = None,
                  prompts: dict[str, str] | None = None,
                  params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        pa = resolve_params(params)
        gather, sink = tuple(pa["gather_tools"]), pa["sink_tool"]
        sys_p = resolve_prompt("decompose", prompts).replace("{cap}", str(pa["child_cap"]))
        if sink != SINK_TOOL:
            sys_p += (f"\nUse tool \"{sink}\" (no args) — not \"llm\" — for the "
                      "final synthesis/answering child.")
        if tools:
            # the JSON-shape example is BUILT FROM THE RUN'S TOOLS: the model imitates
            # examples literally, so a static one dictated a fixed (wrong) domain/shape
            t0, spec0 = next(iter(tools.items()))
            ex = {"children": [
                {"intent": "subtask A", "tool": t0,
                 "args": {k: "..." for k in spec0.get("required") or []},
                 "deps": [], "success_predicate": "checkable outcome"},
                {"intent": "subtask B", "tool": t0,
                 "args": {k: "..." for k in spec0.get("required") or []},
                 "deps": [], "success_predicate": "checkable outcome"},
                {"intent": "answer the parent task from A and B", "tool": sink,
                 "deps": [0, 1], "success_predicate": "covers the parent task"}]}
            sys_p += f"\nExample using the available tools: {json.dumps(ex, separators=(',', ':'))}"
        spec = "; ".join(f"{name}(args: {', '.join(s['required'])})"
                         for name, s in tools.items())
        # bound dep results: one fat search dump must not blow the model's ctx
        brief = {k: json.dumps(v, default=str)[:600] for k, v in inputs.items()}
        sibs = "; ".join(f"{s['id']}: {str(s['intent'])[:60]}"
                         for s in (context or {}).get("siblings", [])[:8])
        known = "; ".join(f"{k['id']}[{k['chars']}ch]: {k['intent']}"
                          for k in (context or {}).get("known", [])[:12])
        prompt = (f"Parent task: {node['intent']}\nDepth {node['depth']} of {max_depth}.\n"
                  f"User constraints the plan must respect: "
                  f"{'; '.join(constraints or []) or 'none'}\n"
                  f"Overall run goal: {str((context or {}).get('run_intent'))[:160]}\n"
                  + (f"This task sits under branch: "
                     f"{str(context['parent_intent'])[:120]}\n"
                     if context and context.get("parent_intent") else "")
                  + (f"Neighboring branches (do NOT duplicate their work): {sibs}\n"
                     if sibs else "")
                  + (f"Known material (summaries that already exist in this run — "
                     f"decompose the gaps, dep on these ids if a child needs one): "
                     f"{known}\n" if known else "")
                  + f"Available tools with required args: {spec}\n"
                  f"Inputs from dependencies: {brief}\n"
                  f"args keys must match the required names exactly "
                  f"(e.g. write_file takes path and content).\n"
                  f"Hint from previous failure: {node.get('hint') or 'none'}")
        out = self.llm.chat_json(sys_p, prompt, model=pa["model"],
                                 temperature=pa["temperature"], schema=CHILDREN_SCHEMA)
        if not out or not isinstance(out.get("children"), list) or not out["children"]:
            return None
        cap = pa["child_cap"]
        kids = out["children"][:cap]
        # leave a slot for the synthesis sink if the planner forgot one
        if cap >= 2 and len(kids) == cap \
                and any(c.get("tool") in gather for c in kids) \
                and not any(c.get("tool") == sink for c in kids):
            kids = kids[:cap - 1]
        # the intent names its pieces ("with an abstract, introduction …") -> every
        # named piece must have a branch; the model merging small sections is its
        # classic failure, so coverage is topped up here, not preached in the prompt
        parts = named_parts(node["intent"])
        if parts:
            covered = " ".join(str(c.get("intent", "")).lower() for c in kids)
            for part in parts:
                if part.lower() not in covered:
                    kids.append({"intent": f"{part} (part of: {node['intent'][:80]})",
                                 "tool": None, "args": {}, "deps": [],
                                 "success_predicate": ""})
        kids = _force_synthesis(kids, node["intent"], gather, sink,
                                node.get("depth", 0), tools)
        # per-run max_results: deterministically bound any tool that takes it
        for c in kids:
            if "max_results" in tools.get(c.get("tool") or "", {}).get("required", []):
                c.setdefault("args", {})["max_results"] = pa["max_results"]
        return {"children": _deepen_branches(kids, node.get("depth", 0), gather, sink)}

    def _write_answer(self, run_id: str, node: dict[str, Any],
                      inputs: dict[str, Any], material: str,
                      constraints: list[str],
                      prompts: dict[str, str] | None = None,
                      params: dict[str, Any] | None = None,
                      tools: dict[str, Any] | None = None) -> str:
        """Write a node's text. The RUN'S HEAD writer (parent == root, several
        branch summaries in) is assembled deterministically: intro + Bottom line
        from the model (small, 3b-safe asks), body = the verified branch summaries
        verbatim as sections — nothing gets dropped in the funnel and the answer
        keeps the tree's coverage. Everyone else: a single synthesis call."""
        pa = resolve_params(params)
        sys_p = resolve_prompt("synthesize", prompts)
        if any(t in (tools or {}) for t in ("web_search", "http_fetch")):
            # grounding rule exists only when the run's tools actually produce URLs
            sys_p += ("\nCite the URL (in parentheses) of any page you use, and only "
                      "use pages whose page_text appears in the material.")
        kw: dict[str, Any] = {"json_mode": False, "model": pa["model"],
                              "temperature": pa["temperature"]}
        pre = (f"User constraints to respect: {'; '.join(constraints)}\n"
               if constraints else "")
        if node.get("parent") == "root":
            rn = self.http.get(f"/runs/{run_id}").json()["nodes"]
            sections: list[tuple[str, str]] = []
            for d, res in inputs.items():
                if isinstance(res, dict) and res.get("text") and d in rn:
                    t = re.sub(r"\n+Sources:\n(?:- .+\s*)+$", "", res["text"]).strip()
                    if t:
                        sections.append((rn[d]["intent"], t))
            if len(sections) >= 2:
                title = rn.get("root", {}).get("intent") or node["intent"]
                # document runs (named parts) keep the paper's own shape: the doc
                # STARTS with its first section (no generated preamble duplicating
                # the abstract), and if this sink IS a named part ("write the
                # conclusion…") its prose becomes that real section, not a
                # generic "Bottom line"
                named = named_parts(title)
                outro_part = next((p for p in named
                                   if p.lower() in node["intent"].lower()), "")
                intro = ""
                if not named:
                    intro = self.llm.chat(
                        sys_p,
                        pre + f"Write ONLY a 3-4 sentence introduction for a document "
                              f"titled \"{title}\" — no headings, no lists, prose only.\n"
                              f"Material from the neighboring nodes:\n{material}",
                        **kw).strip()
                outro_ask = (f"Write the {outro_part} of \"{title}\" in 4-6 grounded "
                             "sentences that compare the sections and say what the "
                             "reader should take away" if outro_part else
                             f"Write ONLY 4-6 concluding sentences comparing the sections "
                             f"and saying what the reader should take away, for a document "
                             f"titled \"{title}\"")
                outro = self.llm.chat(
                    sys_p,
                    pre + outro_ask + " — no headings, prose only.\n"
                                      f"Material from the neighboring nodes:\n{material}",
                    **kw).strip()
                outro_head = outro_part.title() if outro_part else "Bottom line"
                parts = [f"# {title}"]
                if intro:
                    parts.append(intro)
                parts += [f"## {s}\n\n{t}" for s, t in sections]
                if outro:
                    parts.append(f"## {outro_head}\n\n{outro}")
                return "\n\n".join(parts)
        return self.llm.chat(
            sys_p,
            pre + f"Task: {node['intent']}\n"
                  f"Material from the neighboring nodes:\n{material}",
            **kw).strip()

    def judge(self, node: dict[str, Any], exec_result: dict[str, Any],
              prompts: dict[str, str] | None = None,
              params: dict[str, Any] | None = None,
              tools: dict[str, Any] | None = None) -> bool:
        pa = resolve_params(params)
        pred = node.get("condition") or node.get("success_predicate", "")
        if node.get("tool") in pa["gather_tools"]:
            # deterministic, same rationale as _llm_ok: page_text makes the payload
            # big enough that a small LLM-judge silently fails good searches
            return bool(exec_result.get("results") or exec_result.get("content"))
        # a tool may declare "check": a dotted result path that must be truthy —
        # judged deterministically (CRUD tools don't need a model to spot an id)
        check = (tools or {}).get(node.get("tool") or "", {}).get("check")
        if check:
            cur: Any = exec_result
            for part in str(check).split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            return bool(cur)
        if not pred:
            return exec_result.get("rc", 0) == 0 if node.get("tool") == "shell" else True
        out = self.llm.chat_json(resolve_prompt("judge", prompts),
                                 f"Task: {node['intent']}\nPredicate: {pred}\n"
                                 f"Execution result: {exec_result}",
                                 model=pa["model"], temperature=pa["temperature"],
                                 schema=JUDGE_SCHEMA)
        return bool(out and out.get("ok") is True)   # strict: only explicit true passes

    # ---------- approval round-trips ----------
    def _wait(self, exec_id: str) -> dict[str, Any]:
        """Poll an approval/exec until decided. Result dicts carry rc>=-1;
        rejected-by-user returns rc=-2; an unanswered card returns rc=-3 after
        APPROVAL_TIMEOUT so the node requeues and this agent moves on (a card
        left overnight must not hold the agent forever)."""
        deadline = time.time() + APPROVAL_TIMEOUT
        while True:
            if time.time() > deadline:
                return {"rc": -3, "stderr": f"approval unanswered after {APPROVAL_TIMEOUT}s"}
            time.sleep(POLL)
            st = self.http.get(f"/exec/{exec_id}").json()
            if st["status"] == "done":
                return st["result"]
            if st["status"] == "rejected":
                return {"rc": -2, "stderr": "rejected by user"}

    def exec_via_core(self, run_id: str, node: dict[str, Any]) -> dict[str, Any]:
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
    def handle(self, claim: dict[str, Any]) -> None:
        run_id, node, inputs = claim["run_id"], claim["node"], claim["inputs"]
        pro = claim.get("prompts")
        pa = resolve_params(claim.get("params"))
        calls_before = self.llm.calls
        try:
            if node.get("tool"):
                if node["tool"] in ("llm", pa["sink_tool"]):  # virtual: synthesize from neighbor results
                    self._set("writing", run_id, node["id"])
                    # train of thought: each writer reads only its own level's
                    # material (direct deps); branch summaries carry what rose
                    # from below — see _material's summary-preference rule
                    material = _material(inputs)
                    cons = claim.get("constraints") or []
                    try:
                        text = self._write_answer(run_id, node, inputs, material, cons,
                                                  pro, pa, claim.get("tools"))
                    except Exception as e:
                        if _llm_down(e):
                            raise   # infra problem, not a node problem — outer handler
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
                if self.judge(node, res, pro, pa, claim.get("tools")):
                    self.report(run_id, node, "DONE", result=res, before=calls_before)
                else:
                    self.report(run_id, node, "FAILED",
                                hint=f"predicate unmet: {node.get('success_predicate')}",
                                before=calls_before)
            else:
                self._set("decomposing", run_id, node["id"])
                # planners travel upward too: full ancestor context, direct deps win
                ctx = {**self._upstream(run_id, node), **inputs}
                self._ensure_guide(claim, pa)
                sub = self.decompose(node, ctx, claim["tools"], claim["max_depth"],
                                     claim.get("constraints"), claim.get("context"),
                                     pro, pa)
                errs: Any = "LLM returned nothing usable" if sub is None else None
                for _ in range(POLICY["max_repairs"]):
                    if errs is None:
                        r = self.http.post(f"/runs/{run_id}/nodes/{node['id']}/decompose",
                                           json=sub)
                        if r.status_code == 200:
                            if r.json().get("status") == "pending_approval":
                                self._set("waiting-approval", run_id, node["id"])
                                out = self._wait(r.json()["exec_id"])
                                if out.get("rc") == -2:   # user rejected patch: re-decompose with core's hint
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
            if _llm_down(e) and node["attempts"] < POLICY["max_repairs"]:
                # runner wedged/down: RELEASE silently — no report, no spent repair.
                # The claim lease requeues the node; the backoff keeps us off the
                # runner while it recovers (restart policy or a human).
                self._set("llm-down")
                print(f"[{AGENT_ID}] {node['id']}: LLM unreachable "
                      f"({type(e).__name__}) — releasing for lease requeue")
                time.sleep(POLL * 5)
                return
            hint = (f"llm unreachable after {node['attempts']} attempts "
                    f"({type(e).__name__})" if _llm_down(e) else f"agent error: {e}")
            try:
                self.report(run_id, node, "FAILED", hint=hint, before=calls_before)
            except Exception:
                pass
        finally:
            self._set("idle")

    def report(self, run_id: str, node: dict[str, Any], state: str,
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
