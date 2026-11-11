"""Planning: abstract node → validated child subtasks (III-B Task decomposition).

Decomposition is shape-pinned deterministically after the model speaks, in this order:
  chat_json -> cap (slot for sink) -> named-parts top-up -> _force_synthesis (flatten
  gather chains, ground lone/all-llm plans, one convergence sink) -> _deepen_branches
  (root-level research plans become abstract-branch trees). The model proposes;
  these decide.

Coordination: with per-run params ``samples`` (>1) the planner samples K
independent plans, optionally debates them (`debate_rounds`), then votes on the
shaped candidates by digest — one bad sample can't sink the plan anymore.
"""
import json
import re
from typing import Any

from ..ensemble import Ensemble
from ..graph import named_parts
from ..prompts import DEFAULT_PARAMS, resolve_params, resolve_prompt
from .writing import _code_ok

GATHER_TOOLS = tuple(DEFAULT_PARAMS["gather_tools"])
SINK_TOOL = DEFAULT_PARAMS["sink_tool"]

# Structured-output schema (Ollama `format`): the reply's shape is enforced by the
# platform, so the prompt carries only semantic rules — no prose-described JSON.
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

# One failed tool call -> corrected {args, success_predicate} (repair retry).
REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"args": {"type": "object"},
                   "success_predicate": {"type": "string"}},
    "required": ["args"],
}


def _local_path(text: str) -> str:
    """A workspace path named in free text ('fix bugfix-lab/analyze.py',
    'write to data-lab/report.md'). Vocabulary-free: the trigger is path
    syntax (a slash or a document extension), not any domain word."""
    m = re.search(r"\b([\w-]+/[\w./-]*[\w-]+\.\w{1,6}"
                  r"|\w+\.(?:md|txt|csv|tsv|json|jsonl|py|log))\b", text or "")
    return m.group(1) if m else ""


def _ground_args(tools: dict[str, Any] | None, tool: str, intent: str,
                 path: str = "") -> dict[str, Any]:
    """Arg template for grounding a bare subtask onto a gather tool — declared in
    the tool's registry entry ("ground"), so non-search gatherers work too."""
    reg = (tools or {}).get(tool, {})
    tpl: dict[str, Any] = reg.get("ground") or (
        {"path": "{path}"} if "path" in (reg.get("required") or [])
        else {"query": "{intent}"})
    return {k: v.format(intent=intent[:300], path=path or intent[:300])
            for k, v in tpl.items()}


def _ground_on(tools: dict[str, Any] | None, gather: tuple, task: str,
               intent: str) -> tuple[str, dict[str, Any]]:
    """Which gather tool grounds a bare subtask. A task naming a local file is
    FILE work: grounding it onto the web gatherer (gather_tools[0]) produced
    searches like "Extract total revenue" that return nothing usable — pick the
    registry's path-taking gatherer and feed it the named file instead."""
    path = _local_path(intent) or _local_path(task)
    if path:
        for t in (*gather, "read_file"):
            if "path" in ((tools or {}).get(t, {}).get("required") or []):
                return t, _ground_args(tools, t, intent, path)
    return gather[0], _ground_args(tools, gather[0], intent)


_NULLISH = ("", "null", "none")

# a deliverable named after "to <path>" ('write a summary report to data-lab/report.md')
_OUT_RE = re.compile(r"\bto\s+([\w./-]+\.\w{1,6})\b")
# the mutation top-up's own phrasing ('write the corrected contents of X') — its
# sub-plan must end with X WRITTEN from the corrected sink text. Narrow on
# purpose: 'summarize/extract/read the contents of X' must never write over X.
_CONTENTS_RE = re.compile(r"\bwrite\s+(?:\w+\s+){0,2}contents?\s+of\s+"
                          r"([\w./-]+\.\w{1,6})", re.I)
# a bare script invocation ('python analyze_sales.py --flags', './run.sh') —
# with no path separator the file is presumed NOT to exist; the split above
# creates it first. '-c'/'-m' invocations carry code/modules, not a file.
_SCRIPT_RE = re.compile(r"^\s*(?:(?:python3?|bash|sh)\s+|\./)([\w-]+\.\w{1,6})\b")


def _named_output(intent: str) -> str:
    """The file an intent promises to produce ('… write X to path', and the
    mutation child's own 'write the corrected contents of path'). A file merely
    named as the object of 'fix/edit' or 'summarize' is not a deliverable:
    force-writing answer prose over it would destroy it."""
    m = _OUT_RE.search(intent or "") or _CONTENTS_RE.search(intent or "")
    return m.group(1) if m else ""


# in-place change verbs, closed class like the named-parts triggers; 'write'/
# 'create' deliberately absent (a file named after them is usually the OUTPUT),
# 'correct' too — as an adjective ('the correct output in x.md') it would flag
# read-only files for rewriting
_MUTATE_RE = re.compile(r"\b(fix|repair|patch|refactor|edit|update|modify)\b",
                        re.I)
_PATH_RE = re.compile(r"\b[\w-]+/[\w./-]*[\w-]+\.\w{1,6}\b|\b\w+\.(?:py|md|txt|csv|json)\b")


def _mutation_targets(intent: str) -> list[str]:
    """Files the intent says to CHANGE ('fix the bugs in bugfix-lab/analyze.py').
    A path qualifies only if a change verb sits right before it — a path merely
    named for reading ('analyze data-lab/sales.csv') is INPUT and must never be
    topped up with a rewrite (that would clobber the data)."""
    out = []
    for m in _PATH_RE.finditer(intent or ""):
        path = m.group(0)
        if path == _named_output(intent):      # deliverable, handled as a write-back
            continue
        if _MUTATE_RE.search((intent or "")[max(0, m.start() - 48):m.start()]):
            out.append(path)
    return out


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
    # the sink must sit LAST — every sibling feeds it. Shaping appends top-ups
    # after the model's own sink, and the sink's 'deps = all siblings' wiring
    # then pointed FORWARD: core rejected the whole plan (live T3: 'cannot
    # decompose' x3 -> 1-node run).
    si = [i for i, c in enumerate(children) if c.get("tool") == sink]
    if si and si[-1] != len(children) - 1:
        seq = [i for i in range(len(children)) if i not in si] + si
        remap = {old: new for new, old in enumerate(seq)}
        children = [dict(children[i],
                         deps=[remap[d] if isinstance(d, int) and d in remap else d
                               for d in (children[i].get("deps") or [])])
                    for i in seq]
    if depth >= 2 and children \
            and all(not c.get("tool") or c.get("tool") == sink for c in children):
        # anti-recursion guard: small models recurse 'split into pieces' forever
        # (depth-7 abstract trees). Sections/subsections live at depth 0-1; below
        # that, 'abstract' children ARE the leaves — ground them into real work.
        kids = [dict(c, tool=t, args=a, deps=[],
                     success_predicate="relevant results returned")
                for c, (t, a) in
                ((c, _ground_on(tools, gather, intent, str(c.get("intent", ""))))
                 for c in children)]
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
        if (depth == 0 or _named_output(intent)) and children:
            # task shape AT THE ROOT: the head node must still end holding the
            # answer (root.result.text), so one sink converges the tool results.
            # Same for a node promising a file ('write the corrected contents of
            # X'): its write-back binds the sink's text, so a sink must exist.
            kids = [dict(c) for c in children]
            kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                         "tool": sink, "args": {}, "deps": list(range(len(children))),
                         "success_predicate": "answers the parent task from the gathered material"})
            return kids
        return children   # pure task shape deeper in the tree — leave it
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
            g, a = _ground_on(tools, gather, intent, kids[0]["intent"])
            kids[0].update({"tool": g, "args": a,
                            "success_predicate": "relevant hits returned"})
            kids.append({"intent": f"write the answer from the branches: {intent[:52]}",
                         "tool": sink, "args": {}, "deps": [0],
                         "success_predicate": "answers the parent task from the gathered material"})
            return kids
        sink_i = llms[-1]  # last writer always becomes the convergence sink
        if len(llms) >= 2:  # earlier writers become grounding searches
            for i in llms[:-1]:
                g, a = _ground_on(tools, gather, intent, kids[i]["intent"])
                kids[i].update({"tool": g, "args": a,
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


def _plan_digest(plan: dict[str, Any]) -> str:
    """Compact ballot text for one shaped plan: child intents + tools + deps —
    the voter compares plan structure, not full arg payloads."""
    lines = []
    for i, c in enumerate(plan.get("children", [])):
        tool = c.get("tool") or "split-again"
        deps = ",".join(map(str, c.get("deps") or [])) or "-"
        lines.append(f"{i}. {c.get('intent', '')} [{tool}] deps:{deps}")
    return "\n".join(lines)


class PlannerMixin:
    """LLM-assisted decomposition. Requires the BaseAgent plumbing (llm/http)."""

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

    def _repair_args(self, node: dict[str, Any], prompts: dict[str, str] | None,
                     pa: dict[str, Any], tools: dict[str, Any] | None) -> dict[str, Any]:
        """Retry with REWRITTEN args + predicate. Re-executing identical args can
        never repair a failed tool call (measured: three byte-identical attempts
        then escalation), so the failure hint drives one regeneration. No-op on
        first attempts and arg-less tools."""
        if not node.get("hint") or node.get("attempts", 0) < 1:
            return node
        tool = node.get("tool") or ""
        spec = (tools or {}).get(tool) or {}
        if not spec.get("required"):
            return node
        ask = (f"Task: {node['intent']}\nTool: {tool} (required args: "
               f"{', '.join(spec['required'])})\nPrevious args: "
               f"{json.dumps(node.get('args', {}), default=str)}\n"
               f"Why it failed: {node.get('hint')}")
        try:
            out = self.llm.chat_json(resolve_prompt("repair", prompts), ask,
                                     model=pa["model"], temperature=pa["temperature"],
                                     schema=REPAIR_SCHEMA)
        except Exception:
            return node   # repair is a nicety: replay the args rather than stall
        if not out or not isinstance(out.get("args"), dict):
            return node
        fixed = {k: v for k, v in out["args"].items() if k in spec["required"]}
        # structural $from bindings are not repair material: the model mangles
        # them (live T2: bind id 'root/analyze.py') and the exec is then rejected
        fixed = {k: v for k, v in fixed.items()
                 if not (isinstance((node.get("args") or {}).get(k), dict)
                         and "$from" in node["args"][k])}
        if not fixed:
            return node
        node = {**node, "args": {**node.get("args", {}), **fixed}}
        pred = out.get("success_predicate")
        if isinstance(pred, str) and pred.strip():
            node["success_predicate"] = pred.strip()[:300]
        return node

    def _decompose_prompt(self, node: dict[str, Any], inputs: dict[str, Any],
                          tools: dict[str, Any], max_depth: int,
                          constraints: list[str] | None,
                          context: dict[str, Any] | None,
                          prompts: dict[str, str] | None,
                          pa: dict[str, Any]) -> tuple[str, str]:
        """Build (system, user) for one decomposition ask. The JSON-shape example
        is BUILT FROM THE RUN'S TOOLS: the model imitates examples literally, so a
        static one dictated a fixed (wrong) domain/shape."""
        sink = pa["sink_tool"]
        sys_p = resolve_prompt("decompose", prompts).replace("{cap}", str(pa["child_cap"]))
        if sink != SINK_TOOL:
            sys_p += (f"\nUse tool \"{sink}\" (no args) — not \"llm\" — for the "
                      "final synthesis/answering child.")
        if tools:
            specs = list(tools.items())
            # two DIFFERENT tools in the example — the model imitates examples
            # literally, and a single-tool example dictated shell-for-everything
            # plans (measured: file reads planned as fragile greps)
            ex = {"children": [
                {"intent": "subtask A", "tool": specs[0][0],
                 "args": {k: "..." for k in specs[0][1].get("required") or []},
                 "deps": [], "success_predicate": "checkable outcome"},
                {"intent": "subtask B", "tool": specs[1 % len(specs)][0],
                 "args": {k: "..."
                          for k in specs[1 % len(specs)][1].get("required") or []},
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
        return sys_p, prompt

    def _shape_children(self, out: dict[str, Any] | None, node: dict[str, Any],
                        tools: dict[str, Any], pa: dict[str, Any]) -> dict[str, Any] | None:
        """Deterministic post-processing of one raw model plan: cap, sink slot,
        named-parts top-up, synthesis pinning, per-run max_results, branch deepen."""
        if not out or not isinstance(out.get("children"), list) or not out["children"]:
            return None
        gather, sink = tuple(pa["gather_tools"]), pa["sink_tool"]
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
        # a write whose content is the task ECHO, an unfilled TEMPLATE, or (for
        # .py) not even valid Python is not content — live T2 wrote "updated code
        # in bugfix-lab/analyze.py" over source (pytest then rc=2), live T3 wrote
        # '{{total_revenue}}' into the deliverable. Drop them: the top-ups below
        # then plan real writes.
        def _real_write(c: dict[str, Any]) -> bool:
            content = (c.get("args") or {}).get("content")
            path = str((c.get("args") or {}).get("path") or "")
            if not isinstance(content, str):
                return True                       # $from bindings are real content
            body, own = content.strip(), str(c.get("intent", "")).strip()
            if not body or body == own or re.search(r"\{\{.+?\}\}|\$\{.+?\}", body):
                return False
            return not path.endswith(".py") or _code_ok(body)
        kept = [c for c in kids if c.get("tool") != "write_file" or _real_write(c)]
        kids = kept          # empty is fine: the top-ups below plan real work,
                             # and an all-garbage plan should be re-planned
        # a file the intent says to CHANGE needs a child that changes it — plans
        # 'find and fix' without ever writing (measured live: the bug survived).
        # The child stays abstract: the sub-plan writes the real corrected content;
        # a canned write here would clobber the file with answer prose.
        wrote = {str((c.get("args") or {}).get("path", "")).lower()
                 for c in kids if c.get("tool") == "write_file"}
        for path in _mutation_targets(node["intent"]):
            if path.lower() not in wrote:
                kids.append({"intent": f"write the corrected contents of {path}",
                             "tool": None, "args": {}, "deps": [],
                             "success_predicate": f"{path} is corrected"})
        # a task naming a local file needs someone who reads it WHOLE: field
        # extractions ('grep X | awk') leave the sink with a guessed field and
        # nothing else (live T5: the answer lacked all three facts; live T3:
        # 'no material was gathered'). Grounded on the registry's path-taking
        # gatherer, like any bare file subtask — never a write of the file.
        path = _local_path(node["intent"])
        deliverable = _OUT_RE.search(node["intent"] or "")   # 'to <path>' form only
        if path and path != (deliverable.group(1) if deliverable else ""):
            reads = {str((c.get("args") or {}).get("path", "")).lower()
                     for c in kids if c.get("tool") != "write_file"}
            t, a = _ground_on(tools, gather, node["intent"], node["intent"])
            if path.lower() not in reads and t != sink \
                    and "path" in ((tools or {}).get(t, {}).get("required") or []):
                kids.append({"intent": f"read {path} for context",
                             "tool": t, "args": a, "deps": [],
                             "success_predicate": f"{path} is read"})
        # a shell that runs a script nobody creates never runs: the model invents
        # 'python analyze_sales.py' and regenerates it 3x even with 'can't open
        # file' in hand (measured live). The step really wants two things — the
        # script CREATED (the model writes code far better than it does
        # arithmetic) and then run — so split it: the child becomes the script
        # write (abstract: the sub-plan supplies real code) and the run is
        # appended after it.
        made: dict[str, int] = {}
        for i, c in enumerate(list(kids)):
            cmd = str((c.get("args") or {}).get("cmd") or "")
            m = _SCRIPT_RE.match(cmd) if c.get("tool") == "shell" else None
            script = m.group(1) if m else ""
            if not script or _CONTENTS_RE.search(node["intent"] or ""):
                # inside a file-producing task a nested split just recurses
                # (measured: 35 nodes, depth 6) — this level writes ONE file
                continue
            made_i = made.get(script)
            creates = any(str((w.get("args") or {}).get("path") or "") == script
                          for w in kids if w.get("tool") == "write_file")
            if made_i is not None:
                c["deps"] = list(c.get("deps") or []) + [made_i]
            elif not creates:
                made[script] = i
                purpose, run = str(c.get("intent", ""))[:60], dict(c)
                c.clear()
                c.update({"intent": f"write the corrected contents of {script} "
                                    f"(part of: {purpose})",
                          "tool": None, "args": {},
                          "deps": list(run.get("deps") or []),
                          "success_predicate": f"{script} is written"})
                run["deps"] = list(run.get("deps") or []) + [i]
                kids.append(run)
        kids = _force_synthesis(kids, node["intent"], gather, sink,
                                node.get("depth", 0), tools)
        # a task promising corrected FILE contents must have its sink PRODUCE
        # those contents: framed as analysis ('locate the failing test') the sink
        # answered prose and the write-back shipped it over the source (live T2).
        # Skipped when a 'to <path>' deliverable exists — there the sink's text
        # is prose meant for the report, not for the file named earlier.
        m = _CONTENTS_RE.search(node["intent"] or "")
        if m and not _OUT_RE.search(node["intent"] or ""):
            tail = (node["intent"] or "")[m.end():].strip()   # '(part of: …)' purpose
            for c in kids:
                if c.get("tool") == sink:
                    c["intent"] = (f"write the corrected contents of {m.group(1)} "
                                   "(output the complete file)"
                                   + (f" — {tail}" if tail else ""))
        # an intent that names its output ('write a report to data-lab/report.md')
        # must end with that file WRITTEN — plans compute the answer and silently
        # drop the deliverable (measured live), so bind a write of the sink's text
        out_path = _named_output(node["intent"])
        writes_out = any(c.get("tool") == "write_file"
                         and str((c.get("args") or {}).get("path", "")) == out_path
                         for c in kids)
        sink_i = max((i for i, c in enumerate(kids) if c.get("tool") == sink),
                     key=lambda i: len(kids[i].get("deps") or []), default=None)
        if out_path and not writes_out and sink_i is not None:
            kids.append({"intent": f"write the answer to {out_path}",
                         "tool": "write_file", "deps": [sink_i],
                         "args": {"path": out_path,
                                  "content": {"$from": f"{node['id']}.{sink_i + 1}",
                                              "path": "text"}},
                         "success_predicate": f"the file {out_path} is written"})
        # per-run max_results: deterministically bound any tool that takes it
        for c in kids:
            if "max_results" in tools.get(c.get("tool") or "", {}).get("required", []):
                c.setdefault("args", {})["max_results"] = pa["max_results"]
        return {"children": _deepen_branches(kids, node.get("depth", 0), gather, sink)}

    def decompose(self, node: dict[str, Any], inputs: dict[str, Any],
                  tools: dict[str, Any], max_depth: int,
                  constraints: list[str] | None = None,
                  context: dict[str, Any] | None = None,
                  prompts: dict[str, str] | None = None,
                  params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Decompose one node. With the default params this is one structured
        chat_json call; with samples>1 / debate_rounds>0 the ensemble samples K
        plans, debates them, and votes one winner — every candidate is still
        shape-pinned by _shape_children before it can enter the ballot."""
        pa = resolve_params(params)
        sys_p, prompt = self._decompose_prompt(node, inputs, tools, max_depth,
                                               constraints, context, prompts, pa)
        if pa["samples"] <= 1 and pa["debate_rounds"] <= 0:
            out = self.llm.chat_json(sys_p, prompt, model=pa["model"],
                                     temperature=pa["temperature"], schema=CHILDREN_SCHEMA)
            return self._shape_children(out, node, tools, pa)
        ens = Ensemble(self.llm, sys_p, model=pa["model"],
                       temperature=pa["temperature"], prompts=prompts)
        raws = ens.sample(prompt, pa["samples"], CHILDREN_SCHEMA)
        if not raws:
            return None
        raws = ens.debate(prompt, raws, pa["debate_rounds"], CHILDREN_SCHEMA)
        plans = [p for p in (self._shape_children(r, node, tools, pa) for r in raws) if p]
        if not plans:
            return None
        if len(plans) == 1:
            return plans[0]
        return plans[ens.vote(prompt, plans, [_plan_digest(p) for p in plans])]
