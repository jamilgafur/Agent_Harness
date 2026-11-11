"""The composed Agent: dispatch one claimed node, loop the claim queue.

Inheritance structure:

    BaseAgent    — core transport, heartbeat, approvals, reports (base.py)
    PlannerMixin — LLM decomposition + deterministic shaping (planning.py)
    WriterMixin  — virtual-writer synthesis for llm/sink nodes (writing.py)
    JudgeMixin   — execution verdicts, deterministic or panel-voted (judging.py)
    Agent        — all of the above + handle/run below (this file)

Runnable with zero LLM calls when nodes carry auto-approved tools and no
success_predicate (template DAGs) — that path is the offline acceptance test.
"""
import threading
import time
from typing import Any

from ..prompts import resolve_params
from . import base  # module, not names: tests patch harness.agent.base.POLL & co.
from .base import AGENT_ID, POLICY, BaseAgent, _llm_down
from .judging import JudgeMixin
from .planning import PlannerMixin
from .writing import WriterMixin, _llm_ok, _material, _sources, _write_payload


class Agent(PlannerMixin, WriterMixin, JudgeMixin, BaseAgent):
    """One worker: claim → execute (via core) or decompose → judge → report."""

    def handle(self, claim: dict[str, Any]) -> None:
        """Dispatch one claimed node to its handler: virtual writer (llm/sink),
        tool execution (approval-gated via core), or abstract decomposition —
        with the validate-and-resubmit repair loop around bad plans."""
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
                    if node.get("hint"):
                        # repair context: WHY the previous attempt was rejected —
                        # an oblivious retry just repeats the same answer again
                        material = f"Previous attempt rejected: {node['hint']}\n\n{material}"
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
                if node.get("hint"):
                    node = self._repair_args(node, pro, pa, claim.get("tools"))
                if node.get("tool") == "write_file":
                    node = _write_payload(node, inputs)   # land CODE in code files
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
                if self.judge(node, res, pro, pa, claim.get("tools"), inputs):
                    self.report(run_id, node, "DONE", result=res, before=calls_before)
                else:
                    # the REAL failure reason drives repair: without stderr the
                    # model regenerated the same invented script 3x (live T3:
                    # 'can't open file analyze_sales.py' never reached the repair)
                    hint = f"predicate unmet: {node.get('success_predicate')}"
                    if res.get("rc", 0):
                        tail = str(res.get("stderr") or res.get("stdout") or "")
                        hint += f" | rc={res['rc']} {tail[-200:]}"
                    self.report(run_id, node, "FAILED", hint=hint,
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
                last = errs or ""          # most recent FAILURE reason, for the hint
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
                        errs = last = r.json().get("detail", r.text)
                    sub = self.decompose(
                        {**node, "hint": f"previous decomposition rejected: {errs}"},
                        inputs, claim["tools"], claim["max_depth"])
                    if sub:
                        errs = None        # ready to POST again; keep `last` for the hint
                    else:
                        errs = last = "LLM returned nothing usable"
                self.report(run_id, node, "FAILED",
                            hint=f"cannot decompose: {last}", before=calls_before)
        except Exception as e:  # network blip, LLM down, ...
            if _llm_down(e) and node["attempts"] < POLICY["max_repairs"]:
                # runner wedged/down: RELEASE silently — no report, no spent repair.
                # The claim lease requeues the node; the backoff keeps us off the
                # runner while it recovers (restart policy or a human).
                self._set("llm-down")
                print(f"[{AGENT_ID}] {node['id']}: LLM unreachable "
                      f"({type(e).__name__}) — releasing for lease requeue")
                time.sleep(base.POLL * 5)
                return
            hint = (f"llm unreachable after {node['attempts']} attempts "
                    f"({type(e).__name__})" if _llm_down(e) else f"agent error: {e}")
            try:
                self.report(run_id, node, "FAILED", hint=hint, before=calls_before)
            except Exception:
                pass
        finally:
            self._set("idle")

    def run(self) -> None:
        """Claim loop: poll core for ready work forever; back off on errors."""
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print(f"[{AGENT_ID}] polling {base.CORE} "
              f"(ollama={base.CFG['ollama_url']} model={base.CFG['model']})")
        while True:
            try:
                claim = self.http.post("/claim", json={"agent_id": AGENT_ID}).json()
                if claim.get("node"):
                    print(f"[{AGENT_ID}] claimed {claim['node']['id']}: {claim['node']['intent'][:80]}")
                    self.handle(claim)
                else:
                    self._set("idle")
                    time.sleep(base.POLL)
            except Exception as e:
                print(f"[{AGENT_ID}] poll error: {e}")
                time.sleep(base.POLL * 3)
