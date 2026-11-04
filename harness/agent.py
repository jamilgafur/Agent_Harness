"""Agent worker: claim → execute (approval-gated, via core) → judge → report,
or decompose abstract nodes via LLM with a validate-and-resubmit repair loop.

Communicates per paper III-B: Task (claim), Result (report), Redirect (handoff),
Heartbeat (phase + position, for the UI agent layer and the claim lease).

Runnable with zero LLM calls when nodes carry auto-approved tools and no
success_predicate (template DAGs) — that path is the offline acceptance test.
"""
import faulthandler
import os
import signal
import threading
import time
from typing import Any, Dict, Optional

import httpx

from .config import load
from .llm import LLM

# debug: `docker compose kill -s USR1 agent` dumps a live traceback to the logs
signal.signal(signal.SIGUSR1, lambda *_: faulthandler.dump_traceback())

CFG = load()
POLICY: Dict[str, Any] = CFG["policy"]
AGENT_ID = os.environ.get("AGENT_ID", "agent-1")
POLL = float(os.environ.get("POLL_INTERVAL", "2"))
HEARTBEAT = float(os.environ.get("HEARTBEAT_SECONDS", POLICY.get("heartbeat_seconds", 5)))
CORE = CFG["core_url"]

DECOMPOSE_SYS = """You decompose one task into 2-4 smaller subtasks for a DAG.
Reply with JSON only:
{"children":[{"intent":"...","tool":"shell|read_file|write_file|list_dir|null","args":{...},
"deps":[0],"success_predicate":"..."}]}
Rules: deps are indices of earlier siblings in YOUR list (not of the parent).
tool=null means the subtask needs further decomposition. Ground work in tools when possible.
args must contain exactly the required keys for the chosen tool.
Every child with a tool needs a success_predicate (a checkable outcome)."""

JUDGE_SYS = """You judge whether a tool execution satisfied the success predicate.
Reply with JSON only: {"ok": true|false, "why": "short reason"}"""


class Agent:
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
                  tools: Dict[str, Any], max_depth: int) -> Optional[Dict[str, Any]]:
        spec = "; ".join(f"{name}(args: {', '.join(s['required'])})"
                         for name, s in tools.items())
        prompt = (f"Parent task: {node['intent']}\nDepth {node['depth']} of {max_depth}.\n"
                  f"Available tools with required args: {spec}\n"
                  f"Inputs from dependencies: {inputs}\n"
                  f"args keys must match the required names exactly (e.g. write_file takes path and content).\n"
                  f"Hint from previous failure: {node.get('hint') or 'none'}")
        out = self.llm.chat_json(DECOMPOSE_SYS, prompt)
        if not out or not isinstance(out.get("children"), list) or not out["children"]:
            return None
        return {"children": out["children"][:4]}

    def judge(self, node: Dict[str, Any], exec_result: Dict[str, Any]) -> bool:
        pred = node.get("success_predicate", "")
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
                sub = self.decompose(node, inputs, claim["tools"], claim["max_depth"])
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
