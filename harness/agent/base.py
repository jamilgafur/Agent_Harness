"""Base agent: identity, core transport, heartbeat, claim/result plumbing.

Every worker-side concern that touches the core HTTP API (but never the LLM)
lives here; LLM-facing behavior is layered on by the mixins in planning.py,
writing.py and judging.py, and composed into `Agent` in worker.py.

Loop tunables (POLL / APPROVAL_TIMEOUT / HEARTBEAT) live in this module and are
read as module globals at call time, so tests patch them at
``harness.agent.base.<NAME>``.
"""
import faulthandler
import os
import signal
import socket
import time
from typing import Any

import httpx

from ..config import load
from ..llm import LLM

# debug: `docker compose kill -s USR1 agent` dumps a live traceback to the logs
signal.signal(signal.SIGUSR1, lambda *_: faulthandler.dump_traceback())

CFG = load()
POLICY: dict[str, Any] = CFG["policy"]
AGENT_ID = os.environ.get("AGENT_ID") or f"agent-{socket.gethostname()[-4:]}"  # unique per scaled container
POLL = float(os.environ.get("POLL_INTERVAL", "2"))
APPROVAL_TIMEOUT = float(os.environ.get("APPROVAL_TIMEOUT", "1800"))  # unanswered card gives up
HEARTBEAT = float(os.environ.get("HEARTBEAT_SECONDS", POLICY.get("heartbeat_seconds", 5)))
CORE = CFG["core_url"]


def _llm_down(exc: BaseException) -> bool:
    """LLM runner unreachable/mid-crash: timeouts, disconnects, server 5xx.
    Distinguishes 'infra paused' from 'this node is bad' — the repair loop must
    not spend a node's attempts waiting out a wedged ollama (300s timeouts)."""
    if isinstance(exc, (httpx.ReadTimeout, httpx.RemoteProtocolError,
                      httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    return type(exc).__name__ == "ResponseError" and getattr(exc, "status_code", 0) >= 500


class BaseAgent:
    """Transport + liveness for one worker: heartbeat, claim-side reads of the
    run state, approval/exec polling, and the terminal result report.

    Subclass mixins add the LLM-facing duties (planning, writing, judging) and
    worker.py's Agent adds the claim loop and node dispatch.
    """

    def __init__(self) -> None:
        self.llm = LLM(CFG["ollama_url"], CFG["model"])
        self.http = httpx.Client(base_url=CORE, timeout=30)
        self.phase = "idle"
        self.position: dict[str, str] = {"run_id": "", "node_id": ""}

    # ---------- heartbeat (III-B Heartbeat) ----------
    def _heartbeat_loop(self) -> None:
        """POST phase + position forever; keeps the claim lease alive during
        long LLM calls. Failures are silent — liveness is best-effort."""
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

    # ---------- run-state reads ----------
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
        """Run a tool through the core's exec mediator (sandbox + approvals)."""
        r = self.http.post(f"/runs/{run_id}/exec",
                           json={"node_id": node["id"], "tool": node["tool"],
                                 "args": node.get("args", {})})
        data = r.json()
        if r.status_code != 200:
            return {"rc": -1, "stderr": f"exec rejected: {data}"}
        if data["status"] == "done":
            return data["result"]
        return self._wait(data["exec_id"])

    # ---------- terminal report ----------
    def report(self, run_id: str, node: dict[str, Any], state: str,
               result: Any = None, hint: str = "", before: int = 0) -> None:
        """DONE/FAILED with the node's LLM-call delta (budget governor input).
        A 409 means the lease expired underneath us — the report is stale."""
        r = self.http.post(f"/runs/{run_id}/nodes/{node['id']}/result",
                           json={"state": state, "result": result, "hint": hint,
                                 "llm_calls": self.llm.calls - before, "agent_id": AGENT_ID})
        if r.status_code == 409:
            print(f"[{AGENT_ID}] stale report for {node['id']} ignored (lease expired)")
