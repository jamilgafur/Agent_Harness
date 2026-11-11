"""Judging: did a tool execution satisfy its success predicate?

Gather tools and tools declaring a `check` path are judged deterministically —
a small model silently fails on huge search dumps and is unneeded for CRUD ids.
Everything else goes to the LLM judge; with per-run param ``samples`` (>1) the
verdict is a strict-majority vote of a K-judge panel (one bad verdict can't
fail a good execution or pass a bad one on its own).
"""
from typing import Any

from ..prompts import resolve_params, resolve_prompt
from .writing import _code_ok

JUDGE_SCHEMA = {"type": "object",
                "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
                "required": ["ok"]}


class JudgeMixin:
    """Execution verdicts. Requires the BaseAgent plumbing (llm)."""

    def judge(self, node: dict[str, Any], exec_result: dict[str, Any],
              prompts: dict[str, str] | None = None,
              params: dict[str, Any] | None = None,
              tools: dict[str, Any] | None = None,
              inputs: dict[str, Any] | None = None) -> bool:
        """True only if the execution satisfied the predicate. Default: one LLM
        ballot, explicit true required; samples>1: strict majority of K ballots."""
        pa = resolve_params(params)
        if node.get("tool") in pa["gather_tools"]:
            # deterministic, same rationale as _llm_ok: page_text makes the payload
            # big enough that a small LLM-judge silently fails good searches
            return bool(exec_result.get("results") or exec_result.get("content"))
        if node.get("tool") == "write_file":
            # broken code must never pass as 'written': live T2 shipped a status
            # line over source and pytest died on it (rc=2) while the check below
            # saw bytes and said DONE
            content = (node.get("args") or {}).get("content")
            if isinstance(content, dict) and "$from" in content:
                src = (inputs or {}).get(content["$from"])
                p = str(content.get("path") or "")
                content = src.get(p) if isinstance(src, dict) and p else src
            path = str((node.get("args") or {}).get("path") or "")
            if isinstance(content, str) and path.endswith(".py") \
                    and not _code_ok(content):
                return False
        # a tool may declare "check": a dotted result path that must be truthy —
        # judged deterministically (CRUD tools don't need a model to spot an id)
        check = (tools or {}).get(node.get("tool") or "", {}).get("check")
        if check:
            cur: Any = exec_result
            for part in str(check).split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            return bool(cur)
        if node.get("tool") == "shell":
            # deterministic, same rationale as gather tools — a 3b judge flunks
            # large stdout dumps: measured (decomp benchmark), whole runs
            # escalated over CORRECT greps judged "predicate unmet". rc is the
            # only honest signal; content predicates stay as guidance for the
            # planner, like writers.
            return exec_result.get("rc", 0) == 0
        pred = node.get("condition") or node.get("success_predicate", "")
        if not pred:
            return True
        ask = (f"Task: {node['intent']}\nPredicate: {pred}\n"
               f"Execution result: {exec_result}")
        sys_p = resolve_prompt("judge", prompts)
        n = max(1, pa["samples"])   # samples>1 -> K-judge panel
        oks = 0
        for _ in range(n):
            out = self.llm.chat_json(sys_p, ask, model=pa["model"],
                                     temperature=pa["temperature"], schema=JUDGE_SCHEMA)
            oks += bool(out and out.get("ok") is True)
        return oks * 2 > n  # strict majority; unusable ballots count as not-ok
