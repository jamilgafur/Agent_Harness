"""Agent worker package: claim → execute (approval-gated, via core) → judge →
report, or decompose abstract nodes via LLM with a validate-and-resubmit repair
loop.

Structure:
  base.py     BaseAgent    — core transport, heartbeat, approvals, reports
  planning.py PlannerMixin — decomposition + deterministic plan shaping
  writing.py  WriterMixin  — grounded synthesis for virtual-writer nodes
  judging.py  JudgeMixin   — execution verdicts (panel vote when samples>1)
  worker.py   Agent        — the mixins composed + the claim loop

Coordination knobs (`samples`, `debate_rounds` per-run params) live in
harness/ensemble.py. Communicates per paper III-B: Task (claim), Result
(report), Redirect (handoff), Heartbeat (phase + position).

Debug a stuck worker live: `docker compose kill -s USR1 agent` -> traceback in logs.
"""
from .base import (
    AGENT_ID,
    APPROVAL_TIMEOUT,
    CFG,
    CORE,
    HEARTBEAT,
    POLICY,
    POLL,
    BaseAgent,
    _llm_down,
)
from .judging import JUDGE_SCHEMA, JudgeMixin
from .planning import (
    CHILDREN_SCHEMA,
    GATHER_TOOLS,
    SINK_TOOL,
    PlannerMixin,
    _deepen_branches,
    _force_synthesis,
    _ground_args,
    _norm_child,
)
from .worker import Agent
from .writing import WriterMixin, _llm_ok, _material, _sources

__all__ = [
    "AGENT_ID", "APPROVAL_TIMEOUT", "Agent", "BaseAgent", "CFG",
    "CHILDREN_SCHEMA", "CORE", "GATHER_TOOLS", "HEARTBEAT", "JUDGE_SCHEMA",
    "JudgeMixin", "POLICY", "POLL", "PlannerMixin", "SINK_TOOL", "WriterMixin",
    "_deepen_branches", "_force_synthesis", "_ground_args", "_llm_down",
    "_llm_ok", "_material", "_norm_child", "_sources",
]
