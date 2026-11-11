"""Sampling, debate and voting: coordination over several attempts at the SAME job.

The agent mesh already parallelizes different subtasks across workers; this
module coordinates multiple independent attempts at one subtask inside a single
claim:

1. **sample** — K independent proposals; the first keeps the run temperature,
   the rest run hotter for diversity (self-consistency without extra infra).
2. **debate** — every proposal is revised while seeing its rivals (multi-agent
   debate); a failed revision keeps the previous proposal.
3. **vote** — one judge ballot picks the winner; an unusable ballot falls back
   to candidate 0, so the protocol can never do worse than a single sample.

One local endpoint plays every agent and calls run sequentially (the runner has
one slot) — cost is additive per node, so this stays off by default and is
enabled per run with params ``samples`` (>1) / ``debate_rounds`` (>0). With
samples=1 and debate_rounds=0 every path is call-for-call identical to the
pre-ensemble code.
"""
import json
from typing import TYPE_CHECKING, Any

from .prompts import resolve_prompt

if TYPE_CHECKING:
    from .llm import LLM

VOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}, "why": {"type": "string"}},
    "required": ["choice"],
}

_HEAT = 0.3  # extra temperature for non-first samples: diversity without drift


def _show(candidate: Any) -> str:
    """Candidate as prompt text: plans as compact JSON, prose as-is."""
    return json.dumps(candidate, default=str) if isinstance(candidate, (dict, list)) \
        else str(candidate)


class Ensemble:
    """Runs the sample → debate → vote protocol for one ask over one LLM.

    ``system`` is the task's own prompt (decompose/synthesize) used for propose
    and revise calls; debate/vote persona text resolves per run through
    prompts.py. ``schema`` selects structured (JSON) answers — plans are
    debated and voted by the same protocol, just parsed with repair.
    """

    def __init__(self, llm: "LLM", system: str, *, model: str = "",
                 temperature: float = 0.2, prompts: dict[str, str] | None = None) -> None:
        self.llm = llm
        self.system = system
        self.model = model
        self.temperature = temperature
        self.prompts = prompts  # per-run prompt overrides (PUT /runs/{id}/config)

    # ---------- 1. sampling ----------
    def sample(self, ask: str, n: int, schema: dict[str, Any] | None = None) -> list[Any]:
        """K independent proposals; unusable ones (None / empty) are dropped."""
        out: list[Any] = []
        for i in range(max(1, n)):
            t = self.temperature if i == 0 else min(1.0, self.temperature + _HEAT)
            if schema is not None:
                c = self.llm.chat_json(self.system, ask, model=self.model,
                                       temperature=t, schema=schema)
            else:
                c = self.llm.chat(self.system, ask, json_mode=False,
                                  model=self.model, temperature=t).strip()
            if c:
                out.append(c)
        return out

    # ---------- 2. debate ----------
    def debate(self, ask: str, candidates: list[Any], rounds: int,
               schema: dict[str, Any] | None = None) -> list[Any]:
        """R rounds of rival-aware revision; failed revision keeps the old answer."""
        current = list(candidates)
        sys_p = self.system + "\n" + resolve_prompt("debate", self.prompts)
        for _ in range(rounds):
            if len(current) < 2:
                break  # one surviving answer has no rivals to debate
            nxt: list[Any] = []
            for i, mine in enumerate(current):
                rivals = "\n\n".join(f"Rival answer {j + 1}:\n{_show(c)}"
                                     for j, c in enumerate(current) if j != i)
                prompt = (f"Task:\n{ask}\n\nYour answer:\n{_show(mine)}\n\n{rivals}")
                revised = self.llm.chat_json(sys_p, prompt, model=self.model,
                                             temperature=self.temperature,
                                             schema=schema) if schema is not None \
                    else self.llm.chat(sys_p, prompt, json_mode=False,
                                       model=self.model, temperature=self.temperature).strip()
                nxt.append(revised if revised else mine)
            current = nxt
        return current

    # ---------- 3. voting ----------
    def vote(self, ask: str, candidates: list[Any],
             digests: list[str] | None = None) -> int:
        """One judge ballot → winner index. Falls back to 0 on an unusable ballot."""
        if len(candidates) < 2:
            return 0
        shown = digests if digests is not None else [_show(c) for c in candidates]
        ballot = "\n\n".join(f"Candidate {i}:\n{d}" for i, d in enumerate(shown))
        out = self.llm.chat_json(resolve_prompt("vote", self.prompts),
                                 f"Task:\n{ask}\n\n{ballot}",
                                 model=self.model, temperature=self.temperature,
                                 schema=VOTE_SCHEMA)
        try:
            choice = int((out or {}).get("choice", -1))
        except (TypeError, ValueError):
            choice = -1
        return choice if 0 <= choice < len(candidates) else 0

    # ---------- full protocol ----------
    def run(self, ask: str, n: int, rounds: int,
            schema: dict[str, Any] | None = None) -> Any | None:
        """sample → debate → vote. None mirrors 'LLM returned nothing usable' in
        the callers' existing failure paths."""
        candidates = self.sample(ask, n, schema)
        if not candidates:
            return None
        candidates = self.debate(ask, candidates, rounds, schema)
        return candidates[self.vote(ask, candidates)]
