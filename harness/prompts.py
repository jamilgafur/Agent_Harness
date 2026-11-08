"""Prompts and LLM/planner parameters: the single source of truth, overridable
per run via PUT /runs/{id}/config. Core injects overrides into the claim payload;
agents resolve them through here. Missing or invalid values fall back to defaults.
"""
from typing import Any

DEFAULT_PROMPTS: dict[str, str] = {
    # The reply's JSON shape is enforced by the schema passed to Ollama's structured
    # output (`format`) in agent.py — the prompt carries only the semantic rules.
    "decompose": """You split one task into at most {cap} smaller, non-overlapping subtasks.
- Children must be strict sub-parts of the parent, each simpler than it.
- tool=null: only when the subtask genuinely needs another split; a subtask done in
  one pass MUST be a tool child, with args matching the tool's required keys exactly
  and a checkable success_predicate. (An arg can also be
  {"$from": "<id>", "path": "a.b.c"} to pull a value out of that dep's result.)
- deps: indices of earlier siblings in YOUR list, or an id listed under Neighbors/Known —
  only for a real information need (that node's output becomes the child's material).
- "llm" is a virtual writer: it receives its deps' results as material and writes the prose.
  Gather with the available tools first; never answer from memory.
Keep the parent task's qualifiers in every tool call you phrase. If Neighbors or Known
material are listed, complement what exists — never duplicate it.""",
    "synthesize": """You write grounded, specific answers using ONLY the material provided
(labeled [from <node>]). Match the form the task itself asks for.
KEEP THE CONCRETE SPECIFICS — numbers, names, mechanisms, dates. Specifics are what
make an answer useful; vague paraphrases ("it improves efficiency") are failure.
If the material is thin or missing, say what is missing at the end instead of inventing.""",
    "guide": """In at most four short lines, note what THIS task is made of (its natural
pieces, what counts as solid evidence for it) and what an excellent final answer
looks like. Be specific to this exact task; no generic advice.
Task: {task}""",
    "judge": """You judge whether a tool execution satisfied the success predicate.
Reply with JSON only: {"ok": true|false, "why": "short reason"}""",
}

DEFAULT_PARAMS: dict[str, Any] = {
    "model": "",  # "" -> harness default (env MODEL)
    "temperature": 0.2,
    "child_cap": 4,
    "gather_tools": ["web_search", "http_fetch"],
    "sink_tool": "llm",
    "max_results": 8,
}

PROMPT_KEYS = tuple(DEFAULT_PROMPTS)

# key -> (types,) or (types, lo, hi)
PARAM_SPEC: dict[str, tuple] = {
    "model": (str,),
    "temperature": ((int, float), 0.0, 1.0),
    "child_cap": (int, 1, 6),
    "gather_tools": (list,),
    "sink_tool": (str,),
    "max_results": (int, 3, 10),
}


def resolve_prompt(kind: str, overrides: dict[str, str] | None = None) -> str:
    if kind not in DEFAULT_PROMPTS:
        raise KeyError(f"unknown prompt kind: {kind!r}")
    v = (overrides or {}).get(kind)
    return v if isinstance(v, str) and v else DEFAULT_PROMPTS[kind]


def resolve_params(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults merged with valid known keys only; unknown/invalid keys ignored."""
    out = dict(DEFAULT_PARAMS)
    out["gather_tools"] = list(out["gather_tools"])
    overrides = overrides or {}
    for key, spec in PARAM_SPEC.items():
        if key not in overrides or not isinstance(overrides[key], spec[0]):
            continue
        v = overrides[key]
        if len(spec) == 3:  # numeric with [lo, hi] range: clamp
            v = min(spec[2], max(spec[1], v))
        elif key == "gather_tools":
            v = [t for t in v if isinstance(t, str) and t]
            if not v:
                continue
        elif key == "sink_tool" and not v:
            continue
        out[key] = v
    return out
