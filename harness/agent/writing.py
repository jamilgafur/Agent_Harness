"""Writing: turn gathered material into a node's grounded answer.

The RUN'S HEAD writer (parent == root, several branch summaries in) is assembled
deterministically — intro + Bottom line from the model, body = verified branch
summaries verbatim. Branch writers are a single synthesis call, optionally
coordinated by the ensemble (samples/debate_rounds per-run params): K sampled
drafts, rival-aware revision, one vote.
"""
import ast
import json
import re
from typing import Any

from ..ensemble import Ensemble
from ..graph import named_parts
from ..prompts import resolve_params, resolve_prompt


def _code_ok(code: str) -> bool:
    """Is this real code for a source file? It must parse AND contain a statement
    of substance — a bare placeholder ('fixed_content', 'pass') parses as a lone
    expression and landed in live T2's analyze.py while every gate waved it on."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False

    def substantive(s: ast.stmt) -> bool:
        if isinstance(s, ast.Pass):
            return False
        if isinstance(s, ast.Expr) and isinstance(s.value, (ast.Name, ast.Constant)):
            return False          # placeholder word, or a docstring on its own
        return True

    return any(substantive(s) for s in tree.body)


def _write_payload(node: dict[str, Any],
                   inputs: dict[str, Any]) -> dict[str, Any]:
    """Finalize a write's content before it lands. A synthesis answer wraps the
    corrected file in prose and markdown fences — live T2's complete fix sat
    inside ```python ... ``` while the write carried the analysis prose, and the
    judge (correctly) refused it every attempt. A code file must land as CODE:
    the text itself if it is real code, else the largest fenced block that is."""
    args = dict(node.get("args") or {})
    content = args.get("content")
    if isinstance(content, dict) and "$from" in content:
        src = inputs.get(content["$from"])
        p = str(content.get("path") or "")
        if isinstance(src, dict) and p:
            content = src.get(p)
        elif src is not None and not p:
            content = src
    if not isinstance(content, str):
        return node                      # bindings and non-text stay as they are
    path = str(args.get("path") or "")
    if path.endswith(".py") and not _code_ok(content):
        blocks = [b for b in re.findall(r"```(?:python)?\n(.*?)```", content, re.S)
                  if _code_ok(b)]
        if blocks:
            content = max(blocks, key=len)
    args["content"] = content
    return {**node, "args": args}


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


def _llm_ok(text: str) -> bool:
    """Deterministic acceptance for synthesis: present and substantive.
    LLM-judging prose against model-written predicates was brittle (false 'unmet'
    verdicts escalated whole runs); success_predicate stays as guidance only.
    The check runs on the BODY: a Sources footer must not pad a thin answer past
    the bar, and a whole-answer card like "[Missing specific recipe details]" is
    the no-material failure signal, not an answer — accepting it wrote
    "[Missing ...]" chains straight into run heads (decomp benchmark task 7)."""
    body = re.sub(r"\n+Sources:\n(?:- .+\s*)+$", "", text or "").strip()
    if re.fullmatch(r"\[[^\[\]]+\]|\([^\(\)]+\)", body):
        return False
    return len(body) >= 40


class WriterMixin:
    """Answer writing for the virtual `llm` tool. Requires BaseAgent plumbing."""

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
        keeps the tree's coverage. Everyone else: synthesis, with ensemble
        coordination when the run enables it (samples/debate_rounds)."""
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
        if node.get("hint"):
            # error-driven refinement: the author must SEE why the previous
            # output broke (runtime traceback of the file it produced)
            pre += f"A previous attempt failed: {node['hint']}\nProduce corrected output.\n"
        if node.get("parent") == "root":
            rn = self.http.get(f"/runs/{run_id}").json()["nodes"]
            sections: list[tuple[str, str]] = []
            for d, res in inputs.items():
                # every material shape, not just sibling-sink text: shell stdout
                # and read_file content used to be dropped outright (live T5:
                # all deps were tools -> zero sections -> the answer collapsed to
                # the model's first sentence and lost 2 of 3 requested facts)
                raw = (res.get("text") or res.get("stdout") or res.get("content")
                       if isinstance(res, dict) else None)
                if isinstance(raw, str) and d in rn:
                    t = re.sub(r"\n+Sources:\n(?:- .+\s*)+$", "", raw).strip()
                    if t:
                        sections.append((rn[d]["intent"], t))
            if sections:
                # ANY surviving material becomes a verbatim section: regenerating
                # from a single dep paraphrased away a requested fact (live T5:
                # the one read_file result held 'Petra Lund', the chat answer
                # dropped it)
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
        ask = (pre + f"Task: {node['intent']}\n"
                     f"Material from the neighboring nodes:\n{material}")
        best = Ensemble(self.llm, sys_p, model=pa["model"],
                        temperature=pa["temperature"],
                        prompts=prompts).run(ask, pa["samples"], pa["debate_rounds"])
        return (best or "").strip()
