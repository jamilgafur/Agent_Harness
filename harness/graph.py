"""DAG validation and patch application — paper IV-A (Theorem 2 conditions),
extended with runtime patch validation (answered: edits validated, graph grows)."""
import re
from collections.abc import Iterable
from typing import Any

import networkx as nx

from .models import DEFAULT_MAX_DEPTH, Node, NodeState, NodeType, Patch


def _arg_binding(v: Any) -> str | None:
    """An arg value of {"$from": "<node-id>", "path": "a.b.c"?} binds a dep's
    result into args at exec time (data flow into tools, not just the writer)."""
    return v.get("$from") if isinstance(v, dict) and isinstance(v.get("$from"), str) else None


def _to_nx(nodes: dict[str, Node]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes.values():
        g.add_node(n.id)
        for d in n.deps:
            g.add_edge(d, n.id)
    return g


def depth_of(nodes: dict[str, Node]) -> int:
    lengths: dict[str, int] = {}

    def walk(nid: str, seen: set) -> int:
        if nid in lengths:
            return lengths[nid]
        if nid in seen or nid not in nodes:
            return 0
        seen.add(nid)
        n = nodes[nid]
        lengths[nid] = 1 + max((walk(d, seen) for d in n.deps), default=0)
        return lengths[nid]

    return max((walk(nid, set()) for nid in nodes), default=0)


def descendants(nodes: dict[str, Node], nid: str) -> list[str]:
    return [d for d in nx.descendants(_to_nx(nodes), nid) if d in nodes]


def validate(nodes: dict[str, Node], known_tools: Iterable[str],
             max_depth: int = DEFAULT_MAX_DEPTH, max_nodes: int = 300,
             auto_approve: Iterable[str] = ()) -> list[str]:
    """Return a list of validation errors ([] = valid). Pure function, easy to test."""
    errors: list[str] = []
    tools = set(known_tools)
    preapproved = set(auto_approve)
    if len(nodes) > max_nodes:
        errors.append(f"budget: {len(nodes)} nodes exceeds max_nodes={max_nodes}")
    g = nx.DiGraph()
    for n in nodes.values():
        g.add_node(n.id)
        for d in n.deps:
            if d not in nodes:
                errors.append(f"{n.id}: dep {d!r} does not exist")
            else:
                g.add_edge(d, n.id)
        if not n.intent.strip():
            errors.append(f"{n.id}: empty intent")
        if n.tool and n.tool not in tools and n.type == NodeType.TASK:
            errors.append(f"{n.id}: unknown tool {n.tool!r}")
        # judged nodes must state their success criteria up front; auto-approved
        # tools get a deterministic default instead (shell: rc==0)
        if (n.type == NodeType.TASK and n.tool and n.tool not in preapproved
                and not n.success_predicate.strip()):
            errors.append(f"{n.id}: tool node needs a success_predicate (tool not auto-approved)")
        if n.type == NodeType.LOOP and (not n.condition or n.max_iterations < 1):
            errors.append(f"{n.id}: loop node needs condition and max_iterations >= 1")
        if n.type == NodeType.LOOP and not n.tool:
            errors.append(f"{n.id}: loop node needs a tool (its iterated body)")
        for key, v in n.args.items():
            src = _arg_binding(v)
            if src is None:
                continue
            if src not in nodes:
                errors.append(f"{n.id}: arg {key!r} binds unknown node {src!r}")
            elif src not in n.deps:
                # ordering guarantee: $from must ride a declared dep or the value
                # may not exist when the node runs
                errors.append(f"{n.id}: arg {key!r} binds {src!r} but it is not in deps")
    if errors:  # structural errors make graph checks meaningless
        return errors
    if not nx.is_directed_acyclic_graph(g):
        errors.append("graph has a cycle (use a loop node instead of cycle edges)")
        return errors
    d = depth_of(nodes)
    if d > max_depth:
        errors.append(f"depth {d} exceeds max_depth={max_depth}")
    return errors


_NAMED_RE = re.compile(r"\b(?:with|containing|including|sections?:|parts?:|steps?:)\s+"
                       r"(?:an?\s+|the\s+)?([a-z][a-z0-9 ,/&'-]+?)(?:[.;]|$)", re.I)


def named_parts(intent: str) -> list[str]:
    """Generic list extraction: an intent that NAMES its pieces
    ('…with an abstract, introduction, background and conclusion', 'steps: download,
    parse, summarize') yields those pieces. Vocabulary-free — the trigger is the
    list syntax, not any domain word; 2+ pieces means the task is compound and the
    break-up logic (branch coverage, section wiring) should fire."""
    m = _NAMED_RE.search(intent or "")
    if not m:
        return []
    parts: list[str] = []
    for piece in re.split(r",|\band\b|&", m.group(1), flags=re.I):
        piece = re.sub(r"^(?:an?|the)\s+", "", piece.strip(), flags=re.I)
        words = piece.split()
        if 1 <= len(words) <= 6 and not any(w in piece.lower() for w in (" at ", " for ", " on ")):
            parts.append(piece)
    # dedupe, keep order; a real list has 2..8 pieces
    seen = [p for p in dict.fromkeys(parts)]
    return seen if 2 <= len(seen) <= 8 else []


_STOP = frozenset("a an the and or of to in on for with by from at as is are be it its "
                  "this that these those into about over under each per vs".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP}


def lint(nodes: dict[str, Node]) -> list[str]:
    """Non-blocking warnings: unreachable sinks nobody consumes, and children that
    restate their parent (decomposition must get SIMPLER down the tree — a child
    whose tokens are ~the parent's adds a level without adding a split)."""
    warnings: list[str] = []
    consumed = {d for n in nodes.values() for d in n.deps}
    for n in nodes.values():
        if n.deps and n.id not in consumed and n.state == NodeState.PENDING:
            warnings.append(f"{n.id}: output is never consumed (possible wasted work)")
        p = nodes.get(n.parent) if n.parent else None
        if p:
            pt, ct = _tokens(p.intent), _tokens(n.intent)
            if pt and ct and len(pt & ct) / len(pt | ct) > 0.7:
                warnings.append(f"{n.id}: intent restates its parent — replace it with "
                                f"a real sub-part or delete the level")
    return warnings


def apply_patch(nodes: dict[str, Node], patch: Patch, known_tools: Iterable[str],
                max_depth: int = DEFAULT_MAX_DEPTH, max_nodes: int = 300,
                auto_approve: Iterable[str] = ()
                ) -> tuple[dict[str, Node] | None, list[str]]:
    """Validate a patch against a *copy*; return (new_nodes, errors). Original untouched."""
    trial = {k: v.model_copy(deep=True) for k, v in nodes.items()}
    errors: list[str] = []
    for op in patch.ops:
        if op.op == "add_node":
            if op.node is None:
                errors.append("add_node: missing node")
            elif op.node.id in trial:
                errors.append(f"add_node: {op.node.id} already exists")
            else:
                trial[op.node.id] = op.node
        elif op.op == "add_edge":
            if op.src in trial and op.dst in trial and op.src and op.dst:
                if op.src not in trial[op.dst].deps:
                    trial[op.dst].deps.append(op.src)
            else:
                errors.append(f"add_edge: unknown endpoint {op.src}->{op.dst}")
        elif op.op == "replace_tool":
            if op.node_id in trial:
                trial[op.node_id].tool = op.tool
                if op.args is not None:
                    trial[op.node_id].args = op.args
            else:
                errors.append(f"replace_tool: unknown node {op.node_id}")
        elif op.op == "retry_node":
            if op.node_id in trial:
                n = trial[op.node_id]
                if any(v.parent == op.node_id for v in trial.values()):
                    errors.append(f"retry_node: {op.node_id} has children — retry the failing child")
                else:
                    n.state = NodeState.PENDING
                    n.attempts = 0
                    n.hint = ""
                    n.routed_past = []   # user fix resets Alg-2 visited set
            else:
                errors.append(f"retry_node: unknown node {op.node_id}")
        else:
            errors.append(f"unknown op {op.op!r}")
    if errors:
        return None, errors
    verrs = validate(trial, known_tools, max_depth, max_nodes, auto_approve)
    return (None, verrs) if verrs else (trial, [])


_TERMINAL = (NodeState.DONE, NodeState.FAILED, NodeState.BLOCKED, NodeState.ESCALATED)


def ready_nodes(nodes: dict[str, Node], resilient: Iterable[str] = ()) -> list[Node]:
    """Topologically-ready work: PENDING or REPAIRING, all deps DONE.
    `resilient` ids (the writers) are ready once every dep is TERMINAL — they
    answer from whatever material survived a dead sibling; a strict all-DONE
    gate left them PENDING-but-unclaimable and the run never ended."""
    out = []
    for n in nodes.values():
        if n.state not in (NodeState.PENDING, NodeState.REPAIRING):
            continue
        if all(nodes[d].state == NodeState.DONE for d in n.deps):
            out.append(n)
        elif n.id in set(resilient) \
                and all(nodes[d].state in _TERMINAL for d in n.deps):
            out.append(n)
    return out


def inputs_for(nodes: dict[str, Node], node: Node) -> dict[str, Any]:
    """Blackboard stand-in: a node sees its dependencies' results."""
    return {d: nodes[d].result for d in node.deps if d in nodes}


def _dig(result: Any, path: str) -> Any:
    cur = result
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def resolve_args(nodes: dict[str, Node], node: Node) -> tuple[dict[str, Any], str]:
    """Substitute {"$from": id, "path": ...} arg bindings from dep results.
    Returns (resolved_args, error). Unresolvable bindings are an error, not a
    silent null — a tool must not run on placeholder data."""
    out: dict[str, Any] = {}
    for key, v in node.args.items():
        src = _arg_binding(v)
        if src is None:
            out[key] = v
            continue
        dep = nodes.get(src)
        if dep is None or dep.state != NodeState.DONE or dep.result is None:
            return {}, f"{node.id}: arg {key!r} binds {src!r}, which has no result"
        val = dep.result if not v.get("path") else _dig(dep.result, str(v["path"]))
        if val is None:
            return {}, f"{node.id}: arg {key!r} binds {src!r}.{v.get('path')} — not found"
        out[key] = val
    return out, ""
