"""DAG validation and patch application — paper IV-A (Theorem 2 conditions),
extended with runtime patch validation (answered: edits validated, graph grows)."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .models import DEFAULT_MAX_DEPTH, Node, NodeState, NodeType, Patch


def _to_nx(nodes: Dict[str, Node]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes.values():
        g.add_node(n.id)
        for d in n.deps:
            g.add_edge(d, n.id)
    return g


def depth_of(nodes: Dict[str, Node]) -> int:
    lengths: Dict[str, int] = {}

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


def descendants(nodes: Dict[str, Node], nid: str) -> List[str]:
    return [d for d in nx.descendants(_to_nx(nodes), nid) if d in nodes]


def validate(nodes: Dict[str, Node], known_tools: Iterable[str],
             max_depth: int = DEFAULT_MAX_DEPTH, max_nodes: int = 300,
             auto_approve: Iterable[str] = ()) -> List[str]:
    """Return a list of validation errors ([] = valid). Pure function, easy to test."""
    errors: List[str] = []
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
    if errors:  # structural errors make graph checks meaningless
        return errors
    if not nx.is_directed_acyclic_graph(g):
        errors.append("graph has a cycle (use a loop node instead of cycle edges)")
        return errors
    d = depth_of(nodes)
    if d > max_depth:
        errors.append(f"depth {d} exceeds max_depth={max_depth}")
    return errors


def lint(nodes: Dict[str, Node]) -> List[str]:
    """Non-blocking warnings: unreachable sinks nobody consumes are just noise to flag."""
    warnings: List[str] = []
    consumed = {d for n in nodes.values() for d in n.deps}
    for n in nodes.values():
        if n.deps and n.id not in consumed and n.state == NodeState.PENDING:
            warnings.append(f"{n.id}: output is never consumed (possible wasted work)")
    return warnings


def apply_patch(nodes: Dict[str, Node], patch: Patch, known_tools: Iterable[str],
                max_depth: int = DEFAULT_MAX_DEPTH, max_nodes: int = 300,
                auto_approve: Iterable[str] = ()
                ) -> Tuple[Optional[Dict[str, Node]], List[str]]:
    """Validate a patch against a *copy*; return (new_nodes, errors). Original untouched."""
    trial = {k: v.model_copy(deep=True) for k, v in nodes.items()}
    errors: List[str] = []
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
            else:
                errors.append(f"retry_node: unknown node {op.node_id}")
        else:
            errors.append(f"unknown op {op.op!r}")
    if errors:
        return None, errors
    verrs = validate(trial, known_tools, max_depth, max_nodes, auto_approve)
    return (None, verrs) if verrs else (trial, [])


def ready_nodes(nodes: Dict[str, Node]) -> List[Node]:
    """Topologically-ready work: PENDING or REPAIRING, all deps DONE."""
    return [n for n in nodes.values()
            if n.state in (NodeState.PENDING, NodeState.REPAIRING)
            and all(nodes[d].state == NodeState.DONE for d in n.deps)]


def inputs_for(nodes: Dict[str, Node], node: Node) -> Dict[str, Any]:
    """Blackboard stand-in: a node sees its dependencies' results."""
    return {d: nodes[d].result for d in node.deps if d in nodes}
