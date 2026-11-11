"""Shared data models: paper sections III-B (message schema) and IV-A (task DAG)."""
from enum import Enum
from typing import Any

from pydantic import BaseModel

DEFAULT_MAX_DEPTH = 17  # user requirement: decomposition depth cap

# Code-level defaults. harness.toml (volume-mounted into every container) OVERRIDES
# these at runtime; env (HARNESS_CONFIG / per-key vars) wins over both. These values
# are the fallback when the toml is missing and MUST stay in sync with the shipped
# harness.toml — tests/test_core_api.py::test_shipped_policy_matches_code_defaults gates it.
DEFAULT_POLICY = {
    "max_depth": DEFAULT_MAX_DEPTH,
    "max_nodes": 300,
    "max_llm_calls": 500,
    "max_seconds": 7200,
    "max_repairs": 3,
    "confirm_actions": True,      # tools not in auto_approve park an approval card
    "auto_approve": ["web_search", "http_fetch", "read_file", "list_dir", "llm"],  # read-only + synthesis
    "approve_dag_edits": False,   # plans validated server-side; True = eyeball every decomposition
    "claim_lease_seconds": 120,   # stale RUNNING nodes requeue after this
    "heartbeat_seconds": 5,
    "max_agents": 4,
    "event_cap": 500,             # retained audit events per run
}


class NodeState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"      # decomposed further; waits on children
    DONE = "DONE"
    REPAIRING = "REPAIRING"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"      # downstream of a FAILED node (answer 9)
    ESCALATED = "ESCALATED"  # handed to the user with a hint (answer 10)


class NodeType(str, Enum):
    TASK = "task"
    LOOP = "loop"


class Node(BaseModel):
    id: str
    intent: str
    tool: str | None = None
    args: dict[str, Any] = {}
    deps: list[str] = []
    success_predicate: str = ""
    state: NodeState = NodeState.PENDING
    depth: int = 0
    type: NodeType = NodeType.TASK
    result: Any | None = None
    attempts: int = 0
    llm_calls: int = 0          # per-node LLM spend (audit; run stats sum these)
    hint: str = ""
    parent: str = ""           # set when created by decomposition; parent waits on children
    claimed_at: float = 0.0    # heartbeat of the last claim; stale RUNNING nodes are requeued
    routed_past: list[str] = []  # agents that redirected this node (paper Alg-2: visited set)
    # loop nodes only: condition is judged each iteration, body is child ids
    condition: str = ""
    max_iterations: int = 10


# The paper III-B message schema (Task/Result/Redirect/Heartbeat + routing
# metadata) is documented in plan.txt §6; the wire format is plain REST JSON,
# so the envelope classes are not modeled here.


class PatchOp(BaseModel):
    op: str  # add_node | add_edge | replace_tool | retry_node
    node: Node | None = None
    src: str | None = None
    dst: str | None = None
    node_id: str | None = None
    tool: str | None = None
    args: dict[str, Any] | None = None


class Patch(BaseModel):
    ops: list[PatchOp]
    reason: str = ""


# ---------- HTTP request bodies (core API) ----------
class RunIn(BaseModel):
    intent: str = ""
    nodes: list[Node] = []


class ClaimIn(BaseModel):
    agent_id: str          # one node per claim; parallelism is --scale agent=N


class ResultIn(BaseModel):
    state: str               # DONE | FAILED
    result: Any | None = None
    hint: str = ""
    llm_calls: int = 0
    agent_id: str = ""


class ChildIn(BaseModel):
    intent: str
    tool: str | None = None
    args: dict[str, Any] = {}
    deps: list[Any] = []    # sibling indices (agent) or sibling ids (approval round-trip)
    success_predicate: str = ""


class DecomposeIn(BaseModel):
    children: list[ChildIn]


class RedirectIn(BaseModel):
    agent_id: str
    reason: str = ""


class ExecIn(BaseModel):
    node_id: str
    tool: str
    args: dict[str, Any] = {}


class DecisionIn(BaseModel):
    decision: str            # approve | reject
    args: dict[str, Any] | None = None   # exec: edited args; patch: {"children": [...]}


class HeartbeatIn(BaseModel):
    agent_id: str
    run_id: str = ""
    node_id: str = ""
    phase: str = "idle"


class GuideIn(BaseModel):
    text: str


class ConfigIn(BaseModel):
    prompts: dict[str, str] = {}
    params: dict[str, Any] = {}


class ConstraintIn(BaseModel):
    constraints: list[str]


class SkillIn(BaseModel):
    """Declarative runtime skill: spec → sandbox writes tools/{name}.py (shell
    template; {arg} placeholders). Custom python: drop a .py in the workspace."""
    name: str
    desc: str = ""
    required: list[str] = []
    cmd: str


class AddedNode(BaseModel):
    intent: str
    parent: str = "root"


class NodeUpdate(BaseModel):
    intent: str = ""
    args: dict[str, Any] | None = None
