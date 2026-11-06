"""Shared data models: paper sections III-B (message schema) and IV-A (task DAG)."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

DEFAULT_MAX_DEPTH = 17  # user requirement: decomposition depth cap

DEFAULT_POLICY = {
    "max_depth": DEFAULT_MAX_DEPTH,
    "max_nodes": 300,
    "max_llm_calls": 500,
    "max_seconds": 7200,
    "max_repairs": 3,
    "confirm_actions": True,      # answer 13: confirm-everything mode
    "auto_approve": [],           # tool names pre-approved for unattended runs
    "approve_dag_edits": True,    # agent decompositions gated behind approval+edit dialog
    "claim_lease_seconds": 120,   # stale RUNNING nodes requeue after this
    "heartbeat_seconds": 5,
    "max_agents": 2,
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
    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    deps: List[str] = []
    success_predicate: str = ""
    state: NodeState = NodeState.PENDING
    depth: int = 0
    type: NodeType = NodeType.TASK
    result: Optional[Any] = None
    attempts: int = 0
    hint: str = ""
    parent: str = ""           # set when created by decomposition; parent waits on children
    claimed_at: float = 0.0    # heartbeat of the last claim; stale RUNNING nodes are requeued
    routed_past: List[str] = []  # agents that redirected this node (paper Alg-2: visited set)
    # loop nodes only: condition is judged each iteration, body is child ids
    condition: str = ""
    max_iterations: int = 10


class MsgType(str, Enum):
    TASK = "Task"
    RESULT = "Result"
    REDIRECT = "Redirect"
    HEARTBEAT = "Heartbeat"


class Header(BaseModel):
    sender: str
    receiver: str
    type: MsgType


class Routing(BaseModel):
    edge_weight: float = 1.0
    reliability: float = 1.0
    snapshot: Dict[str, Any] = {}


class Message(BaseModel):
    """Paper III-B: header / payload / routing metadata."""
    header: Header
    payload: Dict[str, Any] = {}
    routing: Routing = Routing()


class PatchOp(BaseModel):
    op: str  # add_node | add_edge | replace_tool | retry_node
    node: Optional[Node] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    node_id: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None


class Patch(BaseModel):
    ops: List[PatchOp]
    reason: str = ""
