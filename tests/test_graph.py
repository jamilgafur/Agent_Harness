from harness.graph import apply_patch, ready_nodes, validate
from harness.models import Node, NodeState, Patch, PatchOp

TOOLS = ["shell", "write_file"]


def test_validate_ok_and_ready_progress():
    a = Node(id="a", intent="root", tool="shell", args={"cmd": "true"})
    b = Node(id="b", intent="child", deps=["a"], tool="shell",
             success_predicate="prints ok", args={"cmd": "echo ok"})
    nodes = {"a": a, "b": b}
    assert validate(nodes, TOOLS, auto_approve=["shell"]) == []
    assert [n.id for n in ready_nodes(nodes)] == ["a"]
    a.state = NodeState.DONE
    assert [n.id for n in ready_nodes(nodes)] == ["b"]


def test_tool_node_needs_predicate_unless_auto_approved():
    n = {"x": Node(id="x", intent="work", tool="write_file",
                   args={"path": "f", "content": "c"})}
    assert validate(n, TOOLS)  # missing predicate -> rejected
    assert validate(n, TOOLS, auto_approve=["write_file"]) == []  # exempted


def test_cycle_and_depth_rejected():
    a, b = Node(id="a", intent="a", deps=["b"]), Node(id="b", intent="b", deps=["a"])
    assert "cycle" in validate({"a": a, "b": b}, TOOLS)[0]
    chain = {str(i): Node(id=str(i), intent="n", deps=[str(i - 1)] if i else [])
             for i in range(20)}
    assert any("max_depth" in e for e in validate(chain, TOOLS, max_depth=17))


def test_apply_patch_atomic_and_retry_guard():
    nodes = {"a": Node(id="a", intent="root")}
    kid = Node(id="a.1", intent="kid", parent="a", tool="shell",
               args={"cmd": "true"})
    new, errs = apply_patch(nodes, Patch(ops=[PatchOp(op="add_node", node=kid)]), TOOLS,
                            auto_approve=["shell"])
    assert errs == [] and "a.1" in new and "a.1" not in nodes  # original untouched
    # cannot retry a parent that has children
    _, errs = apply_patch(new, Patch(ops=[PatchOp(op="retry_node", node_id="a")]), TOOLS,
                          auto_approve=["shell"])
    assert errs and "failing child" in errs[0]
    # repairing nodes are re-claimable
    new["a.1"].state = NodeState.REPAIRING
    ready_ids = {n.id for n in ready_nodes(new)}
    assert "a.1" in ready_ids  # REPAIRING re-enters the ready set
    new["a.1"].state = NodeState.DONE
    assert "a.1" not in {n.id for n in ready_nodes(new)}


def test_unknown_tool_and_bad_edge():
    bad = Node(id="x", intent="x", tool="nope", success_predicate="p")
    assert "unknown tool" in validate({"x": bad}, TOOLS)[0]
    _, errs = apply_patch({}, Patch(ops=[PatchOp(op="add_edge", src="a", dst="b")]), TOOLS)
    assert "unknown endpoint" in errs[0]
