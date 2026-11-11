from harness.graph import apply_patch, lint, ready_nodes, resolve_args, validate
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


def test_ready_nodes_resilient_writer_runs_on_terminal_deps():
    a = Node(id="a", intent="gather", tool="shell", args={"cmd": "x"})
    w = Node(id="w", intent="answer", deps=["a"], tool="llm")
    nodes = {"a": a, "w": w}
    a.state = NodeState.FAILED
    assert [n.id for n in ready_nodes(nodes)] == []              # strict: waits
    assert [n.id for n in ready_nodes(nodes, resilient=["w"])] == ["w"]
    a.state = NodeState.RUNNING      # not terminal yet -> the writer still waits
    assert ready_nodes(nodes, resilient=["w"]) == []


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


# ---------- named-parts extraction (document-shape detection) ----------
def test_named_parts_extracts_named_pieces_only():
    from harness.graph import named_parts
    assert named_parts("write a white paper on astrophysics with an abstract, "
                       "introduction, background, and conclusion") == \
        ["abstract", "introduction", "background", "conclusion"]
    assert named_parts("steps: download, parse, summarize") == \
        ["download", "parse", "summarize"]
    assert named_parts("fix the parser") == []
    assert named_parts("tell me about quantization methods for LLMs") == []
    assert named_parts("compare vLLM and TGI") == []            # no list syntax
    assert named_parts("write it with at least 3 sources") == []  # single piece


# ---------- $from arg bindings (data flow into tool args) ----------
def test_from_binding_validated_and_resolved():
    prod = Node(id="mk", intent="create employee", tool="shell", args={"cmd": "mk"},
                success_predicate="p")
    use = Node(id="as", intent="assign them", tool="shell", deps=["mk"],
               args={"cmd": "assign", "who": {"$from": "mk", "path": "employee.id"}},
               success_predicate="p")
    nodes = {"mk": prod, "as": use}
    assert validate(nodes, TOOLS, auto_approve=["shell"]) == []
    # binding a node that is not a dep loses the ordering guarantee -> rejected
    bad_dep = use.model_copy(deep=True)
    bad_dep.deps = []
    assert any("not in deps" in e for e in validate({"mk": prod, "as": bad_dep}, TOOLS))
    # binding an unknown node -> rejected
    unknown = use.model_copy(deep=True)
    unknown.args = {"cmd": "assign", "who": {"$from": "ghost"}}
    assert any("unknown node" in e for e in validate({"as": unknown}, TOOLS))
    # resolution: path digs into the dep's result; missing path is an error
    prod.state, prod.result = NodeState.DONE, {"employee": {"id": 7}}
    args, err = resolve_args(nodes, use)
    assert err == "" and args == {"cmd": "assign", "who": 7}
    prod.result = {"employee": {}}
    _, err = resolve_args(nodes, use)
    assert "not found" in err


def test_lint_flags_child_restating_parent():
    p = Node(id="root", intent="research vLLM tuning in depth")
    dup = Node(id="root.1", intent="research vLLM tuning in depth", parent="root")
    real = Node(id="root.2", intent="gather throughput numbers", parent="root")
    warns = lint({"root": p, "root.1": dup, "root.2": real})
    assert any("root.1" in w and "restates" in w for w in warns)
    assert not any("root.2" in w for w in warns)
