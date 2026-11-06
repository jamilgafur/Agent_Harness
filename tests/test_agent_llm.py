"""Agent-side material formatting + upward traversal for the virtual `llm` tool."""
from harness.agent import Agent, _deepen_branches, _force_synthesis, _llm_ok, _material


def test_material_formats_search_hits():
    got = _material({"root.1": {"results": [
        {"title": "T1", "url": "https://a", "snippet": "s1"},
        {"title": "T2", "url": "https://b", "snippet": "s2"}]}})
    assert "[from root.1]" in got and "- T1: https://a" in got and "s2" in got


def test_material_prefers_text_and_content_fields():
    got = _material({"a": {"text": "prose answer"}, "b": {"content": "page text"}})
    assert "prose answer" in got and "page text" in got


def test_material_skips_none_and_flags_empty():
    assert "no material" in _material({"a": None})
    assert "no material" in _material({})


def test_llm_ok_requires_substantive_text():
    assert not _llm_ok("")
    assert not _llm_ok("short")
    assert _llm_ok("Structured pruning removes entire channels or filters from a network.")


def test_upstream_walks_ancestors_via_deps_and_parents():
    class FakeResp:
        def json(self):
            return {"nodes": {
                "root":     {"parent": None, "deps": [], "state": "DONE", "result": {"text": "root agg"}},
                "root.1":   {"parent": "root", "deps": [], "state": "DONE", "result": {"results": []}},
                "root.2":   {"parent": "root", "deps": ["root.1"], "state": "RUNNING", "result": None},
                "unrelated": {"parent": None, "deps": [], "state": "DONE", "result": {"text": "nope"}},
            }}

    class FakeHttp:
        def get(self, path):
            assert path == "/runs/r"
            return FakeResp()

    a = object.__new__(Agent)
    a.http = FakeHttp()
    got = a._upstream("r", {"id": "root.2", "parent": "root", "deps": ["root.1"]})
    assert set(got) == {"root.1", "root"}          # ancestors only, DONE-with-result only


def _search(i):
    return {"intent": f"search {i}", "tool": "web_search",
            "args": {"query": f"q{i}"}, "deps": [], "success_predicate": "hits"}


def test_force_synthesis_appends_sink_when_missing():
    kids = _force_synthesis([_search(1), _search(2)], "tell me about x")
    assert [c["tool"] for c in kids] == ["web_search", "web_search", "llm"]
    assert kids[2]["deps"] == [0, 1]
    assert "tell me about x" in kids[2]["intent"]


def test_force_synthesis_rewires_misplaced_sink_and_breaks_back_edges():
    llm = {"intent": "write", "tool": "llm", "deps": [0]}
    s2 = dict(_search(2)); s2["deps"] = [1]        # wrongly downstream of the sink
    kids = _force_synthesis([_search(1), llm, s2], "task")
    assert kids[2]["deps"] == []                   # back-edge to sink removed
    assert kids[1]["deps"] == [0, 2]               # sink now depends on every sibling


def test_force_synthesis_leaves_task_shapes_alone():
    kids = [{"intent": "write", "tool": "write_file",
             "args": {"path": "a", "content": "b"}, "deps": []}]
    assert _force_synthesis(kids, "do x") == kids


def test_force_synthesis_converges_sink_over_abstract_branches():
    kids = [{"intent": "branch A", "tool": None, "deps": []},
            {"intent": "branch B", "tool": None, "deps": []},
            {"intent": "write the answer", "tool": "llm", "deps": []}]
    out = _force_synthesis(kids, "big research task")
    assert out[0]["tool"] is None and out[1]["tool"] is None    # branches stay abstract
    assert out[2]["deps"] == [0, 1]                             # sink converges on both


def test_force_synthesis_converts_all_llm_plan_into_grounded_shape():
    llms = [{"intent": f"aspect {i}", "tool": "llm", "deps": []} for i in range(3)]
    kids = _force_synthesis(llms, "research x")
    assert [c["tool"] for c in kids] == ["web_search", "web_search", "llm"]
    assert kids[0]["args"]["query"] == "aspect 0"      # intent becomes the grounding query
    assert kids[2]["deps"] == [0, 1]                   # last writer sinks everything


def test_force_synthesis_flattens_degenerate_gather_prefix_chain():
    s1, s2, s3 = _search(1), _search(2), _search(3)
    s2["deps"], s3["deps"] = [0], [0, 1]          # cumulative prefix chain — model's habit
    kids = _force_synthesis([s1, s2, s3, {"intent": "s", "tool": "llm", "deps": []}], "x")
    assert kids[0]["deps"] == [] and kids[1]["deps"] == [] and kids[2]["deps"] == []
    assert kids[3]["deps"] == [0, 1, 2]            # sink still converges on all three


def test_force_synthesis_keeps_mixed_chains_alone():
    s1, w = _search(1), {"intent": "write", "tool": "write_file",
                         "args": {"path": "a", "content": "b"}, "deps": [0]}
    kids = _force_synthesis([s1, w], "task")
    assert kids[1]["deps"] == [0]                  # gather→task edge is real data flow


def test_deepen_branches_turns_flat_root_research_plan_into_a_tree():
    flat = [_search(1), _search(2),
            {"intent": "write", "tool": "llm", "deps": [0, 1]}]
    out = _deepen_branches(flat, depth=0)
    assert [c["tool"] for c in out] == [None, None, "llm"]     # searches become branches
    assert out[0]["deps"] == [] and out[1]["deps"] == []
    assert out[2]["deps"] == [0, 1]                             # sink keeps all deps


def test_deepen_branches_skips_nonroot_and_task_plans():
    flat = [_search(1), _search(2), {"intent": "s", "tool": "llm", "deps": [0, 1]}]
    assert _deepen_branches(flat, depth=1) == flat              # branch level stays shallow
    task = [_search(1), {"intent": "w", "tool": "shell", "args": {"cmd": "x"}, "deps": [0]}]
    assert _deepen_branches(task, depth=0) == task              # mixed tools = real task plan


def test_force_synthesis_grounds_lone_llm_child():
    out = _force_synthesis([{"intent": "define QAT", "tool": "llm", "deps": []}], "x")
    assert [c["tool"] for c in out] == ["web_search", "llm"]
    assert out[0]["args"]["query"] == "define QAT" and out[1]["deps"] == [0]


def test_force_synthesis_clamps_bad_dep_indices():
    s = _search(1); s["deps"] = [9, "root", 0]
    kids = _force_synthesis([s], "x")
    assert kids[0]["deps"] == []                   # stale, non-int and self-indices dropped
    assert kids[1]["tool"] == "llm" and kids[1]["deps"] == [0]
