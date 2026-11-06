"""Agent-side material formatting + upward traversal for the virtual `llm` tool."""
from harness.agent import Agent, _deepen_branches, _force_synthesis, _llm_ok, _material, _sources


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


def test_sources_footers_dedupe_and_reach_into_aggregates():
    leaf = {"results": [{"url": "https://a", "page_text": "t"},
                        {"url": "https://a", "page_text": "t"},
                        {"url": "https://b", "page_text": "t"}]}
    aggregate = {"root.1.1": leaf, "root.1.2": {"text": "summary"}}
    got = _sources({"root.1": aggregate})
    assert got.startswith("\n\nSources:\n")
    assert got.count("https://a") == 1 and "https://b" in got
    assert _sources({"x": {"text": "no urls here"}}) == ""


def test_sources_recover_links_from_climbing_branch_summaries():
    # parents hold {"text": ..., "from": ...} now — the head only sees prose,
    # so its footer must be mined back out of the branch summaries' own footers
    branch = {"text": "Prose answer.\n\nSources:\n- https://a/p\n- https://b/q",
              "from": "root.1.4"}
    got = _sources({"root.1": branch, "root.2": {"text": "no links", "from": "root.2.2"}})
    assert "- https://a/p" in got and "- https://b/q" in got


def test_judge_gather_tools_is_deterministic_and_never_calls_the_llm():
    class BoomLlm:
        def chat_json(self, *a, **k):
            raise AssertionError("the LLM must not judge gather results")

    a = object.__new__(Agent)
    a.llm = BoomLlm()
    node = {"tool": "web_search", "success_predicate": "relevant hits", "intent": "x"}
    assert a.judge(node, {"results": [{"url": "u", "page_text": "…"}]})
    assert a.judge(node, {"content": "page text"})
    assert not a.judge(node, {"results": []})
    assert not a.judge(node, {"rc": -5, "stderr": "fetch failed"})


def test_material_prefers_sibling_summaries_in_aggregate_results():
    aggregate = {"root.2.1": {"results": [{"title": "raw", "url": "u", "snippet": "s"}]},
                 "root.2.4": {"text": "branch summary prose"}}
    got = _material({"root.2": aggregate})
    assert "branch summary prose" in got
    assert '"results"' not in got          # raw leaf dumps stay below ─ only summaries rise


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


def test_decompose_prompt_carries_user_constraints():
    captured = {}

    class FakeLlm:
        calls = 0

        def chat_json(self, sys_prompt, user_prompt):
            captured["prompt"] = user_prompt
            return {"children": [{"intent": "s", "tool": "web_search",
                                  "args": {"query": "q"}, "deps": [],
                                  "success_predicate": "hits"}]}

    a = object.__new__(Agent)
    a.llm = FakeLlm()
    out = a.decompose({"id": "root", "intent": "t", "depth": 0, "hint": ""}, {},
                      {"web_search": {"required": ["query", "max_results"]}}, 17,
                      ["ignore methodology"])
    assert "ignore methodology" in captured["prompt"]
    assert any(c.get("tool") == "llm" for c in out["children"])   # sink pinned on


def test_decompose_prompt_names_neighbor_branches():
    captured = {}

    class FakeLlm:
        calls = 0

        def chat_json(self, sys_p, user_p):
            captured["prompt"] = user_p
            return {"children": [{"intent": "s", "tool": "web_search",
                                  "args": {"query": "q"}, "deps": [],
                                  "success_predicate": "hits"}]}

    a = object.__new__(Agent)
    a.llm = FakeLlm()
    a.decompose({"id": "root.4", "intent": "compare TCO of engines", "depth": 1, "hint": ""},
                {}, {"web_search": {"required": ["query", "max_results"]}}, 17, [],
                {"parent_intent": "research paper on serving",
                 "run_intent": "research paper on serving",
                 "siblings": [{"id": "root.2", "intent": "quantization basics",
                               "tool": None}]})
    p = captured["prompt"]
    assert "quantization basics" in p and "research paper on serving" in p


# ---------- head-answer assembly (deterministic sections, 3b-safe model asks) ----------
class _WriterLlm:
    def __init__(self):
        self.replies = []

    def chat(self, sys_p, user_p, json_mode=False):
        self.replies.append(user_p)
        if "introduction" in user_p:
            return "INTRO SENTENCES."
        if "concluding" in user_p:
            return "BOTTOM LINE SENTENCES."
        return "GENERIC ANSWER."


class _RunHttp:
    def get(self, path):
        assert path == "/runs/r"

        class R:
            def json(self):
                return {"nodes": {
                    "root":   {"intent": "the run question"},
                    "root.1": {"intent": "branch one intent"},
                    "root.2": {"intent": "branch two intent"},
                }}
        return R()


def test_head_writer_assembles_sections_from_branch_summaries():
    a = object.__new__(Agent)
    a.http = _RunHttp()
    a.llm = _WriterLlm()
    inputs = {
        "root.1": {"text": "branch one prose with specifics.\n\nSources:\n- https://x"},
        "root.2": {"text": "branch two prose."},
    }
    t = a._write_answer("r", {"id": "root.4", "parent": "root", "intent": "sink"},
                        inputs, "MATERIAL", [])
    assert t.startswith("# the run question")
    assert "INTRO SENTENCES." in t and "## Bottom line\n\nBOTTOM LINE SENTENCES." in t
    assert "## branch one intent\n\nbranch one prose with specifics." in t
    assert "## branch two intent\n\nbranch two prose." in t
    assert "https://x" not in t            # per-branch link tails stripped before assembly
    assert len(a.llm.replies) == 2         # intro + outro only: body is not re-generated


def test_branch_writer_and_thin_head_use_plain_synthesis():
    a = object.__new__(Agent)
    a.llm = _WriterLlm()          # replies beyond the first two => "GENERIC ANSWER."
    t = a._write_answer("r", {"id": "root.1.2", "parent": "root.1", "intent": "t"},
                        {"root.1.1": {"text": "x"}}, "MATERIAL", ["discuss cost"])
    assert t == "GENERIC ANSWER."
    assert "discuss cost" in a.llm.replies[0]

    a2 = object.__new__(Agent)
    a2.http = _RunHttp()
    a2.llm = _WriterLlm()
    t2 = a2._write_answer("r", {"id": "root.4", "parent": "root", "intent": "t"},
                          {"root.1": {"text": "only one section"}}, "MATERIAL", [])
    assert t2 == "GENERIC ANSWER."        # <2 sections: falls back to a single chat call


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
