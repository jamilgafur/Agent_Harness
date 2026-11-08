"""Agent-side material formatting + upward traversal for the virtual `llm` tool."""
from harness.agent import GATHER_TOOLS as GATHER
from harness.agent import SINK_TOOL as SINK
from harness.agent import (
    Agent,
    _deepen_branches,
    _force_synthesis,
    _llm_ok,
    _material,
    _sources,
)


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

        def chat_json(self, sys_prompt, user_prompt, **k):
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

        def chat_json(self, sys_p, user_p, **k):
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
        self.systems = []

    def chat(self, sys_p, user_p, **kw):
        self.systems.append(sys_p)
        self.replies.append(user_p)
        if "reader should take away" in user_p:
            return "BOTTOM LINE SENTENCES."
        if "introduction" in user_p:
            return "INTRO SENTENCES."
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
    s2 = dict(_search(2))
    s2["deps"] = [1]                         # wrongly downstream of the sink
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
    s = _search(1)
    s["deps"] = [9, "root", 0]
    kids = _force_synthesis([s], "x")
    assert kids[0]["deps"] == []                   # stale, non-int and self-indices dropped
    assert kids[1]["tool"] == "llm" and kids[1]["deps"] == [0]


def test_grounding_uses_the_tools_declared_arg_template():
    # a non-search gather tool declares how to ground a bare subtask ("ground")
    tools = {"db_query": {"required": ["table"], "ground": {"table": "metrics",
                                                            "q": "{intent}"}}}
    out = _force_synthesis([{"intent": "uptime last week", "tool": "llm", "deps": []}],
                           "x", gather=("db_query",), tools=tools)
    assert out[0]["tool"] == "db_query"
    assert out[0]["args"] == {"table": "metrics", "q": "uptime last week"}


def test_llm_down_releases_without_spending_repairs(monkeypatch):
    """A wedged runner is not a node failure: the agent releases the node (no
    report), and only posts FAILED once the node's attempts are exhausted."""
    import httpx

    import harness.agent as ag

    class DownLlm:
        calls = 0

        def chat_json(self, *a, **k):
            self.calls += 1
            raise httpx.ReadTimeout("wedged")

    class _R:
        def json(self):
            return {"nodes": {}}

    class Http:
        def __init__(self):
            self.posts = []

        def get(self, path):
            return _R()

        def post(self, path, json=None):
            self.posts.append((path, json))

            class R2:
                status_code = 200

                def json(self):
                    return {}

            return R2()

    monkeypatch.setattr(ag, "POLL", 0)
    for attempts, expect_report in ((1, False), (3, True)):
        a = object.__new__(Agent)
        a.llm, a.http, a.phase = DownLlm(), Http(), "x"
        a.handle({"run_id": "r", "inputs": {}, "max_depth": 17,
                  "tools": {"web_search": {"required": ["query", "max_results"]}},
                  "constraints": [], "context": {}, "prompts": {}, "params": {},
                  "node": {"id": "root", "intent": "t", "tool": None, "depth": 0,
                           "attempts": attempts, "hint": ""}})
        results = [p for p, _ in a.http.posts if p.endswith("/result")]
        assert bool(results) is expect_report
        if expect_report:
            hint = a.http.posts[-1][1]["hint"]
            assert "llm unreachable" in hint


def test_judge_prefers_the_tools_deterministic_check():
    a = object.__new__(Agent)

    class Boom:
        def chat_json(self, *a, **k):
            raise AssertionError("checked tools must not call the LLM judge")

    a.llm = Boom()
    node = {"tool": "create_employee", "intent": "hire", "success_predicate": "hired"}
    tools = {"create_employee": {"check": "employee.id"}}
    assert a.judge(node, {"employee": {"id": 7}}, tools=tools) is True
    assert a.judge(node, {"employee": {}}, tools=tools) is False


# ---------- per-run prompts/params honored by the planner and judge ----------
def test_decompose_honors_cap_custom_sink_and_max_results():
    captured = {}

    class FakeLlm:
        def chat_json(self, sys_p, user_p, **k):
            captured["sys"] = sys_p
            return {"children": [
                {"intent": "g1", "tool": "web_search", "args": {"query": "a"}, "deps": []},
                {"intent": "g2", "tool": "web_search", "args": {"query": "b"}, "deps": []},
                {"intent": "g3", "tool": "web_search", "args": {"query": "c"}, "deps": []}]}

    a = object.__new__(Agent)
    a.llm = FakeLlm()
    out = a.decompose({"id": "root.1", "intent": "t", "depth": 1, "hint": ""}, {},
                      {"web_search": {"required": ["query", "max_results"]}}, 17,
                      [], None, None,
                      {"child_cap": 2, "sink_tool": "condense", "max_results": 5})
    kids = out["children"]
    assert [c["tool"] for c in kids] == ["web_search", "condense"]   # cap leaves a slot, sink pinned
    assert kids[0]["args"]["max_results"] == 5          # per-run bound injected
    assert kids[1]["deps"] == [0] and kids[1]["args"] == {}
    assert 'Use tool "condense" (no args)' in captured["sys"]        # model told the sink name


def test_decompose_temperature_and_model_passed_to_llm():
    seen = {}

    class FakeLlm:
        def chat_json(self, sys_p, user_p, **k):
            seen.update(k)
            return {"children": [{"intent": "s", "tool": "web_search",
                                  "args": {"query": "q"}, "deps": []}]}

    a = object.__new__(Agent)
    a.llm = FakeLlm()
    a.decompose({"id": "root", "intent": "t", "depth": 0, "hint": ""}, {},
                {"web_search": {"required": ["query"]}}, 17, [], None, None,
                {"model": "qwen3:8b", "temperature": 0.7})
    assert seen["model"] == "qwen3:8b" and seen["temperature"] == 0.7


def test_judge_uses_run_gather_classification_and_prompt():
    class RecLlm:
        def __init__(self):
            self.seen = None

        def chat_json(self, sys_p, user_p, **k):
            self.seen = sys_p
            return {"ok": True}

    a = object.__new__(Agent)
    a.llm = RecLlm()
    node = {"tool": "http_fetch", "success_predicate": "ok", "intent": "x"}
    # http_fetch declassified as a gather tool -> the LLM judge must run
    assert a.judge(node, {"results": [1]},
                   prompts={"judge": "CUSTOM JUDGE"},
                   params={"gather_tools": ["web_search"]})
    assert a.llm.seen == "CUSTOM JUDGE"

    class BoomLlm:
        def chat_json(self, *a, **k):
            raise AssertionError("gather results judged deterministically")

    b = object.__new__(Agent)
    b.llm = BoomLlm()
    assert b.judge(node, {"results": [1]})              # default: http_fetch IS a gather tool


def test_prompts_module_resolvers():
    import pytest

    from harness.prompts import (
        DEFAULT_PARAMS,
        DEFAULT_PROMPTS,
        resolve_params,
        resolve_prompt,
    )
    assert resolve_prompt("judge") == DEFAULT_PROMPTS["judge"]
    assert resolve_prompt("judge", {"judge": "X"}) == "X"
    assert resolve_prompt("judge", {"judge": "  " if False else ""}) == DEFAULT_PROMPTS["judge"]
    p = resolve_params({"temperature": 5, "child_cap": 0, "unknown": 1,
                        "gather_tools": ["shell"], "max_results": 99})
    assert p["temperature"] == 1 and p["child_cap"] == 1 and p["max_results"] == 10
    assert p["gather_tools"] == ["shell"] and p["model"] == DEFAULT_PARAMS["model"]
    assert "unknown" not in p
    with pytest.raises(KeyError):
        resolve_prompt("nope")


# ---------- named-parts coverage top-up (code, not prompt) ----------
def test_decompose_tops_up_missing_named_branches_and_repins_sink():
    class SparseLlm:
        def chat_json(self, sys_p, user_p, **k):
            return {"children": [
                {"intent": "write the abstract", "tool": None, "args": {}, "deps": []},
                {"intent": "write the introduction", "tool": None, "args": {}, "deps": []}]}

    a = object.__new__(Agent)
    a.llm = SparseLlm()
    out = a.decompose({"id": "root", "intent": "white paper on dark matter with an "
                       "abstract, introduction, and conclusion", "depth": 0,
                       "hint": ""}, {}, {"web_search": {"required": ["query", "max_results"]}}, 4)
    intents = " | ".join(c["intent"].lower() for c in out["children"])
    assert "conclusion" in intents                        # missing branch appended in code
    sinks = [c for c in out["children"] if c["tool"] == "llm"]
    assert len(sinks) == 1 and sinks[0]["deps"] == [0, 1, 2]   # convergence restored


def test_decompose_leaves_tasks_without_named_parts_alone():
    class TwoLlm:
        def chat_json(self, sys_p, user_p, **k):
            return {"children": [
                {"intent": "g", "tool": "web_search", "args": {"query": "q"},
                 "deps": [], "success_predicate": "hits"}]}

    a = object.__new__(Agent)
    a.llm = TwoLlm()
    out = a.decompose({"id": "root", "intent": "tell me about pruning",
                       "depth": 0, "hint": ""}, {}, {"web_search": {"required": ["query", "max_results"]}}, 4)
    assert len(out["children"]) == 2                      # sink pinned, nothing invented


class _PaperRunHttp:
    def get(self, path):
        class R:
            def json(self):
                return {"nodes": {
                    "root":   {"intent": "white paper on dark matter with an abstract, "
                                         "introduction, and conclusion"},
                    "root.1": {"intent": "write the abstract"},
                    "root.2": {"intent": "write the introduction"},
                }}
        return R()


def test_head_of_named_parts_run_assembles_a_real_paper_shape():
    a = object.__new__(Agent)
    a.http = _PaperRunHttp()
    a.llm = _WriterLlm()
    inputs = {"root.1": {"text": "abstract body."},
              "root.2": {"text": "introduction body."}}
    t = a._write_answer("r", {"id": "root.4", "parent": "root",
                              "intent": "write the conclusion"},
                        inputs, "MATERIAL", [])
    assert len(a.llm.replies) == 1              # no generated intro duplicating the abstract
    assert "conclusion" in a.llm.replies[0].lower()   # the outro ask names the sink's part
    assert "## write the abstract\n\nabstract body." in t
    assert "## Conclusion\n\nBOTTOM LINE SENTENCES." in t   # sink IS a named part
    assert "## Bottom line" not in t


# ---------- dynamic-by-default: computed prompts + self-generated task guide ----------
def test_default_prompts_carry_no_domain_text():
    from harness.prompts import DEFAULT_PROMPTS
    joined = " ".join(DEFAULT_PROMPTS.values()).lower()
    for word in ("pruning", "research", "web_search", "page_text", "astrophys"):
        assert word not in joined          # no baked-in domain vocabulary


def test_decompose_cap_is_substituted_into_prompt():
    seen = {}

    class FakeLlm:
        def chat_json(self, sys_p, user_p, **k):
            seen["sys_p"] = sys_p
            return {"children": [{"intent": "s", "tool": "web_search",
                                  "args": {"query": "q"}, "deps": []}]}

    a = object.__new__(Agent)
    a.llm = FakeLlm()
    a.decompose({"id": "root", "intent": "t", "depth": 0, "hint": ""}, {},
                {"web_search": {"required": ["query"]}}, 17, [], None, None,
                {"child_cap": 2})
    assert "at most 2 smaller" in seen["sys_p"] and "{cap}" not in seen["sys_p"]


def test_synthesis_grounding_clause_only_for_url_tools():
    a = object.__new__(Agent)
    a.llm = _WriterLlm()
    a._write_answer("r", {"id": "root.1.2", "parent": "root.1", "intent": "t"},
                    {"root.1.1": {"text": "x"}}, "MATERIAL", [],
                    tools={"web_search": {"required": ["query"]}})
    assert "Cite the URL" in a.llm.systems[-1]
    a2 = object.__new__(Agent)
    a2.llm = _WriterLlm()
    a2._write_answer("r", {"id": "root.1.2", "parent": "root.1", "intent": "t"},
                     {"root.1.1": {"text": "x"}}, "MATERIAL", [],
                     tools={"shell": {"required": ["cmd"]}})
    assert "Cite the URL" not in a2.llm.systems[-1]


class _GuideHttp:
    def __init__(self):
        self.posts = []

    def post(self, path, json=None):
        self.posts.append((path, json))

        class R:
            status_code = 200

            def json(self):
                return {"ok": True}
        return R()


class _GuideLlm:
    calls = 0

    def chat(self, sys_p, user_p, **kw):
        type(self).calls += 1
        assert "black holes" in sys_p
        return " Sections: overview, evidence. Evidence: papers and measurements. "

    def chat_json(self, *a, **k):
        raise AssertionError("not reached in this test")


def test_ensure_guide_generates_once_at_root_split():
    a = object.__new__(Agent)
    a.http, a.llm = _GuideHttp(), _GuideLlm()
    pa = {"model": "", "temperature": 0.2}
    claim = {"run_id": "r", "node": {"depth": 0}, "constraints": [],
             "context": {"run_intent": "white paper on black holes"}, "prompts": {}}
    a._ensure_guide(claim, pa)
    assert a.http.posts == [("/runs/r/guide",
                             {"text": "Sections: overview, evidence. Evidence: papers and measurements."})]
    a._ensure_guide(claim, pa)                        # http-side dedupe not needed but
    assert len(a.http.posts) == 2                     # generate-gate is constraint-based:
    claim["constraints"] = ["guide: already there"]
    a._ensure_guide(claim, pa)
    assert len(a.http.posts) == 2                     # existing guide -> no new call
    deep = dict(claim, node={"depth": 2}, constraints=[])
    a._ensure_guide(deep, pa)
    assert len(a.http.posts) == 2                     # only the root split generates


# ---------- anti-recursion guards (small models loop 'split into pieces') ----------
def test_decompose_example_is_built_from_the_runs_tools():
    seen = {}

    class FakeLlm:
        def chat_json(self, sys_p, user_p, **k):
            seen["sys_p"] = sys_p
            return {"children": [{"intent": "s", "tool": "shell",
                                  "args": {"cmd": "x"}, "deps": []}]}

    a = object.__new__(Agent)
    a.llm = FakeLlm()
    a.decompose({"id": "root", "intent": "t", "depth": 0, "hint": ""}, {},
                {"shell": {"required": ["cmd"]}}, 17)
    assert '"tool":"shell"' in seen["sys_p"] and '"cmd":"..."' in seen["sys_p"]
    assert "web_search" not in seen["sys_p"]        # no static example leaking another domain


def test_abstract_only_children_ground_into_leaves_below_depth_1():
    kids = _force_synthesis(
        [{"intent": "define black hole properties", "tool": None, "args": {}, "deps": []},
         {"intent": "observational evidence", "tool": None, "args": {}, "deps": []}],
        "background section", GATHER, SINK, depth=2)
    assert kids[0]["tool"] == GATHER[0] and kids[0]["args"]["query"].startswith("define")
    assert kids[0]["deps"] == [] and kids[1]["tool"] == GATHER[0]
    assert kids[2]["tool"] == SINK and kids[2]["deps"] == [0, 1]
    intact = _force_synthesis(
        [{"intent": "piece one", "tool": None, "args": {}, "deps": []},
         {"intent": "piece two", "tool": None, "args": {}, "deps": []}],
        "section", GATHER, SINK, depth=1)
    assert intact[0]["tool"] is None                  # sections keep nesting above depth 2
    assert intact[2]["tool"] == SINK


def test_null_sink_with_deps_grounds_into_writer_not_deadlock():
    # the 9ea37d2e failure: model wrote the convergence sink with tool:"null"
    kids = _force_synthesis(
        [{"intent": "branch a", "tool": None, "args": {}, "deps": []},
         {"intent": "branch b", "tool": "null", "args": {}, "deps": []},
         {"intent": "write the answer", "tool": "null", "args": {}, "deps": [0, 1]}],
        "learn about x", GATHER, SINK)
    assert kids[0]["tool"] is None and kids[1]["tool"] is None
    assert kids[2]["tool"] == SINK and kids[2]["deps"] == [0, 1]  # writer, not abstract
    assert kids[2]["success_predicate"]                            # core validation passes


def test_approval_wait_times_out_after_deadline(monkeypatch):
    """An unanswered approval card must free the agent (rc -3) instead of
    polling forever while the heartbeat keeps the node alive."""
    class NeverDecide:
        def get(self, _url):
            class R:
                def json(self):
                    return {"status": "pending"}
            return R()

    monkeypatch.setattr("harness.agent.APPROVAL_TIMEOUT", 0.0)
    a = object.__new__(Agent)
    a.http = NeverDecide()
    out = a._wait("card-1")
    assert out["rc"] == -3 and "unanswered" in out["stderr"]
