"""Agent-side material formatting + upward traversal for the virtual `llm` tool."""
from harness import core
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


def test_llm_ok_rejects_failure_cards_and_footer_padding():
    # a whole-answer "[Missing ...]" card is the no-material failure signal, not
    # an answer — accepting it wrote "[Missing ...]" chains into run heads
    assert not _llm_ok("[Missing specific recipe details and instructions]")
    assert not _llm_ok("(no material was gathered)")
    # a Sources footer must not pad a thin body past the length bar
    assert not _llm_ok("It was built in 1876.\n\nSources:\n- https://x.example/page")
    assert _llm_ok("Built in 1876 in Valdara by Brandt with a 1.2 m mirror."
                   "\n\nSources:\n- https://x.example/page")


def test_writer_retry_sees_why_it_was_rejected():
    """A REPAIRING writer node must carry the rejection into its material —
    an oblivious retry just repeats the same thin answer."""
    captured = {}

    class EchoLlm:
        calls = 0

        def chat(self, sys_p, user_p, **kw):
            captured["prompt"] = user_p
            return "A substantive grounded answer with specific facts about the topic."

    class Http:
        def __init__(self):
            self.posts = []

        def post(self, path, json=None):
            self.posts.append((path, json))

            class R:
                status_code = 200

                def json(self):
                    return {}

            return R()

    a = object.__new__(Agent)
    a.llm, a.http, a.phase = EchoLlm(), Http(), "x"
    a.handle({"run_id": "r", "inputs": {"root.1": {"text": "material here"}}, "max_depth": 17,
              "tools": {}, "constraints": [], "context": {}, "prompts": {}, "params": {},
              "node": {"id": "root.1.2", "parent": "root.1", "intent": "write about x",
                       "tool": "llm", "depth": 2, "attempts": 2,
                       "hint": "synthesis produced an empty/thin answer"}})
    assert "Previous attempt rejected: synthesis produced an empty/thin answer" \
        in captured["prompt"]


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


def test_tool_repair_rewrites_args_and_predicate_from_the_hint():
    """A repair retry must RE-DERIVE args: re-executing identical args can never
    fix a failed tool call (measured: three byte-identical attempts, escalation)."""
    captured = {}

    class FixLlm:
        calls = 0

        def chat_json(self, sys_p, ask, **kw):
            FixLlm.calls += 1
            captured["ask"] = ask
            return {"args": {"cmd": "cat f"},
                    "success_predicate": "output contains the record"}

    class Http:
        def __init__(self):
            self.posts = []

        def post(self, path, json=None):
            self.posts.append((path, json))

            class R:
                status_code = 200

                def json(self):
                    return {"status": "done", "result": {"rc": 0, "stdout": "the record"}}

            return R()

    a = object.__new__(Agent)
    a.llm, a.http, a.phase = FixLlm(), Http(), "x"
    a.handle({"run_id": "r", "inputs": {}, "max_depth": 17, "constraints": [],
              "context": {}, "prompts": {}, "params": {},
              "tools": {"shell": {"required": ["cmd"]}},
              "node": {"id": "root.1", "parent": "root", "intent": "show the record",
                       "tool": "shell", "depth": 1, "attempts": 1,
                       "args": {"cmd": "grep record f2"},       # the args that failed
                       "hint": "predicate unmet: no output",
                       "success_predicate": "shows"}})
    execs = [p for p in a.http.posts if p[0].endswith("/exec")]
    assert execs and execs[0][1]["args"] == {"cmd": "cat f"}   # rewritten, not replayed
    assert "predicate unmet: no output" in captured["ask"]     # the hint drives the fix
    assert FixLlm.calls == 1
    # first attempt (no hint) must not spend an LLM call on repair
    b = object.__new__(Agent)
    b.llm, b.http, b.phase = FixLlm(), Http(), "x"
    b.handle({"run_id": "r", "inputs": {}, "max_depth": 17, "constraints": [],
              "context": {}, "prompts": {}, "params": {},
              "tools": {"shell": {"required": ["cmd"]}},
              "node": {"id": "root.2", "parent": "root", "intent": "show",
                       "tool": "shell", "depth": 1, "attempts": 1,
                       "args": {"cmd": "cat f"}, "hint": "",
                       "success_predicate": "shows"}})
    assert FixLlm.calls == 1
    # a broken repair generation must not stall the node: args replay, work happens
    class BoomLlm:
        calls = 0

        def chat_json(self, *a, **k):
            raise RuntimeError("llm hiccup")

    c = object.__new__(Agent)
    c.llm, c.http, c.phase = BoomLlm(), Http(), "x"
    c.handle({"run_id": "r", "inputs": {}, "max_depth": 17, "constraints": [],
              "context": {}, "prompts": {}, "params": {},
              "tools": {"shell": {"required": ["cmd"]}},
              "node": {"id": "root.3", "parent": "root", "intent": "show",
                       "tool": "shell", "depth": 1, "attempts": 1,
                       "args": {"cmd": "cat f"}, "hint": "predicate unmet: x",
                       "success_predicate": "shows"}})
    execs = [p for p in c.http.posts if p[0].endswith("/exec")]
    assert execs and execs[0][1]["args"] == {"cmd": "cat f"}   # replayed, not dropped


def test_crud_tool_registry_checks_judge_without_the_llm():
    """write_file/read_file/list_dir carry a `check` result path, so state
    predicates ("the file contains …") are decided from the result deterministically
    — an LLM judge seeing {bytes: N} rejected TRULY written files as unmet."""
    class Boom:
        calls = 0

        def chat_json(self, *a, **k):
            Boom.calls += 1
            return {"ok": True}

    a = object.__new__(Agent)
    a.llm = Boom()
    node = {"tool": "write_file", "intent": "write f",
            "success_predicate": "the file contains the payload"}
    assert a.judge(node, {"bytes": 42}, {}, {}, core.TOOLS) is True
    assert a.judge(node, {"bytes": 0}, {}, {}, core.TOOLS) is False
    assert Boom.calls == 0
    assert core.TOOLS["write_file"].get("check") == "bytes"
    assert core.TOOLS["read_file"].get("check") == "content"
    assert core.TOOLS["list_dir"].get("check") == "entries"


def test_failed_shell_hint_carries_stderr_for_repair():
    """Repair re-derives args from the failure hint — if the hint drops stderr
    the model regenerates the same invented script every attempt (live T3:
    'python analyze_sales.py' x3 while 'can't open file' never surfaced)."""
    class Llm:
        calls = 0

        def chat_json(self, *a, **k):
            return {"ok": True}

    class Http:
        def __init__(self):
            self.posts = []

        def post(self, path, json=None):
            self.posts.append((path, json))

            class R:
                status_code = 200

                def json(self):
                    return {"status": "done",
                            "result": {"rc": 2,
                                       "stderr": "can't open file 'analyze_sales.py'"}}

            return R()

    a = object.__new__(Agent)
    a.llm, a.http, a.phase = Llm(), Http(), "x"
    a.handle({"run_id": "r", "inputs": {}, "max_depth": 17, "constraints": [],
              "context": {}, "prompts": {}, "params": {},
              "tools": {"shell": {"required": ["cmd"]}},
              "node": {"id": "root.1", "parent": "root", "intent": "total revenue",
                       "tool": "shell", "depth": 1, "attempts": 1,
                       "args": {"cmd": "python analyze_sales.py"},
                       "success_predicate": "prints the total"}})
    reports = [p for p in a.http.posts if p[0].endswith("/result")]
    assert reports and "can't open file 'analyze_sales.py'" in \
        str(reports[-1][1].get("hint", ""))


def test_write_payload_lands_the_code_not_the_analysis():
    """Live T2: the corrected file sat inside ```python fences while the write
    carried analysis prose; the judge refused it and the bug survived. A .py
    write must land as code — the text if it parses, else the largest fenced
    block that does. Prose without code stays untouched (the judge refuses it)."""
    from harness.agent.writing import _write_payload
    prose = ("To identify and correct the bugs, here is the analysis.\n\n"
             "**Updated Code:**\n```python\ndef mean(data):\n"
             "    return sum(data) / len(data)\n```\n\n"
             "The updated analyze.py should look like this:\n```python\n"
             "def mean(data):\n    return sum(data) / len(data)\n\n\n"
             "def median(data):\n    xs = sorted(data)\n    n = len(xs)\n"
             "    mid = n // 2\n"
             "    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2\n```\n")
    node = {"tool": "write_file",
            "args": {"path": "bugfix-lab/analyze.py",
                     "content": {"$from": "root.4.2", "path": "text"}}}
    out = _write_payload(node, {"root.4.2": {"text": prose}})
    content = out["args"]["content"]
    assert "def median" in content and "```" not in content   # the complete file
    assert "analysis" not in content
    # plain-text content is bound, and code that already parses is left alone
    good = {"tool": "write_file", "args": {"path": "a.py", "content": "x = 1\n"}}
    assert _write_payload(good, {})["args"]["content"] == "x = 1\n"
    # prose without any code block cannot be fixed up — left for the judge to refuse
    bad = {"tool": "write_file", "args": {"path": "a.py",
                                          "content": "updated code in a.py"}}
    assert _write_payload(bad, {})["args"]["content"] == "updated code in a.py"
    # non-code targets keep their prose (a report is prose)
    rep = {"tool": "write_file", "args": {"path": "r.md", "content": prose}}
    assert _write_payload(rep, {})["args"]["content"] == prose


def test_judge_never_passes_broken_python_writes():
    """A *.py write whose content is not valid Python must fail the judge even
    when bytes landed: live T2 shipped a status line over source and pytest died
    on it (rc=2) while the bytes check said DONE. Covers plain content and the
    $from write-back resolved from dep inputs."""
    a = object.__new__(Agent)
    node = {"tool": "write_file", "intent": "fix it", "success_predicate": "fixed",
            "args": {"path": "bugfix-lab/analyze.py",
                     "content": "updated code in bugfix-lab/analyze.py"}}
    assert a.judge(node, {"bytes": 36}, {}, {}, core.TOOLS) is False
    node["args"]["content"] = "def mean(v):\n    return sum(v) / len(v)\n"
    assert a.judge(node, {"bytes": 40}, {}, {}, core.TOOLS) is True
    node["args"]["content"] = "fixed_content"      # parses as a lone Expr — not a file
    assert a.judge(node, {"bytes": 13}, {}, {}, core.TOOLS) is False
    node["args"]["content"] = {"$from": "root.4", "path": "text"}
    assert a.judge(node, {"bytes": 40}, {}, {}, core.TOOLS,
                   inputs={"root.4": {"text": "the fix is done"}}) is False
    assert a.judge(node, {"bytes": 40}, {}, {}, core.TOOLS,
                   inputs={"root.4": {"text": "def mean(v):\n    return 0\n"}}) is True


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


def test_head_writer_carries_tool_output_sections_not_just_sink_text():
    """Live T5: every dep was a tool (shell stdout / read_file content), the
    assembly collected only sibling-sink TEXT, found zero sections, and the
    answer collapsed to the model's first sentence — 2 of 3 requested facts
    dropped though they sat verbatim in the dep results."""
    a = object.__new__(Agent)
    a.http = _RunHttp()
    a.llm = _WriterLlm()
    inputs = {
        "root.1": {"rc": 0, "stdout": "2031\nHalden Voss\n"},
        "root.2": {"rc": 0, "stdout": "2033\nPetra Lund\n"},
        "root.3": {"bytes": 12},                     # a write result: no text, skipped
    }
    t = a._write_answer("r", {"id": "root.4", "parent": "root", "intent": "sink"},
                        inputs, "MATERIAL", [])
    assert "## branch one intent\n\n2031\nHalden Voss" in t
    assert "## branch two intent\n\n2033\nPetra Lund" in t
    assert len(a.llm.replies) == 2                   # assembled, not re-generated


def test_synthesis_carries_the_failure_hint_to_the_author():
    """Error-driven refinement only works if the author SEES the runtime error
    of the file its previous output produced."""
    class Capturing(_WriterLlm):
        pass

    a = object.__new__(Agent)
    a.http = _RunHttp()
    a.llm = Capturing()
    t = a._write_answer("r", {"id": "root.1.3", "parent": "root.1",
                              "intent": "write the corrected contents of a.py",
                              "hint": "the file produced from your previous output "
                                      "failed at run time: ModuleNotFoundError: pandas"},
                        {"root.1.2": {"text": "material"}}, "MATERIAL", [])
    assert t == "GENERIC ANSWER."
    assert "ModuleNotFoundError: pandas" in a.llm.replies[0]


def test_branch_writer_and_empty_head_use_plain_synthesis():
    a = object.__new__(Agent)
    a.llm = _WriterLlm()          # replies beyond the first two => "GENERIC ANSWER."
    t = a._write_answer("r", {"id": "root.1.2", "parent": "root.1", "intent": "t"},
                        {"root.1.1": {"text": "x"}}, "MATERIAL", ["discuss cost"])
    assert t == "GENERIC ANSWER."
    assert "discuss cost" in a.llm.replies[0]

    a2 = object.__new__(Agent)
    a2.http = _RunHttp()
    a2.llm = _WriterLlm()
    # zero material -> the model's single chat call is all there is
    t2 = a2._write_answer("r", {"id": "root.4", "parent": "root", "intent": "t"},
                          {"root.1": {"bytes": 3}}, "MATERIAL", [])
    assert t2 == "GENERIC ANSWER."
    # ONE section still assembles verbatim: regenerating paraphrased away a
    # requested fact (live T5: the lone read_file dep held 'Petra Lund' and the
    # chat answer dropped it) — the funnel drops nothing
    a3 = object.__new__(Agent)
    a3.http = _RunHttp()
    a3.llm = _WriterLlm()
    t3 = a3._write_answer("r", {"id": "root.4", "parent": "root", "intent": "t"},
                          {"root.1": {"text": "only one section"}}, "MATERIAL", [])
    assert "## branch one intent\n\nonly one section" in t3
    assert t3.startswith("# the run question")


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
    assert [c["tool"] for c in kids] == ["web_search", "web_search", "llm"]  # sink last
    assert kids[1]["deps"] == []                   # back-edge to the sink removed
    assert kids[2]["deps"] == [0, 1]               # sink depends on every sibling


def test_ground_on_local_tasks_reads_the_named_file():
    """A task naming a local file must ground onto a path-taking gatherer —
    web queries like 'Extract total revenue' return nothing usable (live T3)."""
    from harness.agent.planning import _ground_on
    tools = {"web_search": {"required": ["query"], "ground": {"query": "{intent}"}},
             "read_file": {"required": ["path"], "ground": {"path": "{path}"}}}
    tool, args = _ground_on(tools, ("web_search",),
                            "Analyze the sales data in data-lab/sales.csv and "
                            "write a summary report to data-lab/report.md",
                            "Extract total revenue")
    assert tool == "read_file" and args == {"path": "data-lab/sales.csv"}
    tool, args = _ground_on(tools, ("web_search",), "tell me about pruning",
                            "find primary sources")
    assert tool == "web_search" and args == {"query": "find primary sources"}


def test_named_output_sees_deliverables_not_mutation_targets():
    from harness.agent.planning import _mutation_targets, _named_output
    assert _named_output("Analyze data-lab/sales.csv and write a summary "
                         "report to data-lab/report.md") == "data-lab/report.md"
    assert _named_output("fix the bugs in bugfix-lab/analyze.py") == ""   # mutation
    assert _mutation_targets("The tests are failing. Find and fix the bug(s) in "
                             "bugfix-lab/analyze.py so the suite passes.") \
        == ["bugfix-lab/analyze.py"]
    assert _mutation_targets("Analyze the sales data in data-lab/sales.csv and "
                             "write a report to data-lab/report.md.") == []
    assert _mutation_targets("show the correct output in results.txt") == []
    # 'contents of <path>' is only a deliverable after 'write' — summarizing or
    # reading a file's contents must never mark it for rewriting
    assert _named_output("write the corrected contents of bugfix-lab/analyze.py") \
        == "bugfix-lab/analyze.py"
    assert _named_output("summarize the contents of data-lab/sales.csv") == ""
    assert _named_output("read the contents of data-lab/sales.csv") == ""


def test_shape_children_adds_a_mutation_child_when_the_plan_has_none():
    """'Fix X' without a child that writes X never fixes anything (live T2: the
    bug survived because the plan only ran tests). The top-up child stays
    abstract — the sub-plan supplies the real corrected content."""
    from harness.prompts import resolve_params
    a = object.__new__(Agent)
    out = {"children": [
        {"intent": "run the tests", "tool": "shell",
         "args": {"cmd": "pytest bugfix-lab"}, "deps": [], "success_predicate": "ok"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0,
                                     "intent": "Find and fix the bug(s) in "
                                               "bugfix-lab/analyze.py"},
                               {"shell": {"required": ["cmd"]},
                                "write_file": {"required": ["path", "content"]}},
                               resolve_params({}))
    kids = shaped["children"]
    fixers = [c for c in kids if "bugfix-lab/analyze.py" in c["intent"]
              and c.get("tool") is None]
    assert len(fixers) == 1                    # abstract: content comes from a sub-plan
    # a READ of the path is not a write — live T2: the top-up was skipped because
    # a read_file child carried the path in its args and the bug survived
    out3 = {"children": [
        {"intent": "locate the failing test", "tool": "read_file",
         "args": {"path": "bugfix-lab/analyze.py"}, "deps": [],
         "success_predicate": "shows the file"},
        {"intent": "run the tests", "tool": "shell",
         "args": {"cmd": "pytest bugfix-lab"}, "deps": [], "success_predicate": "ok"}]}
    shaped3 = a._shape_children(out3, {"id": "root", "depth": 0,
                                       "intent": "Find and fix the bug(s) in "
                                                 "bugfix-lab/analyze.py"},
                                {"shell": {"required": ["cmd"]},
                                 "write_file": {"required": ["path", "content"]},
                                 "read_file": {"required": ["path"]}},
                                resolve_params({}))
    assert len([c for c in shaped3["children"]
                if c.get("tool") is None and "bugfix-lab/analyze.py" in c["intent"]]) == 1
    # a plan that already writes the file is left alone
    out2 = {"children": [
        {"intent": "fix it", "tool": "write_file",
         "args": {"path": "bugfix-lab/analyze.py", "content": "def mean(d): ..."},
         "deps": [], "success_predicate": "ok"}]}
    shaped2 = a._shape_children(out2, {"id": "root", "depth": 0,
                                       "intent": "Fix the bug(s) in bugfix-lab/analyze.py"},
                                {"shell": {"required": ["cmd"]},
                                 "write_file": {"required": ["path", "content"]}},
                                resolve_params({}))
    assert not [c for c in shaped2["children"] if c.get("tool") is None]


def test_shape_children_writes_the_named_output_from_the_sink():
    """An intent promising a file ('write a report to X') must plan the write:
    the sink's text becomes the file content via a $from binding (live T3 left
    the deliverable nonexistent while computing the answer)."""
    from harness.prompts import resolve_params
    a = object.__new__(Agent)
    out = {"children": [
        {"intent": "load the data", "tool": "read_file",
         "args": {"path": "data-lab/sales.csv"}, "deps": [], "success_predicate": "ok"},
        {"intent": "write the report", "tool": "llm", "deps": [0],
         "success_predicate": "covers"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0,
                                     "intent": "Analyze data-lab/sales.csv and write "
                                               "a summary report to data-lab/report.md"},
                               {"read_file": {"required": ["path"]},
                                "write_file": {"required": ["path", "content"]}},
                               resolve_params({}))
    kids = shaped["children"]
    writes = [c for c in kids if c.get("tool") == "write_file"]
    assert len(writes) == 1
    w = writes[0]
    assert w["args"]["path"] == "data-lab/report.md"
    assert w["args"]["content"] == {"$from": "root.2", "path": "text"}   # the sink
    assert w["deps"] == [1]                                             # after the sink


def test_shape_children_reads_a_named_input_file_whole():
    """A task naming a local input file needs a child that reads it whole: live
    T5 planned only 'grep X | awk' field guesses, the sink saw nothing but the
    guessed fields and answered without any of the facts."""
    from harness.prompts import resolve_params
    a = object.__new__(Agent)
    tools = {"shell": {"required": ["cmd"]}, "read_file": {"required": ["path"]},
             "write_file": {"required": ["path", "content"]}}
    out = {"children": [
        {"intent": "extract the year", "tool": "shell",
         "args": {"cmd": "grep Nettlebrook data-lab/facts.txt | awk '{print $4}'"},
         "deps": [], "success_predicate": "ok"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0,
                                     "intent": "Read data-lab/facts.txt and answer "
                                               "three things"}, tools, resolve_params({}))
    readers = [c for c in shaped["children"]
               if (c.get("args") or {}).get("path") == "data-lab/facts.txt"
               and c.get("tool") == "read_file"]
    assert len(readers) == 1
    # a plan that already reads the file is left alone...
    out2 = {"children": [
        {"intent": "read it", "tool": "read_file",
         "args": {"path": "data-lab/facts.txt"}, "deps": [], "success_predicate": "ok"}]}
    shaped2 = a._shape_children(out2, {"id": "root", "depth": 0,
                                       "intent": "Read data-lab/facts.txt"}, tools,
                                resolve_params({}))
    assert len([c for c in shaped2["children"] if c.get("tool") == "read_file"]) == 1
    # ...and the deliverable is never READ back (it does not exist yet) — its
    # write-back child still carries the path, that is the point of the task
    out3 = {"children": [
        {"intent": "compute", "tool": "shell", "args": {"cmd": "true"},
         "deps": [], "success_predicate": "ok"}]}
    shaped3 = a._shape_children(out3, {"id": "root", "depth": 0,
                                       "intent": "Analyze data-lab/sales.csv and write "
                                                 "a report to data-lab/report.md"},
                                tools, resolve_params({}))
    assert not [c for c in shaped3["children"]
                if c.get("tool") == "read_file"
                and (c.get("args") or {}).get("path") == "data-lab/report.md"]
    assert any(c.get("tool") == "write_file"
               and (c.get("args") or {}).get("path") == "data-lab/report.md"
               for c in shaped3["children"])


def test_fake_writes_are_dropped_and_top_ups_fire():
    """A write whose content is the task echo or an unfilled template is not
    content: live T2 wrote instruction prose over source (bug marker gone, file
    broken), live T3 wrote '{{total_revenue}}' into the deliverable. Dropped —
    so the mutation top-up and the sink-bound write-back plan real writes."""
    from harness.prompts import resolve_params
    a = object.__new__(Agent)
    tools = {"shell": {"required": ["cmd"]}, "read_file": {"required": ["path"]},
             "write_file": {"required": ["path", "content"]}}
    intent2 = ("Fix the identified bug(s) in bugfix-lab/analyze.py and update "
               "the file with the changes")
    out = {"children": [
        {"intent": intent2, "tool": "write_file",
         "args": {"path": "bugfix-lab/analyze.py", "content": intent2},
         "deps": [], "success_predicate": "ok"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0, "intent": intent2},
                               tools, resolve_params({}))
    kids = shaped["children"]
    assert not [c for c in kids if c.get("tool") == "write_file"]   # echo dropped
    assert any(c.get("tool") is None and "bugfix-lab/analyze.py" in c["intent"]
               for c in kids)                                       # fix top-up fires
    out2 = {"children": [
        {"intent": "Write summary report to report.md", "tool": "write_file",
         "args": {"path": "data-lab/report.md",
                  "content": "Total Revenue: {{total_revenue}}"},
         "deps": [], "success_predicate": "ok"}]}
    shaped2 = a._shape_children(out2, {"id": "root", "depth": 0,
                                       "intent": "Analyze data-lab/sales.csv and "
                                                 "write a report to data-lab/report.md"},
                                tools, resolve_params({}))
    kids2 = shaped2["children"]
    writes = [c for c in kids2 if c.get("tool") == "write_file"]
    assert len(writes) == 1                     # template gone, sink-bound write back
    assert writes[0]["args"]["content"] == {"$from": "root.2", "path": "text"}
    # a *.py write that is not valid Python is not content either (live T2 wrote
    # the status line 'updated code in bugfix-lab/analyze.py' over the source)
    intent3 = "Fix the bug in bugfix-lab/analyze.py"
    out3 = {"children": [
        {"intent": intent3, "tool": "write_file",
         "args": {"path": "bugfix-lab/analyze.py",
                  "content": "updated code in bugfix-lab/analyze.py"},
         "deps": [], "success_predicate": "ok"}]}
    shaped3 = a._shape_children(out3, {"id": "root", "depth": 0, "intent": intent3},
                                tools, resolve_params({}))
    assert not [c for c in shaped3["children"] if c.get("tool") == "write_file"]
    assert any(c.get("tool") is None and "bugfix-lab/analyze.py" in c["intent"]
               for c in shaped3["children"])
    # a bare placeholder word PARSES as Python (lone Expr) but is not a file —
    # live T2 shipped literally 'fixed_content' into analyze.py
    out4 = {"children": [
        {"intent": intent3, "tool": "write_file",
         "args": {"path": "bugfix-lab/analyze.py", "content": "fixed_content"},
         "deps": [], "success_predicate": "ok"}]}
    shaped4 = a._shape_children(out4, {"id": "root", "depth": 0, "intent": intent3},
                                tools, resolve_params({}))
    assert not [c for c in shaped4["children"] if c.get("tool") == "write_file"]
    assert any(c.get("tool") is None for c in shaped4["children"])


def test_invented_script_shell_splits_into_write_then_run():
    """A shell running a script nobody creates never runs (live T3: the model
    regenerated 'python analyze_sales.py' 3x per child while 'can't open file'
    stared it in the face). Split it: the child becomes the script write, the
    run is appended after it and depends on it."""
    from harness.prompts import resolve_params
    from harness.core import _build_children
    from harness.models import Node
    a = object.__new__(Agent)
    tools = {"shell": {"required": ["cmd"]}, "read_file": {"required": ["path"]},
             "write_file": {"required": ["path", "content"]}}
    out = {"children": [
        {"intent": "Extract total revenue", "tool": "shell",
         "args": {"cmd": "python analyze_sales.py --csv data-lab/sales.csv --total"},
         "deps": [], "success_predicate": "prints"},
        {"intent": "Identify top product", "tool": "shell",
         "args": {"cmd": "python analyze_sales.py --top"},
         "deps": [], "success_predicate": "prints"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0,
                                     "intent": "Analyze data-lab/sales.csv and "
                                               "write a report to data-lab/report.md"},
                               tools, resolve_params({}))
    kids = shaped["children"]
    writes = [c for c in kids if c.get("tool") is None
              and "analyze_sales.py" in c["intent"]]
    assert len(writes) == 1                      # ONE write, not one per runner
    w_i = kids.index(writes[0])
    runners = [c for c in kids if c.get("tool") == "shell"
               and "analyze_sales.py" in str((c.get("args") or {}).get("cmd", ""))]
    assert len(runners) == 2
    assert all(w_i in (c.get("deps") or []) for c in runners)   # run AFTER the write
    _, errs = _build_children({"root": Node(id="root", intent="t", depth=0)},
                              "root", kids)
    assert errs == []
    # a script a sibling already writes is left alone; -c/-m are not files
    out2 = {"children": [
        {"intent": "make the helper", "tool": "write_file",
         "args": {"path": "analyze_sales.py", "content": "x = 1\n"},
         "deps": [], "success_predicate": "ok"},
        {"intent": "run it", "tool": "shell",
         "args": {"cmd": "python analyze_sales.py"}, "deps": [0], "success_predicate": "ok"},
        {"intent": "inline", "tool": "shell",
         "args": {"cmd": "python -c 'print(1)'"}, "deps": [], "success_predicate": "ok"}]}
    shaped2 = a._shape_children(out2, {"id": "root", "depth": 0,
                                       "intent": "count things in data-lab/sales.csv"},
                                tools, resolve_params({}))
    assert [c for c in shaped2["children"] if c.get("tool") is None] == []
    # inside a file-producing task the split must NOT fire again — it recursed
    # to 35 nodes / depth 6 live (this level writes ONE file)
    out3 = {"children": [
        {"intent": "compute", "tool": "shell",
         "args": {"cmd": "python helper.py"}, "deps": [], "success_predicate": "ok"}]}
    shaped3 = a._shape_children(out3, {"id": "root.1", "depth": 1,
                                       "intent": "write the corrected contents of "
                                                 "analyze_sales.py (part of: totals)"},
                                tools, resolve_params({}))
    assert [c for c in shaped3["children"] if c.get("tool") is None] == []


def test_sink_intent_kept_for_deliverables_over_input_contents():
    """'write the contents of X to Y' is a report task: the sink's text goes to
    Y — it must NOT be reframed to 'corrected contents of X', which would mark
    the INPUT data for rewriting."""
    from harness.prompts import resolve_params
    a = object.__new__(Agent)
    tools = {"shell": {"required": ["cmd"]}, "read_file": {"required": ["path"]},
             "write_file": {"required": ["path", "content"]}}
    out = {"children": [
        {"intent": "compute", "tool": "shell", "args": {"cmd": "true"},
         "deps": [], "success_predicate": "ok"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0,
                                     "intent": "Write the contents of data-lab/"
                                               "sales.csv to data-lab/report.md"},
                               tools, resolve_params({}))
    sinks = [c for c in shaped["children"] if c.get("tool") == "llm"]
    assert "corrected contents" not in sinks[0]["intent"]


def test_shape_children_never_emits_forward_deps():
    """Top-ups append after the model's own sink; the sink's 'deps = all
    siblings' wiring must not point FORWARD or core rejects the whole plan
    (live T3: 'cannot decompose' x3 -> 1-node run). The sink moves last."""
    from harness.prompts import resolve_params
    from harness.core import _build_children
    from harness.models import Node
    a = object.__new__(Agent)
    tools = {"shell": {"required": ["cmd"]}, "read_file": {"required": ["path"]},
             "write_file": {"required": ["path", "content"]}}
    out = {"children": [
        {"intent": "compute", "tool": "shell", "args": {"cmd": "true"},
         "deps": [], "success_predicate": "ok"},
        {"intent": "report", "tool": "llm", "args": {}, "deps": [0],
         "success_predicate": "ok"}]}
    shaped = a._shape_children(out, {"id": "root", "depth": 0,
                                     "intent": "Analyze data-lab/sales.csv and write "
                                               "a report to data-lab/report.md"},
                               tools, resolve_params({}))
    kids = shaped["children"]
    sink_i = max(i for i, c in enumerate(kids) if c.get("tool") == "llm")
    assert sink_i == len(kids) - 2          # sink last-but-the-write-back
    assert all(d < sink_i for d in kids[sink_i]["deps"])
    _, errs = _build_children({"root": Node(id="root", intent="t", depth=0)},
                              "root", kids)
    assert errs == []


def test_mutation_sub_plan_gets_read_sink_write():
    """'write the corrected contents of X' (the mutation top-up's own child
    intent) must shape into read -> sink -> write: live T2's sub-plan ran a read
    only, never wrote, and the bug survived."""
    from harness.prompts import resolve_params
    a = object.__new__(Agent)
    tools = {"shell": {"required": ["cmd"]}, "read_file": {"required": ["path"]},
             "write_file": {"required": ["path", "content"]}}
    out = {"children": [
        {"intent": "retrieve the contents of bugfix-lab/analyze.py",
         "tool": "read_file", "args": {"path": "bugfix-lab/analyze.py"},
         "deps": [], "success_predicate": "ok"}]}
    shaped = a._shape_children(out, {"id": "root.5", "depth": 1,
                                     "intent": "write the corrected contents of "
                                               "bugfix-lab/analyze.py"},
                               tools, resolve_params({}))
    kids = shaped["children"]
    assert any(c.get("tool") == "read_file"
               and (c.get("args") or {}).get("path") == "bugfix-lab/analyze.py"
               for c in kids)                                  # material for the fix
    sinks = [i for i, c in enumerate(kids) if c.get("tool") == "llm"]
    assert len(sinks) == 1                                      # corrected content
    assert kids[sinks[0]]["intent"] == (
        "write the corrected contents of bugfix-lab/analyze.py "
        "(output the complete file)")   # the sink PRODUCES the file, not analysis
    writer = [c for c in kids if c.get("tool") == "write_file"]
    assert len(writer) == 1 and \
        writer[0]["args"]["path"] == "bugfix-lab/analyze.py"     # the fix lands
    assert writer[0]["args"]["content"] == {"$from": "root.5.2", "path": "text"}
    assert writer[0]["deps"] == [sinks[0]]


def test_repair_never_touches_from_bindings():
    """$from bindings are structural: the repair model mangles them (live T2:
    bind id 'root/analyze.py') and the exec is then rejected outright. Other
    args still re-derive."""
    captured = {}

    class FixLlm:
        calls = 0

        def chat_json(self, sys_p, ask, **kw):
            captured["ask"] = ask
            return {"args": {"content": {"$from": "root/analyze.py", "path": "text"},
                             "path": "other.py"},
                    "success_predicate": "written"}

    class Http:
        def __init__(self):
            self.posts = []

        def post(self, path, json=None):
            self.posts.append((path, json))

            class R:
                status_code = 200

                def json(self):
                    return {"status": "done", "result": {"bytes": 3}}

            return R()

    a = object.__new__(Agent)
    a.llm, a.http, a.phase = FixLlm(), Http(), "x"
    a.handle({"run_id": "r", "inputs": {}, "max_depth": 17, "constraints": [],
              "context": {}, "prompts": {}, "params": {},
              "tools": {"write_file": {"required": ["path", "content"]}},
              "node": {"id": "root.4.5", "parent": "root.4", "intent": "write it",
                       "tool": "write_file", "depth": 2, "attempts": 1,
                       "args": {"path": "bugfix-lab/analyze.py",
                                "content": {"$from": "root.4.4", "path": "text"}},
                       "hint": "predicate unmet: x",
                       "success_predicate": "ok"}})
    execs = [p for p in a.http.posts if p[0].endswith("/exec")]
    assert execs
    args = execs[0][1]["args"]
    assert args["content"] == {"$from": "root.4.4", "path": "text"}   # kept verbatim
    assert args["path"] == "other.py"                                 # re-derived


def test_force_synthesis_pins_root_sink_on_task_shapes():
    """A pure-task plan at the ROOT still converges on one writer — the run head
    must end holding the answer (root.result.text); deeper task splits stay lean."""
    kids = [{"intent": "write", "tool": "write_file",
             "args": {"path": "a", "content": "b"}, "deps": []}]
    out = _force_synthesis(kids, "do x")
    assert [c["tool"] for c in out] == ["write_file", "llm"]
    assert out[1]["deps"] == [0]
    assert _force_synthesis(kids, "do x", depth=1) == kids


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

    import harness.agent.base as ag

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


def test_shell_judge_is_deterministic_and_never_calls_the_llm():
    # a 3b judge flunks large stdout dumps: whole benchmark runs escalated over
    # correct greps judged "predicate unmet" — shell verdicts ride rc only
    class BoomLlm:
        def chat_json(self, *a, **k):
            raise AssertionError("the LLM must not judge shell results")

    a = object.__new__(Agent)
    a.llm = BoomLlm()
    node = {"tool": "shell", "success_predicate": "the response contains the height",
            "intent": "get the height"}
    assert a.judge(node, {"rc": 0, "stdout": "46 meters\n", "stderr": ""})
    assert not a.judge(node, {"rc": 2, "stdout": "", "stderr": "grep: no such file"})


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

    monkeypatch.setattr("harness.agent.base.APPROVAL_TIMEOUT", 0.0)
    a = object.__new__(Agent)
    a.http = NeverDecide()
    out = a._wait("card-1")
    assert out["rc"] == -3 and "unanswered" in out["stderr"]
