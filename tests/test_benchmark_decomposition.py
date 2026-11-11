"""Benchmark harness (examples/benchmark_decomposition.py): the scoring and
report math are verified offline; the real numbers come from live runs."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import benchmark_decomposition as bench  # noqa: E402


def test_suite_is_scorable_and_invented():
    # every part must be findable in its own corpus — the eval can't ask for
    # something the material doesn't carry
    for task in bench.SUITE:
        assert len(task["parts"]) == 4
        for exp in task["parts"].values():
            assert exp.lower() in task["corpus"].lower(), (task["q"][:40], exp)


def test_score_counts_parts_case_insensitively():
    task = bench.SUITE[0]
    s = bench.score("1876 in Valdara, by Brandt, 1.2 m mirror", task["parts"])
    assert s["covered"] == 4 and s["total"] == 4
    s2 = bench.score("it was built long ago in Valdara", task["parts"])
    assert s2["covered"] == 1
    assert s2["hits"]["city"] and not s2["hits"]["year built"]


def test_report_renders_both_arms_and_tree_stats():
    rows = [
        {"arm": "flat", "task": 0, "covered": 3, "total": 4, "calls": 1, "seconds": 5.0},
        {"arm": "tree", "task": 0, "covered": 4, "total": 4, "calls": 10, "seconds": 60.0,
         "nodes": 13, "max_depth": 2, "settled": True, "grounded": True},
        {"arm": "flat", "task": 1, "covered": 4, "total": 4, "calls": 1, "seconds": 5.0},
        {"arm": "tree", "task": 1, "covered": 0, "total": 4, "calls": 9, "seconds": 70.0,
         "nodes": 5, "max_depth": 1, "settled": False, "grounded": True},
    ]
    out = bench.report(rows, "qwen2.5:3b")
    assert "| flat | 7/8 (87%) | 1/2 | 1.0 | 5.0 |" in out
    assert "| tree | 4/8 (50%) | 1/2 | 9.5 | 65.0 |" in out
    assert "9.0 nodes/task" in out and "delegation depth 2" in out
    assert "settled 1/2" in out and "grounded 2/2" in out
    # per-task matrix: reliability (settled) split from content quality (winner)
    assert "| 0 | 3/4 | 4/4 | yes | tree |" in out
    assert "| 1 | 4/4 | 0/4 | no | unsettled |" in out
    assert "settled tasks only (1/2): flat 3/4 vs tree 4/4" in out


def test_flat_arm_makes_exactly_one_call():
    class Rec:
        model = "fake"

        def __init__(self):
            self.calls = 0

        def chat(self, system, prompt, **kw):
            self.calls += 1
            return "1876 in Valdara by Brandt, 1.2"

    llm = Rec()
    row = bench.run_flat(llm, bench.SUITE[0])
    assert row["calls"] == 1 and llm.calls == 1
    assert row["covered"] == 4


def test_canned_sandbox_serves_corpus_through_any_retrieval_tool(monkeypatch):
    # regression: plans that grep a named file (shell) or read it must reach the
    # corpus like web_search plans do — tool choice is not the measured variable
    monkeypatch.setitem(bench._CORPUS, "text", "The flour is spelt flour.\n")
    monkeypatch.setattr(bench, "_FILES", {})
    run = lambda tool, args: asyncio.run(bench._canned_exec(tool, args))  # noqa: E731

    hits = run("web_search", {"query": "recipe"})
    assert hits["results"][0]["page_text"] == "The flour is spelt flour.\n"
    assert "spelt" in run("read_file", {"path": "recipe.txt"})["content"]

    out = run("shell", {"cmd": "grep spelt recipe.txt"})
    assert out["rc"] == 0 and "spelt flour" in out["stdout"]

    assert run("write_file", {"path": "notes/a.txt", "content": "yzella"})["bytes"] == len("yzella")
    assert "yzella" in run("shell", {"cmd": "cat a.txt"})["stdout"]
    assert "yzella" in run("read_file", {"path": "a.txt"})["content"]

    bad = run("boomerang", {"path": "x"})
    assert bad["rc"] == -1 and "not in scope" in bad["stderr"]
