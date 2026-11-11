"""The benchmark harness itself: grading, cost model, and reporting run offline
with a scripted LLM; the real numbers come from running the example against a
live model (examples/benchmark_ensemble.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import benchmark_ensemble as bench  # noqa: E402


class FakeLlm:
    """Answers 'Answer: 6' (wrong) except every third chat — our scripted
    ensemble must surface the right 'Answer: 5' via sampling; the vote ballot
    picks candidate 0 (fallback when ballots are unusable)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.model = "fake"

    def chat(self, system, prompt, **kw):
        self.calls += 1
        return self.replies.pop(0)

    def chat_json(self, system, prompt, **kw):
        self.calls += 1
        return {"choice": 0}


def test_extract_and_grade():
    assert bench.extract("steps…\nAnswer: $126.0") == "126.0"
    assert bench.extract("the total is 45 cats") == "45"
    assert bench.grade("Answer: 6.0", "6") and not bench.grade("Answer: 7", "6")
    assert bench.grade("half covered on day 29", "29")


def test_cost_model():
    assert bench.calls_model(1, 0) == 1
    assert bench.calls_model(3, 0) == 4
    assert bench.calls_model(3, 1) == 7


def test_run_suite_counts_calls_and_accuracy_per_task():
    tasks = [("q", "5")]
    # 3 drafts + ballot choice 0 -> the first (correct) draft wins
    replies = ["Answer: 5", "Answer: 6", "Answer: 6"]

    def make_llm():
        return FakeLlm(replies)

    rows = bench.run_suite(make_llm, tasks, n=3, rounds=0, temperature=0.2)
    assert rows[0]["ok"] is True and rows[0]["calls"] == 4  # 3 samples + 1 vote
    out = bench.report({"sampled": rows, "baseline": rows}, n=3, rounds=0)
    assert "| config |" in out and "DIFFERS FROM" in out    # baseline calls=4 lies, flagged


def test_dead_llm_call_becomes_a_failed_row_not_a_crash():
    class Dead(FakeLlm):
        def chat(self, system, prompt, **kw):
            self.calls += 1
            raise TimeoutError("wedged runner")

    rows = bench.run_suite(lambda: Dead([]), [("q", "5")], n=1, rounds=0,
                           temperature=0.2)
    assert rows[0]["ok"] is False and rows[0]["error"] == "TimeoutError"
    assert rows[0]["calls"] == 1
    out = bench.report({"baseline": rows}, n=3, rounds=0)
    assert "1 task(s) errored" in out                        # honest, visible


def test_report_marks_honest_runs_as_matching():
    tasks = [("q", "5")]
    replies = ["Answer: 5", "Answer: 6", "Answer: 6", "Answer: 5"]

    def make_llm():
        return FakeLlm(replies)

    results = {
        "baseline": bench.run_suite(make_llm, tasks, 1, 0, 0.2),
        "sampled": bench.run_suite(make_llm, tasks, 3, 0, 0.2),
    }
    out = bench.report(results, n=3, rounds=0)
    assert "matches the cost model" in out
    assert results["baseline"][0]["calls"] == 1


def test_report_handles_the_real_grid_names():
    # the live-run KeyError: main() names configs "sampled (samples=3)" etc.,
    # so the cost-model check must match by prefix, not exact key
    row = {"ok": True, "calls": 4, "seconds": 1.0, "expected": "5", "got": "5"}
    one = dict(row, calls=1)
    seven = dict(row, calls=7)
    out = bench.report({"baseline": [one],
                        "sampled (samples=3)": [row],
                        "debated (samples=3, rounds=1)": [seven]},
                       n=3, rounds=1)
    assert "matches the cost model" in out
