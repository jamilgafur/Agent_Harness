"""The audit demo (examples/audit_run.py) is CI-verified: offline mode must run
a full intent run — decompose -> human checkpoint -> exec cards -> grounded
answer — and pass every audit invariant deterministically."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import audit_run  # noqa: E402


def test_offline_audit_passes_every_invariant():
    report = audit_run.run_audit()
    assert report["settled"] is True            # every node DONE
    assert report["nodes"] == 8                 # 2-branch tree, delegation depth 2
    assert report["max_depth"] == 2
    assert report["repairs"] == 1               # scripted flake -> repair loop recovered
    assert report["first_pass"] == "7/8"        # honest: one node needed a second attempt
    # grounding: every URL in the head answer is a URL the tree actually gathered
    assert report["grounding"]["ok"] is True
    assert len(report["grounding"]["cited"]) == 4
    assert report["grounding"]["dangling"] == []
    # human-in-the-loop ledger: plan checkpoint + 3 exec cards, one edited
    assert report["human"]["count"] == 4
    kinds = [d.get("kind") for d in report["human"]["decisions"]]
    assert kinds.count("exec") == 3 and "plan" in kinds
    assert sum(1 for d in report["human"]["decisions"] if d.get("edited")) == 1
    # audit trail survives a simulated restart
    assert report["persistence"]["states_match"] is True
    assert report["persistence"]["events"] > 0
    # budget ledger: the trail's accounted calls vs observed model calls
    assert report["budget"]["llm_calls"] > 0
    assert report["budget"]["model_calls_observed"] >= report["budget"]["llm_calls"]


def test_audit_render_names_every_check_and_the_quirk():
    report = audit_run.run_audit()
    text = audit_run.render(report)
    for heading in ("CHECKS", "HUMAN LEDGER", "GROUNDING", "BUDGET", "TRAIL"):
        assert heading in text
    assert "no invented citations" in text
    assert "ride no report" in text            # the accounting quirk is stated, not hidden
