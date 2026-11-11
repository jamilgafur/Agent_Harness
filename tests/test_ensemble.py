"""Ensemble coordination: sampling temperatures, debate rounds, vote ballots,
and the agent wiring (writer, planner, judge panel). Off-by-default passthrough
is the hard requirement: samples=1/debate_rounds=0 must make one call, no vote.
"""
import json

from harness.agent import Agent
from harness.ensemble import Ensemble
from harness.prompts import resolve_params


class ScriptLlm:
    """Scripted LLM: chat() and chat_json() pop replies in order, every call is
    recorded as (kind, system, prompt, kwargs)."""

    def __init__(self, chat=(), chat_json=()):
        self.chat_replies = list(chat)
        self.json_replies = list(chat_json)
        self.calls = []

    def chat(self, system, prompt, **kw):
        self.calls.append(("chat", system, prompt, kw))
        return self.chat_replies.pop(0)

    def chat_json(self, system, prompt, **kw):
        self.calls.append(("chat_json", system, prompt, kw))
        return self.json_replies.pop(0)


# ---------- the protocol itself ----------
def test_off_passthrough_makes_exactly_one_call_no_vote():
    llm = ScriptLlm(chat=["the answer"])
    out = Ensemble(llm, "SYS").run("ASK", n=1, rounds=0)
    assert out == "the answer"
    assert [k for k, *_ in llm.calls] == ["chat"]          # no vote, no debate
    assert llm.calls[0][1] == "SYS" and llm.calls[0][2] == "ASK"


def test_later_samples_run_hotter_and_vote_picks_winner():
    llm = ScriptLlm(chat=["draft A", "draft B", "draft C"],
                    chat_json=[{"choice": 1, "why": "sharper"}])
    out = Ensemble(llm, "SYS", temperature=0.2).run("ASK", n=3, rounds=0)
    assert out == "draft B"
    temps = [kw["temperature"] for k, s, p, kw in llm.calls if k == "chat"]
    assert temps == [0.2, 0.5, 0.5]                        # first keeps run temp
    kind, sys_p, ballot, kw = llm.calls[-1]
    assert kind == "chat_json" and "Candidate 2" in ballot  # vote saw all three
    assert "draft A" in ballot and "draft C" in ballot


def test_debate_revises_each_candidate_against_its_rivals():
    llm = ScriptLlm(chat=["A1", "B1", "A2", "B2"], chat_json=[{"choice": 0}])
    ens = Ensemble(llm, "SYS")
    ens.run("ASK", n=2, rounds=1)
    proposal_prompts = [p for k, s, p, kw in llm.calls[:2]]
    revision_prompts = [p for k, s, p, kw in llm.calls[2:4]]
    assert all("Rival" not in p for p in proposal_prompts)  # independent drafts
    assert "B1" in revision_prompts[0] and "A1" in revision_prompts[1]
    assert "Rival answer" in revision_prompts[0]


def test_failed_revision_keeps_the_previous_candidate():
    llm = ScriptLlm(chat=["A1", "B1", "", "B2"], chat_json=[{"choice": 0}])
    Ensemble(llm, "SYS").run("ASK", n=2, rounds=1)
    ballot = llm.calls[-1][2]
    assert "A1" in ballot                                   # empty revision -> kept


def test_unusable_ballot_falls_back_to_candidate_zero():
    llm = ScriptLlm(chat=["A", "B"], chat_json=[None, {"choice": 9}])
    assert Ensemble(llm, "SYS").run("ASK", n=2, rounds=0) == "A"
    llm2 = ScriptLlm(chat=["A", "B"], chat_json=[{"choice": 9}])
    assert Ensemble(llm2, "SYS").run("ASK", n=2, rounds=0) == "A"


def test_all_samples_failing_yields_none_not_an_exception():
    llm = ScriptLlm(chat=["", "  "])
    assert Ensemble(llm, "SYS").run("ASK", n=2, rounds=0) is None


def test_structured_candidates_use_schema_and_json_repair_channel():
    p1, p2 = {"children": [{"intent": "a"}]}, {"children": [{"intent": "b"}]}
    llm = ScriptLlm(chat_json=[p1, p2, {"choice": 1}])
    schema = {"type": "object"}
    out = Ensemble(llm, "SYS").run("ASK", n=2, rounds=0, schema=schema)
    assert out == p2
    assert llm.calls[0][3]["schema"] is schema              # proposals via chat_json


# ---------- the knobs ----------
def test_ensemble_params_validate_and_clamp():
    p = resolve_params({"samples": 9, "debate_rounds": -3})
    assert p["samples"] == 5 and p["debate_rounds"] == 0
    assert resolve_params(None) == resolve_params({"samples": 1, "debate_rounds": 0})


# ---------- wiring: judge panel ----------
def test_judge_panel_is_strict_majority():
    class P(ScriptLlm):
        pass

    a = Agent.__new__(Agent)
    a.llm = P(chat_json=[{"ok": True}, {"ok": False}, {"ok": True}])
    # any tool without a deterministic path: not gather, not check-declared,
    # not shell (shell verdicts ride rc — a small model flunks stdout dumps)
    node = {"tool": "custom_exec", "intent": "x", "success_predicate": "did the thing"}
    assert a.judge(node, {"rc": 0}, params={"samples": 3}) is True

    b = Agent.__new__(Agent)
    b.llm = P(chat_json=[{"ok": False}, None, {"ok": True}])
    # 1 of 3 explicit true -> fails; None ballot counts as not-ok
    assert b.judge(node, {"rc": 0}, params={"samples": 3}) is False


# ---------- wiring: writer ----------
def _tool_node():
    return {"id": "root.1.2", "parent": "root.1", "intent": "t"}


def test_writer_samples_debates_and_votes_prose():
    llm = ScriptLlm(chat=["thin answer one. " * 4, "better answer two. " * 4],
                    chat_json=[{"choice": 1}])
    a = Agent.__new__(Agent)
    a.llm = llm
    t = a._write_answer("r", _tool_node(), {"root.1.1": {"text": "x"}},
                        "MATERIAL", [], params={"samples": 2})
    assert t.startswith("better answer two")


def test_writer_ensemble_codepath_with_samples_one_is_one_call():
    llm = ScriptLlm(chat=["the grounded answer. " * 4])
    a = Agent.__new__(Agent)
    a.llm = llm
    t = a._write_answer("r", _tool_node(), {"root.1.1": {"text": "x"}}, "MATERIAL", [])
    assert t == ("the grounded answer. " * 4).strip()
    assert [k for k, *_ in llm.calls] == ["chat"]           # unchanged call shape


# ---------- wiring: planner ----------
def _plan(tag):
    return {"children": [
        {"intent": f"search {tag}", "tool": "web_search",
         "args": {"query": f"q {tag}"}, "deps": [], "success_predicate": "hits"}]}


def test_decompose_samples_plans_and_votes_on_digests():
    llm = ScriptLlm(chat_json=[_plan("A"), _plan("B"), {"choice": 1, "why": "tighter"}])
    a = Agent.__new__(Agent)
    a.llm = llm
    out = a.decompose({"id": "root.1", "intent": "t", "depth": 1, "hint": ""}, {},
                      {"web_search": {"required": ["query", "max_results"]}}, 17,
                      [], None, None, {"samples": 2})
    kids = out["children"]
    assert kids[0]["args"]["query"] == "q B"                # voted plan won
    assert any(c["tool"] == "llm" for c in kids)            # sink pinned on the winner
    ballot = llm.calls[-1][2]
    assert "0. search A [web_search] deps:-" in ballot     # digest, not raw JSON
    assert json.dumps(_plan("B")) not in ballot


def test_decompose_debate_round_revises_raw_plans_before_shaping():
    llm = ScriptLlm(chat_json=[_plan("A"), _plan("B"), _plan("A2"), _plan("B2"),
                               {"choice": 0}])
    a = Agent.__new__(Agent)
    a.llm = llm
    out = a.decompose({"id": "root.1", "intent": "t", "depth": 1, "hint": ""}, {},
                      {"web_search": {"required": ["query", "max_results"]}}, 17,
                      [], None, None, {"samples": 2, "debate_rounds": 1})
    assert out["children"][0]["args"]["query"] == "q A2"
    revision_sys = llm.calls[2][1]
    assert "rivals" in revision_sys                         # debate persona appended


def test_decompose_all_ensemble_samples_dead_returns_none():
    llm = ScriptLlm(chat_json=[None, None])
    a = Agent.__new__(Agent)
    a.llm = llm
    assert a.decompose({"id": "root", "intent": "t", "depth": 0, "hint": ""}, {},
                       {"web_search": {"required": ["query"]}}, 17,
                       [], None, None, {"samples": 2}) is None
