"""Core API: claim → exec (approval-gated) → report → cascade/retry.

The sandbox HTTP call is monkeypatched; approvals and the state machine are
exercised for real against a temp SQLite file.
"""
import pytest
from fastapi.testclient import TestClient

import harness.core as core


@pytest.fixture()
def client(monkeypatch, tmp_path):
    core.POLICY["confirm_actions"] = True
    core.POLICY["auto_approve"] = []
    core.POLICY["max_repairs"] = 2
    core.POLICY["approve_dag_edits"] = False
    core.POLICY["claim_lease_seconds"] = 120
    import os
    os.makedirs("/tmp/harness-test", exist_ok=True)
    core.store = core.Store(tmp_path / "t.db")
    core.RUNS.clear()
    core.RUNS.update(core.store.load_runs())
    core.AGENTS.clear()
    return TestClient(core.app)


def mk_template_run(client):
    r = client.post("/runs", json={"nodes": [
        {"id": "w", "intent": "write file", "tool": "write_file",
         "args": {"path": "a.txt", "content": "hi"}, "success_predicate": "file exists"},
        {"id": "r", "intent": "read back", "tool": "shell",
         "args": {"cmd": "cat a.txt"}, "deps": ["w"], "success_predicate": "prints hi"},
    ]})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def test_run_claim_exec_approve_flow(client, monkeypatch):
    run_id = mk_template_run(client)
    claim = client.post("/claim", json={"agent_id": "t"}).json()
    assert claim["node"]["id"] == "w" and claim["node"]["state"] == "RUNNING"

    # not auto-approved -> pending approval, no execution
    ex = client.post(f"/runs/{run_id}/exec", json={
        "node_id": "w", "tool": "write_file",
        "args": {"path": "a.txt", "content": "hi"}}).json()
    assert ex["status"] == "pending"
    pend = client.get("/approvals?status=pending").json()
    assert len(pend) == 1 and pend[0]["tool"] == "write_file"

    # user edits args then approves -> core calls the (mocked) sandbox
    async def fake_exec(tool, args):
        return {"bytes": len(args["content"])}
    monkeypatch.setattr(core, "_exec_tool", fake_exec)
    d = client.post(f"/approvals/{ex['exec_id']}",
                    json={"decision": "approve", "args": {"path": "b.txt", "content": "hii"}}).json()
    assert d["status"] == "done" and d["result"]["bytes"] == 3

    ok = client.post(f"/runs/{run_id}/nodes/w/result",
                     json={"state": "DONE", "result": d["result"]})
    assert ok.status_code == 200
    claim2 = client.post("/claim", json={"agent_id": "t"}).json()
    assert claim2["node"]["id"] == "r"  # dependency unblocked
    assert claim2["inputs"] == {"w": {"bytes": 3}}


def test_failure_repairs_then_escalates_and_cascades(client):
    run_id = mk_template_run(client)
    for i in range(2):  # max_repairs=2 -> second failure is terminal
        claim = client.post("/claim", json={"agent_id": "t"}).json()
        assert claim["node"]["id"] == "w"
        client.post(f"/runs/{run_id}/nodes/w/result",
                    json={"state": "FAILED", "hint": f"boom {i}"})
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["w"]["state"] == "ESCALATED"
    assert run["nodes"]["r"]["state"] == "BLOCKED"
    # claiming yields nothing now
    assert client.post("/claim", json={"agent_id": "t"}).json()["node"] is None
    # user fix: retry -> blocked descendant reopens
    ok = client.post(f"/runs/{run_id}/patch",
                     json={"ops": [{"op": "retry_node", "node_id": "w"}]})
    assert ok.status_code == 200, ok.text
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["w"]["state"] == "PENDING"
    assert run["nodes"]["r"]["state"] == "PENDING"


def test_decompose_validates_and_waits(client):
    r = client.post("/runs", json={"nodes": [
        {"id": "root", "intent": "big task"}]})
    run_id = r.json()["run_id"]
    # bad: unknown tool
    bad = client.post(f"/runs/{run_id}/nodes/root/decompose",
                      json={"children": [{"intent": "x", "tool": "nope",
                                          "success_predicate": "p"}]})
    assert bad.status_code == 400 and "unknown tool" in str(bad.json())
    # good: two children with a sibling dependency
    good = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "first", "tool": "write_file", "args": {"path": "f", "content": "c"},
         "success_predicate": "exists"},
        {"intent": "second", "tool": "shell", "args": {"cmd": "cat f"}, "deps": [0],
         "success_predicate": "prints"},
    ]})
    assert good.status_code == 200, good.text
    assert good.json()["children"] == ["root.1", "root.2"]
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root"]["state"] == "WAITING"
    assert run["nodes"]["root.2"]["deps"] == ["root.1"]
    # completing children completes the parent (bottom-up settle)
    for nid in ("root.1", "root.2"):
        client.post("/claim", json={"agent_id": "t"})
        client.post(f"/runs/{run_id}/nodes/{nid}/result", json={"state": "DONE", "result": 1})
    assert client.get(f"/runs/{run_id}").json()["nodes"]["root"]["state"] == "DONE"


def test_decompose_rejects_bad_arg_names(client):
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "big task"}]})
    run_id = r.json()["run_id"]
    bad = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "write", "tool": "write_file",
         "args": {"filename": "f", "text": "c"}, "success_predicate": "exists"},
    ]})
    assert bad.status_code == 400
    assert "missing required arg" in str(bad.json())
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root"]["state"] == "PENDING"  # rejected patch leaves no trace


def test_checkpoint_resume(tmp_path, client):
    run_id = mk_template_run(client)
    # simulate restart: reload from sqlite only
    core.RUNS.clear()
    core.RUNS.update(core.store.load_runs())
    run = client.get(f"/runs/{run_id}").json()
    assert run["id"] == run_id and set(run["nodes"]) == {"w", "r"}
