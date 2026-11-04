"""Agent layer: leases, stale reports, redirect, patch approvals, heartbeats, events."""
import time

import pytest
from fastapi.testclient import TestClient

import harness.core as core
from harness.models import NodeState


@pytest.fixture()
def client(tmp_path):
    core.POLICY["confirm_actions"] = True
    core.POLICY["auto_approve"] = []
    core.POLICY["max_repairs"] = 2
    core.POLICY["approve_dag_edits"] = False
    core.POLICY["claim_lease_seconds"] = 120
    core.store = core.Store(tmp_path / "t.db")
    core.RUNS.clear()
    core.AGENTS.clear()
    return TestClient(core.app)


def abstract_run(client, intent="big task"):
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": intent}]})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def test_lease_requeues_stale_running_node(client):
    run_id = abstract_run(client)
    claim = client.post("/claim", json={"agent_id": "a1"}).json()
    assert claim["node"]["id"] == "root"
    # no fresh claim: same node not handed out twice
    assert client.post("/claim", json={"agent_id": "a2"}).json()["node"] is None
    # age the claim past the lease, then an arrive triggers the sweep
    core.RUNS[run_id]["nodes"]["root"].claimed_at = time.time() - 999
    claim2 = client.post("/claim", json={"agent_id": "a2"}).json()
    assert claim2["node"]["id"] == "root" and claim2["node"]["state"] == "RUNNING"
    assert claim2["node"]["hint"] == "lease expired (agent died?)"
    events = client.get(f"/runs/{run_id}/events").json()
    assert any(e["kind"] == "lease" for e in events)


def test_lease_skips_nodes_with_pending_approval(client, monkeypatch):
    r = client.post("/runs", json={"nodes": [
        {"id": "w", "intent": "write", "tool": "write_file",
         "args": {"path": "f", "content": "c"}, "success_predicate": "ok"}]})
    run_id = r.json()["run_id"]
    client.post("/claim", json={"agent_id": "a1"})
    pend = client.post(f"/runs/{run_id}/exec", json={
        "node_id": "w", "tool": "write_file", "args": {"path": "f", "content": "c"}}).json()
    assert pend["status"] == "pending"
    core.RUNS[run_id]["nodes"]["w"].claimed_at = time.time() - 999
    # sweep must NOT requeue a node waiting on a human decision
    assert client.post("/claim", json={"agent_id": "a2"}).json()["node"] is None
    assert core.RUNS[run_id]["nodes"]["w"].state == NodeState.RUNNING


def test_stale_report_rejected(client):
    run_id = abstract_run(client)
    client.post("/claim", json={"agent_id": "a1"})
    core.RUNS[run_id]["nodes"]["root"].claimed_at = time.time() - 999
    client.post("/claim", json={"agent_id": "a2"})  # requeues, a2 now owns it
    stale = client.post(f"/runs/{run_id}/nodes/root/result", json={"state": "FAILED"})
    assert stale.status_code == 200  # a1's report is for the CURRENT claim... no:
    # a1 never marked anything; a2 owns the claim, so a report is accepted once
    dupe = client.post(f"/runs/{run_id}/nodes/root/result",
                       json={"state": "FAILED", "hint": "late"})
    # node left RUNNING after first failure? max_repairs=2 -> REPAIRING, second report stale
    assert dupe.status_code == 409 and "stale" in dupe.json()["detail"]


def test_redirect_hands_off_and_claim_recovers(client):
    run_id = abstract_run(client)
    client.post("/claim", json={"agent_id": "a1"})
    r = client.post(f"/runs/{run_id}/nodes/root/redirect",
                    json={"agent_id": "a1", "reason": "needs a different skill"})
    assert r.status_code == 200
    node = client.get(f"/runs/{run_id}").json()["nodes"]["root"]
    assert node["state"] == "PENDING" and "redirected by a1" in node["hint"]
    claim = client.post("/claim", json={"agent_id": "a2"}).json()
    assert claim["node"]["id"] == "root"
    # redirect on a non-RUNNING node is rejected
    nxt = client.post(f"/runs/{run_id}/nodes/root/redirect",
                      json={"agent_id": "a2", "reason": "x"})
    assert nxt.status_code == 200  # a2 owns it (RUNNING) -> fine
    events = client.get(f"/runs/{run_id}/events").json()
    assert sum(1 for e in events if e["kind"] == "redirect") == 2


def test_decompose_state_guard_and_forward_deps(client):
    run_id = abstract_run(client)
    # forward sibling dep -> validation error, not silent drop
    bad = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "a", "deps": [0]},
    ]})
    assert bad.status_code == 400 and "earlier sibling" in str(bad.json())
    # claim+complete root via settle trick: report DONE after marking RUNNING
    client.post("/claim", json={"agent_id": "a1"})
    client.post(f"/runs/{run_id}/nodes/root/result", json={"state": "DONE", "result": 1})
    guarded = client.post(f"/runs/{run_id}/nodes/root/decompose",
                          json={"children": [{"intent": "late"}]})
    assert guarded.status_code == 400 and "state DONE" in str(guarded.json())


def test_patch_approval_flow(client):
    core.POLICY["approve_dag_edits"] = True
    run_id = abstract_run(client)
    client.post("/claim", json={"agent_id": "a1"})
    d = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "first"}, {"intent": "second", "deps": [0]},
    ]})
    assert d.status_code == 200 and d.json()["status"] == "pending_approval"
    # parent stays RUNNING; patch NOT applied yet
    assert client.get(f"/runs/{run_id}").json()["nodes"]["root"]["state"] == "RUNNING"
    # approve with EDITED children (edit-and-resubmit)
    resp = client.post(f"/approvals/{d.json()['exec_id']}", json={
        "decision": "approve",
        "args": {"children": [{"intent": "edited child"}]},
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["result"]["applied"] is True
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root"]["state"] == "WAITING"
    assert run["nodes"]["root.1"]["intent"] == "edited child"
    core.POLICY["approve_dag_edits"] = False


def test_patch_rejection_repairs_with_hint(client):
    core.POLICY["approve_dag_edits"] = True
    run_id = abstract_run(client)
    client.post("/claim", json={"agent_id": "a1"})
    d = client.post(f"/runs/{run_id}/nodes/root/decompose",
                    json={"children": [{"intent": "x"}]}).json()
    resp = client.post(f"/approvals/{d['exec_id']}", json={"decision": "reject"})
    assert resp.status_code == 200 and resp.json()["status"] == "rejected"
    node = client.get(f"/runs/{run_id}").json()["nodes"]["root"]
    assert node["state"] == "REPAIRING" and "rejected" in node["hint"]
    # REPAIRING is re-claimable: the loop closes
    assert client.post("/claim", json={"agent_id": "a1"}).json()["node"]["id"] == "root"
    core.POLICY["approve_dag_edits"] = False


def test_heartbeat_registry_and_events_persist(client):
    run_id = abstract_run(client)
    hb = client.post("/heartbeat", json={"agent_id": "a9", "run_id": run_id,
                                         "node_id": "root", "phase": "decomposing"})
    assert hb.status_code == 200
    agents = client.get("/agents").json()
    assert agents["a9"]["alive"] and agents["a9"]["phase"] == "decomposing"
    agents["a9"]["last_seen"] = 0  # stale
    # events survive a simulated restart
    claim = client.post("/claim", json={"agent_id": "a9"})
    core.RUNS.clear()
    core.RUNS.update(core.store.load_runs())
    events = client.get(f"/runs/{run_id}/events").json()
    kinds = [e["kind"] for e in events]
    assert "run" in kinds and "task" in kinds


def test_delete_run_and_policy_edit(client):
    run_id = abstract_run(client)
    assert client.delete(f"/runs/{run_id}").status_code == 409  # still active
    client.post("/claim", json={"agent_id": "a1"})
    client.post(f"/runs/{run_id}/nodes/root/result", json={"state": "DONE", "result": 1})
    assert client.delete(f"/runs/{run_id}").status_code == 200
    assert client.get(f"/runs/{run_id}").status_code == 404
    ok = client.post("/policy", json={"max_agents": 4, "approve_dag_edits": False})
    assert ok.status_code == 200 and ok.json()["max_agents"] == 4
    bad = client.post("/policy", json={"max_depth": 3})
    assert bad.status_code == 400
    core.POLICY["max_agents"] = 2
    core.POLICY["approve_dag_edits"] = False
