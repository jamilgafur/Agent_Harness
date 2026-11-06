"""Core API: claim → exec (approval-gated) → report → cascade/retry.

The sandbox HTTP call is monkeypatched; approvals and the state machine are
exercised for real against a temp SQLite file.
"""
import pytest
from fastapi.testclient import TestClient

import harness.core as core

_REAL_EXEC_TOOL = core._exec_tool   # captured pre-fixture; the fixture stubs the attr


@pytest.fixture()
def client(monkeypatch, tmp_path):
    core.POLICY["confirm_actions"] = True
    core.POLICY["auto_approve"] = []
    core.POLICY["max_repairs"] = 2
    core.POLICY["approve_dag_edits"] = False
    core.POLICY["claim_lease_seconds"] = 120
    core.POLICY["max_llm_calls"] = 500
    core.POLICY["max_seconds"] = 7200
    for k in [k for k, v in core.TOOLS.items() if v.get("dynamic")]:
        del core.TOOLS[k]              # isolate: /tools/sync side effects don't leak
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


def test_web_tools_registered_and_arg_validated(client):
    tools = client.get("/tools").json()["tools"]
    assert tools["web_search"]["required"] == ["query"]
    assert tools["http_fetch"]["required"] == ["url"]
    # template with a web_search node validates end-to-end
    ok = client.post("/runs", json={"nodes": [
        {"id": "s", "intent": "search", "tool": "web_search",
         "args": {"query": "q"}, "success_predicate": "has results"}]})
    assert ok.status_code == 200, ok.text
    # LLM invents arg names -> repairable validation error
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "big"}]})
    bad = client.post(f"/runs/{r.json()['run_id']}/nodes/root/decompose",
                      json={"children": [{"intent": "s", "tool": "web_search",
                                          "args": {"term": "q"},
                                          "success_predicate": "p"}]})
    assert bad.status_code == 400 and "missing required arg 'query'" in str(bad.json())


def test_workspace_proxy(client, monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"entries": ["paper.md", "refs/"]}

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, json=None): return R()

    monkeypatch.setattr(core.httpx, "AsyncClient", FakeClient)
    assert client.get("/workspace/files").json()["entries"] == ["paper.md", "refs/"]
    assert client.get("/workspace/file?path=paper.md").json()["entries"] is not None


def mk_loop_run(client, max_iterations=2):
    r = client.post("/runs", json={"nodes": [
        {"id": "L", "type": "loop", "intent": "refine the draft",
         "tool": "write_file", "args": {"path": "d.txt", "content": "v1"},
         "condition": "draft reads well", "max_iterations": max_iterations}]})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def test_loop_iterates_bounded_then_escalates(client):
    run_id = mk_loop_run(client, max_iterations=2)
    for i in range(1, 3):
        claim = client.post("/claim", json={"agent_id": "t"}).json()
        assert claim["node"]["id"] == "L"     # requeued after each unmet condition
        r = client.post(f"/runs/{run_id}/nodes/L/result",
                        json={"state": "FAILED", "hint": f"still rough {i}",
                              "result": {"v": i}})
        assert r.status_code == 200, r.text
        node = client.get(f"/runs/{run_id}").json()["nodes"]["L"]
        if i < 2:
            assert node["state"] == "PENDING" and node["attempts"] == i
    node = client.get(f"/runs/{run_id}").json()["nodes"]["L"]
    assert node["state"] == "ESCALATED" and node["attempts"] == 2
    # loop exhausted: never offered again (infinite-loop fallback)
    assert client.post("/claim", json={"agent_id": "t"}).json()["node"] is None


def test_loop_done_first_iteration(client):
    run_id = mk_loop_run(client, max_iterations=5)
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/L/result", json={"state": "DONE", "result": {"ok": 1}})
    node = client.get(f"/runs/{run_id}").json()["nodes"]["L"]
    assert node["state"] == "DONE" and node["attempts"] == 1


def test_loop_validation(client):
    r = client.post("/runs", json={"nodes": [
        {"id": "L", "type": "loop", "intent": "x", "tool": "shell",
         "args": {"cmd": "true"}, "max_iterations": 3}]})
    assert r.status_code == 400 and "condition" in str(r.json())
    r = client.post("/runs", json={"nodes": [
        {"id": "L", "type": "loop", "intent": "x", "condition": "c",
         "max_iterations": 3}]})
    assert r.status_code == 400 and "needs a tool" in str(r.json())


def test_budget_governor_llm_calls(client):
    core.POLICY["max_llm_calls"] = 3
    run_id = mk_template_run(client)
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/w/result",
                json={"state": "DONE", "result": 1, "llm_calls": 3})
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["w"]["state"] == "DONE"        # settled work is kept
    assert run["nodes"]["r"]["state"] == "ESCALATED"   # remaining work is stopped
    assert run["nodes"]["r"]["hint"].startswith("budget governor")
    assert run["active"] is False
    assert client.post("/claim", json={"agent_id": "t"}).json()["node"] is None


def test_budget_governor_time(client):
    core.POLICY["max_seconds"] = 0       # immediately over budget
    run_id = mk_template_run(client)
    assert client.post("/claim", json={"agent_id": "t"}).json()["node"] is None
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["w"]["state"] == "ESCALATED"
    assert "max_seconds" in run["nodes"]["w"]["hint"]


def test_budget_governor_decompose(client):
    core.POLICY["max_llm_calls"] = 0
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "big task"}]})
    run_id = r.json()["run_id"]
    r = client.post(f"/runs/{run_id}/nodes/root/decompose",
                    json={"children": [{"intent": "x", "tool": "shell",
                                        "args": {"cmd": "true"}, "success_predicate": "p"}]})
    assert r.status_code == 429 and "budget governor" in str(r.json())
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root"]["state"] == "ESCALATED" and run["active"] is False


def test_claim_parallel_ready_set(client):
    r = client.post("/runs", json={"nodes": [
        {"id": x, "intent": f"task {x}", "tool": "write_file",
         "args": {"path": x, "content": "1"}, "success_predicate": "done"}
        for x in ("a", "b", "c")]})
    run_id = r.json()["run_id"]
    claim = client.post("/claim", json={"agent_id": "t", "max_nodes": 2}).json()
    ids = {n["id"] for n in claim["nodes"]}
    assert len(ids) == 2 and claim["node"]["id"] in ids     # back-compat "node" = first
    assert all(n["state"] == "RUNNING" for n in claim["nodes"])
    claim2 = client.post("/claim", json={"agent_id": "u"}).json()
    remaining = {"a", "b", "c"} - ids
    assert claim2["node"]["id"] == remaining.pop()
    assert client.post("/claim", json={"agent_id": "v"}).json()["node"] is None


def test_dynamic_tools_sync_and_exec_routing(client, monkeypatch):
    calls = []

    class R:
        def raise_for_status(self): pass
        def json(self):
            return {"tools": {"shout": {"file": "shout.py",
                                        "desc": "uppercase", "required": ["text"]}}} \
                if calls and calls[-1][0].endswith("/discover_tools") else {"output": "HI"}

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): calls.append((url, None)); return R()
        async def post(self, url, json=None): calls.append((url, json)); return R()

    monkeypatch.setattr(core.httpx, "AsyncClient", FakeClient)
    # sync pulls sandbox dynamic tools into the registry
    out = client.post("/tools/sync").json()
    assert "shout" in out["synced"]
    assert client.get("/tools").json()["tools"]["shout"]["required"] == ["text"]
    # template can now use it; validation enforces its required args
    bad = client.post("/runs", json={"nodes": [
        {"id": "s", "intent": "shout", "tool": "shout",
         "args": {}, "success_predicate": "ok"}]})
    assert bad.status_code == 400 and "missing required arg 'text'" in str(bad.json())
    ok = client.post("/runs", json={"nodes": [
        {"id": "s", "intent": "shout", "tool": "shout",
         "args": {"text": "hi"}, "success_predicate": "ok"}]})
    assert ok.status_code == 200, ok.text
    # exec of a dynamic tool routes to the sandbox /run_tool endpoint with the name
    monkeypatch.setattr(core, "_exec_tool", _REAL_EXEC_TOOL)
    core.POLICY["auto_approve"] = ["shout"]
    run_id = ok.json()["run_id"]
    client.post("/claim", json={"agent_id": "t"})
    res = client.post(f"/runs/{run_id}/exec",
                      json={"node_id": "s", "tool": "shout", "args": {"text": "hi"}}).json()
    assert res["status"] == "done"
    assert calls[-1] == (core.CFG["sandbox_url"] + "/run_tool",
                         {"name": "shout", "args": {"text": "hi"}})
    # hot-reload prunes: re-sync with an empty discovery -> dynamic tools gone
    class EmptyClient(FakeClient):
        async def get(self, url):
            calls.append((url, None))
            return R2()

    class R2:
        def raise_for_status(self): pass
        def json(self): return {"tools": {}}

    monkeypatch.setattr(core.httpx, "AsyncClient", EmptyClient)
    out2 = client.post("/tools/sync").json()
    assert "shout" not in out2["tools"] and "shout" not in core.TOOLS


def test_autosize_and_policy_auto_approve(client, monkeypatch):
    class R:
        status_code = 200
    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return R()
    monkeypatch.setattr(core.httpx, "AsyncClient", FakeClient)
    out = client.get("/autosize").json()
    assert out["ollama_ok"] is True
    assert 1 <= out["suggested_max_agents"] <= 8 and out["ping_ms"] is not None

    class DownClient(FakeClient):
        async def get(self, url): raise core.httpx.ConnectError("down")
    monkeypatch.setattr(core.httpx, "AsyncClient", DownClient)
    out = client.get("/autosize").json()
    assert out["ollama_ok"] is False and out["suggested_max_agents"] == 1

    ok = client.post("/policy", json={"auto_approve": ["shell", "write_file"]})
    assert ok.status_code == 200 and ok.json()["auto_approve"] == ["shell", "write_file"]
    core.POLICY["auto_approve"] = []


def test_metrics(client):
    run_id = mk_template_run(client)
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/w/result", json={"state": "FAILED", "hint": "boom"})
    client.post("/claim", json={"agent_id": "t"})   # REPAIRING is re-claimed
    client.post(f"/runs/{run_id}/nodes/w/result", json={"state": "DONE", "result": 1})
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/r/result", json={"state": "DONE", "result": 2})
    m = client.get("/metrics").json()
    per = m["runs"][run_id]
    assert per["all_done"] is True and per["frt_ms"] is not None  # failure -> recovery measured
    assert m["total"]["tsr_pct"] == 100
    assert m["total"]["first_pass_pct"] == 50   # w took 2 attempts, r took 1


def test_decompose_normalizes_null_tool_string(client):
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "big task"}]})
    run_id = r.json()["run_id"]
    ok = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "think further", "tool": "null", "args": {"junk": 1}},
        {"intent": "write", "tool": "write_file", "args": {"path": "f", "content": "c"},
         "success_predicate": "exists"},
    ]})
    assert ok.status_code == 200, ok.text
    node = client.get(f"/runs/{run_id}").json()["nodes"]["root.1"]
    assert node["tool"] is None and node["args"] == {}   # "null" string -> abstract child


def test_decompose_validates_before_parking_approval(client):
    core.POLICY["approve_dag_edits"] = True
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "big task"}]})
    run_id = r.json()["run_id"]
    # bad tool must 400 at submit (agent repair loop), not when the user approves
    bad = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "x", "tool": "bogus_tool", "args": {}, "success_predicate": "p"}]})
    assert bad.status_code == 400 and "unknown tool" in str(bad.json())
    assert client.get("/approvals?status=pending").json() == []   # nothing parked
    # a clean decomposition still parks an approval as before
    ok = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "first"}, {"intent": "second", "deps": [0]}]})
    assert ok.status_code == 200 and ok.json()["status"] == "pending_approval"
    # UI round-trip: approve with the parked children verbatim (resolved deps ids)
    pend = client.get("/approvals?status=pending").json()
    assert len(pend) == 1
    deps = pend[0]["args"]["children"][1]["deps"]
    assert deps == ["root.1"]                      # parked form carries resolved ids
    resp = client.post(f"/approvals/{pend[0]['id']}", json={
        "decision": "approve", "args": {"children": pend[0]["args"]["children"]}})
    assert resp.status_code == 200 and resp.json()["result"]["applied"] is True
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root.2"]["deps"] == ["root.1"]
    core.POLICY["approve_dag_edits"] = False


def test_llm_virtual_tool_registered(client):
    tools = client.get("/tools").json()["tools"]
    assert tools["llm"]["required"] == []
    # decomposition with an llm synthesis child validates and parks cleanly
    core.POLICY["approve_dag_edits"] = False
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "research x"}]})
    run_id = r.json()["run_id"]
    ok = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "gather", "tool": "web_search", "args": {"query": "x"},
         "success_predicate": "hits"},
        {"intent": "write the answer", "tool": "llm", "deps": [0],
         "success_predicate": "answers the parent question"}]})
    assert ok.status_code == 200, ok.text
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root.2"]["tool"] == "llm"
    assert run["nodes"]["root.2"]["deps"] == ["root.1"]


def test_delete_active_run_requires_force_then_succeeds(client):
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "x"}]})
    run_id = r.json()["run_id"]
    assert client.delete(f"/runs/{run_id}").status_code == 409      # active
    assert client.delete(f"/runs/{run_id}?force=1").status_code == 200
    assert client.get(f"/runs/{run_id}").status_code == 404
    # and the agent's stale report afterwards must not resurrect it
    client.post(f"/runs/{run_id}/nodes/root/result",
                json={"state": "DONE", "result": {"x": 1}})
    assert client.get(f"/runs/{run_id}").status_code == 404


def test_checkpoint_resume(tmp_path, client):
    run_id = mk_template_run(client)
    # simulate restart: reload from sqlite only
    core.RUNS.clear()
    core.RUNS.update(core.store.load_runs())
    run = client.get(f"/runs/{run_id}").json()
    assert run["id"] == run_id and set(run["nodes"]) == {"w", "r"}
