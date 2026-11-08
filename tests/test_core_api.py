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


def test_claim_hands_out_one_node_per_call(client):
    """One node per claim: a claimed-but-unhandled node would sit RUNNING until its
    lease expired. Parallelism is scaled agents (--scale agent=N), not bulk claims."""
    client.post("/runs", json={"nodes": [
        {"id": x, "intent": f"task {x}", "tool": "write_file",
         "args": {"path": x, "content": "1"}, "success_predicate": "done"}
        for x in ("a", "b", "c")]})
    c1 = client.post("/claim", json={"agent_id": "t"}).json()
    assert [n["id"] for n in c1["nodes"]] == [c1["node"]["id"]]   # single, "nodes" mirrors
    c2 = client.post("/claim", json={"agent_id": "u"}).json()
    c3 = client.post("/claim", json={"agent_id": "v"}).json()
    assert len({c1["node"]["id"], c2["node"]["id"], c3["node"]["id"]}) == 3
    assert client.post("/claim", json={"agent_id": "w"}).json()["node"] is None


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
        async def get(self, url):
            calls.append((url, None))
            return R()

        async def post(self, url, json=None):
            calls.append((url, json))
            return R()

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
    # a tool synced after startup is "unvetted": first use parks a card even with
    # auto_approve; approving vets it, later calls fly directly
    res = client.post(f"/runs/{run_id}/exec",
                      json={"node_id": "s", "tool": "shout", "args": {"text": "hi"}}).json()
    assert res["status"] == "pending"
    d = client.post(f"/approvals/{res['exec_id']}", json={"decision": "approve"}).json()
    assert d["status"] == "done"
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
    # plan-checkpoint fields survive the restart too
    pid = client.post("/runs", json={"intent": "research y"}).json()["run_id"]
    client.put(f"/runs/{pid}/constraints", json={"constraints": ["discuss cost"]})
    core.RUNS.clear()
    core.RUNS.update(core.store.load_runs())
    run = client.get(f"/runs/{pid}").json()
    assert run["phase"] == "planning" and run["constraints"] == ["discuss cost"]


# ---------- plan-first checkpoint (intent runs) ----------
def _intent_children():
    return {"children": [
        {"intent": "gather a", "tool": "web_search",
         "args": {"query": "a", "max_results": 3}, "deps": [],
         "success_predicate": "hits"},
        {"intent": "gather b", "tool": "web_search",
         "args": {"query": "b", "max_results": 3}, "deps": [],
         "success_predicate": "hits"},
        {"intent": "synthesize", "tool": "llm", "deps": [0, 1],
         "success_predicate": "answers the question"}]}


def test_intent_runs_plan_first_checkpoint_then_go(client):
    run_id = client.post("/runs", json={"intent": "research something"}).json()["run_id"]
    got = client.get(f"/runs/{run_id}").json()
    assert got["phase"] == "planning" and got["plan_ready"] is False

    # only abstract nodes are claimable during planning — the tree grows fully first
    claim = client.post("/claim", json={"agent_id": "t1"}).json()
    assert claim["node"]["id"] == "root" and not claim["node"]["tool"]
    d = client.post(f"/runs/{run_id}/nodes/root/decompose", json=_intent_children())
    assert d.status_code == 200, d.text
    got = client.get(f"/runs/{run_id}").json()
    assert got["plan_ready"] is True

    # tool work is held at the checkpoint until the user approves the plan
    assert client.post("/claim", json={"agent_id": "t2"}).json() == {"node": None}
    assert client.post(f"/runs/{run_id}/go").json() == {"ok": True}
    got = client.post("/claim", json={"agent_id": "t2"}).json()
    assert got["node"]["tool"] == "web_search"           # bottom-up: leaves first
    assert client.post(f"/runs/{run_id}/go").status_code == 409     # idempotent guard


def test_templates_skip_the_checkpoint_and_head_owns_the_answer(client):
    run_id = client.post("/runs", json={"nodes": [
        {"id": "root", "intent": "head question"},
        {"id": "root.1", "intent": "gather", "tool": "web_search", "depth": 1,
         "args": {"query": "q", "max_results": 3}, "parent": "root",
         "success_predicate": "hits"},
        {"id": "root.2", "intent": "write", "tool": "llm", "depth": 1,
         "deps": ["root.1"], "parent": "root",
         "success_predicate": "answers the question"}]}).json()["run_id"]
    assert client.get(f"/runs/{run_id}").json()["phase"] == "executing"

    assert client.post("/claim", json={"agent_id": "a"}).json()["node"]["id"] == "root.1"
    client.post(f"/runs/{run_id}/nodes/root.1/result", json={
        "state": "DONE", "result": {"results": [{"url": "u", "page_text": "t"}]}})
    assert client.post("/claim", json={"agent_id": "a"}).json()["node"]["id"] == "root.2"
    client.post(f"/runs/{run_id}/nodes/root.2/result",
                json={"state": "DONE", "result": {"text": "THE HEAD ANSWER"}})
    got = client.get(f"/runs/{run_id}").json()["nodes"]
    # the writer's text climbed to the head; the raw search dump stays on its leaf
    assert got["root"]["result"] == {"text": "THE HEAD ANSWER", "from": "root.2"}
    assert got["root"]["state"] == "DONE"

    # no writer child: parents keep the classic child-id aggregate
    run2 = client.post("/runs", json={"nodes": [
        {"id": "root", "intent": "agg"},
        {"id": "root.1", "intent": "t1", "tool": "shell", "depth": 1,
         "args": {"cmd": "true"}, "parent": "root", "success_predicate": "ok"},
        {"id": "root.2", "intent": "t2", "tool": "shell", "depth": 1,
         "args": {"cmd": "true"}, "parent": "root", "success_predicate": "ok"}]}).json()["run_id"]
    for nid in ("root.1", "root.2"):
        c = client.post("/claim", json={"agent_id": "b"}).json()
        assert c["node"]["id"] == nid
        assert client.post(f"/runs/{run2}/nodes/{nid}/result",
                           json={"state": "DONE", "result": {"rc": 0, "stdout": nid}}
                           ).status_code == 200
    root = client.get(f"/runs/{run2}").json()["nodes"]["root"]
    assert root["state"] == "DONE"
    assert root["result"] == {"root.1": {"rc": 0, "stdout": "root.1"},
                              "root.2": {"rc": 0, "stdout": "root.2"}}


def test_constraints_roundtrip_and_ride_the_claim_payload(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    r = client.put(f"/runs/{run_id}/constraints",
                   json={"constraints": ["ignore methodology", "discuss cost"]})
    assert r.status_code == 200
    assert r.json()["constraints"] == ["ignore methodology", "discuss cost"]
    claim = client.post("/claim", json={"agent_id": "t"}).json()
    assert claim["constraints"] == ["ignore methodology", "discuss cost"]
    assert client.get(f"/runs/{run_id}").json()["constraints"] == ["ignore methodology",
                                                                    "discuss cost"]
    assert client.put(f"/runs/{run_id}/constraints",
                      json={"constraints": [f"c{i}" for i in range(13)]}).status_code == 400
    assert client.put(f"/runs/{run_id}/constraints",
                      json={"constraints": []}).json()["constraints"] == []


def test_replan_breaks_down_further_during_planning(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    client.post("/claim", json={"agent_id": "t"})
    assert client.post(f"/runs/{run_id}/nodes/root/decompose",
                       json=_intent_children()).status_code == 200
    # break one leaf down further: back to abstract PENDING, re-claimable by a planner
    assert client.post(f"/runs/{run_id}/nodes/root.1/replan").status_code == 200
    n = client.get(f"/runs/{run_id}").json()["nodes"]["root.1"]
    assert n["tool"] is None and n["state"] == "PENDING"
    c = client.post("/claim", json={"agent_id": "t2"}).json()
    assert c["node"]["id"] == "root.1" and not c["node"]["tool"]
    # replanning a parent removes the whole subtree (release the planner first)
    client.post(f"/runs/{run_id}/nodes/root.1/redirect",
                json={"agent_id": "t2", "reason": "finishing the test"})
    assert client.post(f"/runs/{run_id}/nodes/root/replan").status_code == 200
    assert set(client.get(f"/runs/{run_id}").json()["nodes"]) == {"root"}
    # after /go the op is gone — fix & retry takes over
    assert client.post(f"/runs/{run_id}/go").status_code == 200
    assert client.post(f"/runs/{run_id}/nodes/root/replan").status_code == 409


def test_user_adds_and_edits_branches_during_planning(client):
    run_id = client.post("/runs", json={"intent": "research x"}).json()["run_id"]
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/root/decompose", json=_intent_children())

    # add a branch: enters abstract (the LLM planner owns it), claimable at once
    r = client.post(f"/runs/{run_id}/nodes",
                    json={"intent": "compare TCO of vLLM vs TGI", "parent": "root"})
    assert r.status_code == 200 and r.json()["id"] == "root.4"
    # the convergence invariant was re-pinned: the sink now consumes the new branch
    sink = client.get(f"/runs/{run_id}").json()["nodes"]["root.3"]
    assert sink["tool"] == "llm" and set(sink["deps"]) == {"root.1", "root.2", "root.4"}
    assert client.get(f"/runs/{run_id}").json()["plan_ready"] is False
    # edit its instruction before it gets claimed
    assert client.patch(f"/runs/{run_id}/nodes/root.4",
                        json={"intent": "compare production TCO of vLLM vs TGI"}) \
        .status_code == 200
    c = client.post("/claim", json={"agent_id": "t2"}).json()
    assert c["node"]["id"] == "root.4" and not c["node"]["tool"]
    assert c["node"]["intent"].startswith("compare production")
    # the planner sees its neighbors: parent intent + sibling branches
    assert c["context"]["parent_intent"] == "research x"
    assert {s["id"] for s in c["context"]["siblings"]} == {"root.1", "root.2", "root.3"}
    assert all("intent" in s for s in c["context"]["siblings"])
    # edge cases: missing parent, empty intent, and everything freezes after /go
    assert client.post(f"/runs/{run_id}/nodes",
                       json={"intent": "dangling", "parent": "nope"}).status_code == 404
    assert client.post(f"/runs/{run_id}/nodes",
                       json={"intent": "  ", "parent": "root"}).status_code == 400
    client.post(f"/runs/{run_id}/nodes/root.4/redirect",
                json={"agent_id": "t2", "reason": "back to the planner pool"})
    assert client.get(f"/runs/{run_id}").json()["plan_ready"] is False  # still abstract pending
    client.post(f"/runs/{run_id}/go")
    assert client.post(f"/runs/{run_id}/nodes",
                       json={"intent": "late", "parent": "root"}).status_code == 409
    assert client.patch(f"/runs/{run_id}/nodes/root.4",
                        json={"intent": "late"}).status_code == 409


def test_result_gated_planner_nodes_keep_deps_and_refine_after_go(client):
    """The model asks for a result-gated re-plan (abstract child deps'd on a search).
    The dep edge is PRESERVED — during planning the claim gate holds the node, the
    checkpoint exempts it, and after /go it refines with the real result in hand."""
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    client.post("/claim", json={"agent_id": "t"})
    d = client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "read the source", "tool": "web_search",
         "args": {"query": "q", "max_results": 3}, "deps": [], "success_predicate": "hits"},
        {"intent": "break the findings down", "deps": [0], "success_predicate": ""}]})
    assert d.status_code == 200, d.text
    nodes = client.get(f"/runs/{run_id}").json()["nodes"]
    # the information edge survives into the stored graph (the audit trail)
    assert nodes["root.2"]["deps"] == ["root.1"] and nodes["root.2"]["tool"] is None
    # gated abstract nodes are exempt from the checkpoint…
    assert client.get(f"/runs/{run_id}").json()["plan_ready"] is True
    # …and are NOT claimable during planning (their context arrives with the results)
    assert client.post("/claim", json={"agent_id": "t2"}).json()["node"] is None
    client.post(f"/runs/{run_id}/go")
    # still gated after /go until the search actually returns
    assert client.post("/claim", json={"agent_id": "t2"}).json()["node"]["id"] == "root.1"
    client.post(f"/runs/{run_id}/nodes/root.1/result",
                json={"state": "DONE", "result": {"results": [{"url": "u"}]}})
    c = client.post("/claim", json={"agent_id": "t2"}).json()
    assert c["node"]["id"] == "root.2"
    assert c["inputs"] == {"root.1": {"results": [{"url": "u"}]}}   # refined with real data


# ---------- per-run prompts/params overrides (PUT /config) ----------
def test_config_roundtrip_clamps_and_persists(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    r = client.put(f"/runs/{run_id}/config",
                   json={"prompts": {"synthesize": "BRIEF ONLY", "bogus": "dropped"},
                         "params": {"child_cap": 9, "temperature": 0.9}})
    assert r.status_code == 200, r.text
    assert r.json()["prompts"] == {"synthesize": "BRIEF ONLY"}   # unknown prompt key dropped
    assert r.json()["params"] == {"child_cap": 6, "temperature": 0.9}   # clamped like agents do
    got = client.get(f"/runs/{run_id}").json()
    assert got["prompts"] == {"synthesize": "BRIEF ONLY"}
    assert got["params"]["child_cap"] == 6
    core.RUNS.clear()                                            # persisted, not process-memory
    core.RUNS.update(core.store.load_runs())
    assert client.get(f"/runs/{run_id}").json()["params"]["child_cap"] == 6


def test_config_rejects_unknown_params_and_executing_runs(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    assert client.put(f"/runs/{run_id}/config",
                      json={"params": {"nonsense": 1}}).status_code == 400
    run_id2 = mk_template_run(client)                            # executing, not planning
    assert client.put(f"/runs/{run_id2}/config",
                      json={"params": {"child_cap": 2}}).status_code == 409


def test_claim_carries_config_overrides(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    client.put(f"/runs/{run_id}/config",
               json={"prompts": {"judge": "BE STRICT"}, "params": {"child_cap": 2}})
    client.post(f"/runs/{run_id}/nodes/root/decompose", json=_intent_children())
    client.post(f"/runs/{run_id}/go")
    claim = client.post("/claim", json={"agent_id": "t"}).json()
    assert claim["prompts"] == {"judge": "BE STRICT"}
    assert claim["params"].get("child_cap") == 2


def test_patch_node_edits_args_and_validates_required_keys(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json=_intent_children())
    r = client.patch(f"/runs/{run_id}/nodes/root.1", json={"args": {"max_results": 3}})
    assert r.status_code == 400                                  # web_search still needs query
    r = client.patch(f"/runs/{run_id}/nodes/root.1",
                     json={"args": {"query": "new q", "max_results": 9}})
    assert r.status_code == 200, r.text
    node = client.get(f"/runs/{run_id}").json()["nodes"]["root.1"]
    assert node["args"]["query"] == "new q" and node["args"]["max_results"] == 9


# ---------- document-shaped runs: section wiring + knowledge tree ----------
def test_named_parts_wire_section_writers_to_earlier_sections(client):
    run_id = client.post("/runs", json={
        "intent": "white paper on dark matter with an abstract, background, "
                  "and conclusion"}).json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "abstract section", "tool": None, "args": {}, "deps": []},
        {"intent": "background section", "tool": None, "args": {}, "deps": []},
        {"intent": "conclusion section", "tool": None, "args": {}, "deps": []},
        {"intent": "assemble", "tool": "llm", "args": {}, "deps": [0, 1, 2],
         "success_predicate": "assembled"}]})
    r = client.post(f"/runs/{run_id}/nodes/root.2/decompose", json={"children": [
        {"intent": "gather", "tool": "web_search",
         "args": {"query": "q", "max_results": 3}, "deps": [], "success_predicate": "hits"},
        {"intent": "write the section", "tool": "llm", "deps": [0],
         "success_predicate": "written"}]})
    assert r.status_code == 200, r.text
    nodes = client.get(f"/runs/{run_id}").json()["nodes"]
    assert "root.1" in nodes["root.2.2"]["deps"]      # writer consumes the earlier section

    r2 = client.post(f"/runs/{run_id}/nodes/root.3/decompose", json={"children": [
        {"intent": "write the conclusion", "tool": "llm", "deps": [],
         "success_predicate": "written"}]})
    assert r2.status_code == 200, r2.text
    nodes = client.get(f"/runs/{run_id}").json()["nodes"]
    assert nodes["root.3.1"]["deps"] == ["root.1", "root.2"]

    r3 = client.post(f"/runs/{run_id}/nodes/root.1/decompose", json={"children": [
        {"intent": "write the abstract", "tool": "llm", "deps": [],
         "success_predicate": "written"}]})
    assert r3.status_code == 200, r3.text
    assert client.get(f"/runs/{run_id}").json()["nodes"]["root.1.1"]["deps"] == []


def test_unnamed_runs_get_no_section_wiring(client):
    run_id = client.post("/runs", json={"intent": "compare vLLM and TGI"}).json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "branch a", "tool": None, "args": {}, "deps": []},
        {"intent": "branch b", "tool": None, "args": {}, "deps": []},
        {"intent": "synthesize", "tool": "llm", "args": {}, "deps": [0, 1],
         "success_predicate": "done"}]})
    client.post(f"/runs/{run_id}/nodes/root.2/decompose", json={"children": [
        {"intent": "write b", "tool": "llm", "deps": [], "success_predicate": "w"}]})
    nodes = client.get(f"/runs/{run_id}").json()["nodes"]
    assert nodes["root.2.1"]["deps"] == []            # parallel branches stay independent


def test_knowledge_endpoint_nests_text_holders(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json=_intent_children())
    nodes = core.RUNS[run_id]["nodes"]
    nodes["root.1"].state = core.NodeState.DONE
    nodes["root.1"].result = {"text": "branch one summary " * 40}
    nodes["root.2"].state = core.NodeState.DONE       # done but no text -> pruned
    k = client.get(f"/runs/{run_id}/knowledge").json()["knowledge"]
    assert len(k) == 1 and k[0]["id"] == "root" and k[0]["text"] == ""
    kids = {c["id"]: c for c in k[0]["children"]}
    assert set(kids) == {"root.1"} and kids["root.1"]["chars"] == len("branch one summary " * 40)


def test_claim_context_carries_known_material(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json=_intent_children())
    client.post(f"/runs/{run_id}/go")
    nodes = core.RUNS[run_id]["nodes"]
    nodes["root.1"].state = core.NodeState.DONE
    nodes["root.1"].result = {"text": "summary of branch one " * 30}
    claim = client.post("/claim", json={"agent_id": "t"}).json()
    assert claim["node"]["id"] == "root.2"
    known = {x["id"] for x in claim["context"]["known"]}
    assert known == {"root.1"}


# ---------- agent-generated task guide ----------
def test_guide_endpoint_is_idempotent_and_visible(client):
    run_id = client.post("/runs", json={"intent": "x"}).json()["run_id"]
    r1 = client.post(f"/runs/{run_id}/guide", json={"text": "  sections and evidence  "})
    assert r1.status_code == 200
    assert r1.json()["constraints"] == ["guide: sections and evidence"]
    r2 = client.post(f"/runs/{run_id}/guide", json={"text": "something else"})
    assert r2.json()["constraints"] == ["guide: sections and evidence"]   # first wins
    assert client.get(f"/runs/{run_id}").json()["constraints"] == ["guide: sections and evidence"]
    assert client.post(f"/runs/{run_id}/guide", json={"text": " "}).status_code == 400


def test_planning_claims_abstract_node_behind_abstract_deps(client):
    """9ea37d2e corner: the writer-slot node the model made tool:null is abstract
    with abstract deps — it must still be claimable in planning."""
    run_id = client.post("/runs", json={"intent": "learn about x"}).json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "branch a", "tool": None, "args": {}, "deps": []},
        {"intent": "branch b", "tool": None, "args": {}, "deps": []},
        {"intent": "branch c", "tool": None, "args": {}, "deps": []},
        {"intent": "the answer", "tool": None, "args": {}, "deps": [0, 1, 2],
         "success_predicate": ""}]})
    claim = client.post("/claim", json={"agent_id": "t"}).json()
    assert claim["node"]["id"] in {"root.1", "root.2", "root.3", "root.4"}
    got = client.post("/claim", json={"agent_id": "t"}).json()
    ids = {got["node"]["id"]} if got["node"] else set()
    assert ids  # keeps handing out abstracts; root.4 is not gated on its abstract deps
    for nid in ("root.1", "root.2", "root.3"):
        core.RUNS[run_id]["nodes"][nid].state = core.NodeState.WAITING
    claim = client.post("/claim", json={"agent_id": "t2"}).json()
    assert claim["node"]["id"] == "root.4"


def test_shipped_policy_matches_code_defaults():
    """models.DEFAULT_POLICY is the no-toml fallback; any drift from the shipped
    harness.toml must fail here (one source of truth, enforced)."""
    import tomllib
    from pathlib import Path

    from harness.models import DEFAULT_POLICY
    shipped = tomllib.loads(Path("harness.toml").read_text())["policy"]
    for k, v in shipped.items():
        assert DEFAULT_POLICY.get(k) == v, \
            f"policy drift on {k}: code={DEFAULT_POLICY.get(k)!r} toml={v!r}"


def test_escalation_scopes_to_the_branch_and_recovery_rejoins(client):
    """A leaf failing terminally escalates only ITSELF (plus BLOCKING its dep
    consumers); the parent keeps working other children with a pointer hint.
    fix & retry on the leaf reopens the branch and the run completes."""
    r = client.post("/runs", json={"nodes": [
        {"id": "root", "intent": "head"},
        {"id": "root.1", "intent": "gather", "tool": "write_file", "depth": 1,
         "args": {"path": "f", "content": "c"}, "parent": "root", "success_predicate": "ok"},
        {"id": "root.2", "intent": "write from f", "tool": "shell", "depth": 1,
         "args": {"cmd": "cat f"}, "deps": ["root.1"], "parent": "root",
         "success_predicate": "ok"}]})
    run_id = r.json()["run_id"]
    assert r.status_code == 200, r.text
    for i in range(2):   # max_repairs=2 -> terminal
        assert client.post("/claim", json={"agent_id": "t"}).json()["node"]["id"] == "root.1"
        client.post(f"/runs/{run_id}/nodes/root.1/result",
                    json={"state": "FAILED", "hint": f"boom {i}"})
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root.1"]["state"] == "ESCALATED"
    assert run["nodes"]["root.2"]["state"] == "BLOCKED"
    assert run["nodes"]["root"]["state"] == "PENDING"      # NOT escalated
    assert "root.1" in run["nodes"]["root"]["hint"]
    # fix & retry: leaf reopens, the BLOCKED consumer rejoins, run completes
    client.post(f"/runs/{run_id}/patch",
                json={"ops": [{"op": "retry_node", "node_id": "root.1"}]})
    run = client.get(f"/runs/{run_id}").json()
    assert run["nodes"]["root.1"]["state"] == "PENDING"
    assert run["nodes"]["root.2"]["state"] == "PENDING"
    for nid in ("root.1", "root.2"):
        client.post("/claim", json={"agent_id": "t"})
        client.post(f"/runs/{run_id}/nodes/{nid}/result", json={"state": "DONE", "result": 1})
    assert client.get(f"/runs/{run_id}").json()["nodes"]["root"]["state"] == "DONE"


def test_stale_report_does_not_spend_llm_budget(client):
    run_id = mk_template_run(client)
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/w/result",
                json={"state": "DONE", "result": 1, "llm_calls": 5})
    dupe = client.post(f"/runs/{run_id}/nodes/w/result",
                       json={"state": "DONE", "result": 1, "llm_calls": 5})
    assert dupe.status_code == 409
    assert client.get(f"/runs/{run_id}").json()["stats"]["llm_calls"] == 5
    assert client.get(f"/runs/{run_id}").json()["nodes"]["w"]["llm_calls"] == 5


def test_user_patch_validates_required_tool_args(client):
    r = client.post("/runs", json={"nodes": [{"id": "root", "intent": "x"}]})
    run_id = r.json()["run_id"]
    bad = client.post(f"/runs/{run_id}/patch", json={"ops": [
        {"op": "add_node", "node": {"id": "root.1", "intent": "run it", "tool": "shell",
                                    "args": {}, "parent": "root", "depth": 1,
                                    "success_predicate": "p"}}]})
    assert bad.status_code == 400 and "missing required arg 'cmd'" in str(bad.json())


def test_stale_exec_card_does_not_replay(client, monkeypatch):
    """The agent times out on an unanswered card and the node moves on; approving
    the late card must not execute the tool again (mutation replay)."""
    calls = []

    async def fake_exec(tool, args):
        calls.append(tool)
        return {"rc": 0}

    monkeypatch.setattr(core, "_exec_tool", fake_exec)
    run_id = mk_template_run(client)
    client.post("/claim", json={"agent_id": "t"})
    ex = client.post(f"/runs/{run_id}/exec", json={
        "node_id": "w", "tool": "write_file",
        "args": {"path": "a.txt", "content": "hi"}}).json()
    assert ex["status"] == "pending"
    # agent gives up -> FAILED -> REPAIRING; re-claim bumps the attempts epoch
    client.post(f"/runs/{run_id}/nodes/w/result",
                json={"state": "FAILED", "hint": "approval unanswered"})
    client.post("/claim", json={"agent_id": "t"})
    d = client.post(f"/approvals/{ex['exec_id']}", json={"decision": "approve"}).json()
    assert d["status"] == "stale" and calls == []
    # a subsequent card for the CURRENT claim still executes fine
    ex2 = client.post(f"/runs/{run_id}/exec", json={
        "node_id": "w", "tool": "write_file",
        "args": {"path": "a.txt", "content": "hi"}}).json()
    d2 = client.post(f"/approvals/{ex2['exec_id']}", json={"decision": "approve"}).json()
    assert d2["status"] == "done" and calls == ["write_file"]


def test_exec_resolves_from_bindings(client, monkeypatch):
    """$from args resolve against dep results server-side before execution."""
    core.POLICY["auto_approve"] = ["shell"]
    seen = []

    async def fake_exec(tool, args):
        seen.append(args)
        return {"rc": 0}

    monkeypatch.setattr(core, "_exec_tool", fake_exec)
    r = client.post("/runs", json={"nodes": [
        {"id": "a", "intent": "create", "tool": "shell",
         "args": {"cmd": "mk"}, "success_predicate": "p"},
        {"id": "b", "intent": "assign", "tool": "shell",
         "args": {"cmd": "assign", "who": {"$from": "a", "path": "employee.id"}},
         "deps": ["a"], "success_predicate": "p"}]})
    run_id = r.json()["run_id"]
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/a/result",
                json={"state": "DONE", "result": {"employee": {"id": 42}}})
    client.post("/claim", json={"agent_id": "t"})
    ex = client.post(f"/runs/{run_id}/exec", json={
        "node_id": "b", "tool": "shell",
        "args": {"cmd": "assign", "who": {"$from": "a", "path": "employee.id"}}}).json()
    assert ex["status"] == "done" and seen[-1]["who"] == 42
    # unresolvable binding is an error, not a silent null
    bad = client.post(f"/runs/{run_id}/exec", json={
        "node_id": "b", "tool": "shell",
        "args": {"cmd": "assign", "who": {"$from": "a", "path": "employee.nope"}}})
    assert bad.status_code == 400
    core.POLICY["auto_approve"] = []


def test_snapshot_resets_graph_for_template_reuse(client):
    run_id = mk_template_run(client)
    client.post("/claim", json={"agent_id": "t"})
    client.post(f"/runs/{run_id}/nodes/w/result",
                json={"state": "DONE", "result": {"bytes": 2}, "llm_calls": 3})
    snap = client.get(f"/runs/{run_id}/snapshot").json()
    client.post("/runs", json={"nodes": snap["nodes"]})   # re-importable as a template
    states = {n["id"]: n["state"] for n in snap["nodes"]}
    assert set(states.values()) == {"PENDING"}
    assert all(n["result"] is None and n["attempts"] == 0 and n["llm_calls"] == 0
               for n in snap["nodes"])


def test_run_warnings_flag_restatement(client):
    run_id = client.post("/runs", json={"intent": "research vLLM tuning in depth"}) \
        .json()["run_id"]
    client.post(f"/runs/{run_id}/nodes/root/decompose", json={"children": [
        {"intent": "gather tuning levers", "tool": "web_search",
         "args": {"query": "q", "max_results": 3}, "deps": [], "success_predicate": "h"},
        {"intent": "research vLLM tuning in depth", "tool": None, "deps": []}]})
    got = client.get(f"/runs/{run_id}").json()
    assert any("restates" in w for w in got["warnings"])   # lint is surfaced, not silent


def test_dynamic_tool_synced_after_startup_shows_card_once(client, monkeypatch):
    """Audit fix: a dropped+synced tool is "unvetted" — auto_approve cannot fly
    it silently; the first exec parks a card, approving it vets the tool."""
    core.POLICY["auto_approve"] = ["planted"]
    core.TOOLS["planted"] = {"desc": "", "required": [], "dynamic": True, "unvetted": True}
    run_id = client.post("/runs", json={"intent": "t"}).json()["run_id"]
    try:
        ex = client.post(f"/runs/{run_id}/exec",
                         json={"node_id": "root", "tool": "planted", "args": {}}).json()
        assert ex["status"] == "pending"          # card despite auto_approve
        async def fake_exec(tool, args):
            return {"rc": 0, "text": "ok"}
        monkeypatch.setattr(core, "_exec_tool", fake_exec)
        d = client.post(f"/approvals/{ex['exec_id']}",
                        json={"decision": "approve"}).json()
        assert d["status"] == "done"
        assert not core.TOOLS["planted"].get("unvetted")      # vetted by the human card
        ex2 = client.post(f"/runs/{run_id}/exec",
                          json={"node_id": "root", "tool": "planted", "args": {}}).json()
        assert ex2["status"] == "done"                        # auto_approve now applies
    finally:
        core.POLICY["auto_approve"] = []
