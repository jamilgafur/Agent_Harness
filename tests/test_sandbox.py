"""Sandbox web tools: HTML scraping and text extraction are pure functions."""
import json
import sys
from pathlib import Path

import pytest

import harness.sandbox as sb

SEARX_HTML = """
<html><body>
<article class="result result-default category-general">
  <a href="https://a.test/one" class="url_header"><h3>
    <a href="https://a.test/one" rel="noreferrer">First &amp; Result</a></h3></a>
  <p class="content">snippet <b>one</b> here</p>
</article>
<article class="result result-default category-general">
  <h3>
    <a href="https://b.test/two">Second</a></h3>
</article>
</body></html>
"""


def test_parse_searx_html_extracts_title_url_snippet():
    got = sb._parse_searx_html(SEARX_HTML, limit=10)
    assert got[0]["url"] == "https://a.test/one"
    assert got[0]["title"] == "First & Result"
    assert "snippet one here" in got[0]["snippet"]
    assert got[1] == {"url": "https://b.test/two", "title": "Second", "snippet": ""}


def test_parse_searx_html_respects_limit():
    assert len(sb._parse_searx_html(SEARX_HTML, limit=1)) == 1


def test_html_to_text_strips_scripts_and_tags():
    html = "<h1>Title</h1><script>var x=1;</script><p>Hello <b>world</b>&amp; co</p>"
    text = sb._html_to_text(html)
    assert "var x" not in text
    assert "Title" in text and "Hello world& co" in text
    assert "<" not in text


# ---------- MCP stdio client (P2) ----------
EXAMPLE_MCP = Path(sb.__file__).parent.parent / "examples" / "mcp_hello.py"


@pytest.fixture()
def mcp_client(monkeypatch, tmp_path):
    monkeypatch.setattr(sb, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(sb, "MCP_MANIFEST", tmp_path / "mcp.json")
    man = sb._MCPMan()
    monkeypatch.setattr(sb, "MCP", man)
    from fastapi.testclient import TestClient
    c = TestClient(sb.app)
    yield c, man, tmp_path
    man.close_all()


def test_mcp_discover_call_and_hot_reload(mcp_client):
    c, man, tmp = mcp_client
    manifest = {"servers": {"hello": {"command": [sys.executable, str(EXAMPLE_MCP)]}}}
    (tmp / "mcp.json").write_text(json.dumps(manifest))
    import time as _t; _t.sleep(1.05)  # mtime granularity on some filesystems
    tools = c.get("/discover_tools").json()["tools"]
    assert tools["mcp_hello_greet"]["required"] == ["name"]
    ok = c.post("/run_tool", json={"name": "mcp_hello_greet",
                                   "args": {"name": "ada"}}).json()
    assert ok == {"text": "hello ada"}
    # hot-reload: empty manifest -> tools vanish, server processes terminated
    (tmp / "mcp.json").unlink()
    assert c.get("/discover_tools").json()["tools"] == {}
    assert not man.servers


def test_mcp_broken_manifest_keeps_good_servers(mcp_client):
    c, man, tmp = mcp_client
    manifest = {"servers": {"hello": {"command": [sys.executable, str(EXAMPLE_MCP)]}}}
    (tmp / "mcp.json").write_text(json.dumps(manifest))
    import time as _t; _t.sleep(1.05)
    assert "mcp_hello_greet" in c.get("/discover_tools").json()["tools"]
    import time as _t2; _t2.sleep(1.05)
    (tmp / "mcp.json").write_text("{ not json")
    # broken manifest: current tool set keeps working
    ok = c.post("/run_tool", json={"name": "mcp_hello_greet",
                                   "args": {"name": "bob"}}).json()
    assert ok == {"text": "hello bob"}
TOOL_FILE = '''
TOOL = {"name": "shout", "desc": "uppercase a string", "required": ["text"]}

def run(text):
    return {"output": text.upper()}
'''


def test_dynamic_tool_discover_and_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sb, "TOOLS_DIR", tmp_path)
    (tmp_path / "shout.py").write_text(TOOL_FILE)
    (tmp_path / "_ignored.py").write_text("x=1")          # private files skipped
    (tmp_path / "broken.py").write_text("import nope_x")  # broken files skipped
    from fastapi.testclient import TestClient
    c = TestClient(sb.app)
    disc = c.get("/discover_tools").json()["tools"]
    assert disc == {"shout": {"file": "shout.py", "desc": "uppercase a string",
                              "required": ["text"]}}
    ok = c.post("/run_tool", json={"name": "shout", "args": {"text": "hi"}}).json()
    assert ok == {"output": "HI"}
    missing = c.post("/run_tool", json={"name": "shout", "args": {}}).json()
    assert missing["rc"] == -6 and "missing required args" in missing["stderr"]
    r = c.post("/run_tool", json={"name": "nope", "args": {}})
    assert r.status_code == 404
