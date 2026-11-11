"""Sandbox web tools: HTML scraping and text extraction are pure functions."""
import json
import sys
import time
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


# ---------- web_search reads the pages it links ----------
class _Resp:
    def __init__(self, status_code=200, text="", data=None):
        self.status_code, self.text = status_code, text
        self.headers = {"content-type": "text/html"}
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("http error")


def test_web_search_reads_every_hit_and_survives_dead_urls(monkeypatch):
    pages = {"https://a.test/one": "<h1>Alpha</h1><p>full article text here</p>"}

    class FakeClient:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None, headers=None, timeout=None,
                follow_redirects=False):
            if url.endswith("/search"):
                if params and params.get("format") == "json":
                    return _Resp(403)            # force the HTML scrape path
                return _Resp(200, SEARX_HTML)
            if url in pages:
                return _Resp(200, pages[url])
            raise sb.httpx.ConnectError("dead")

    monkeypatch.setattr(sb.httpx, "Client", FakeClient)
    from fastapi.testclient import TestClient
    out = TestClient(sb.app).post(
        "/web_search", json={"query": "x", "max_results": 5}).json()
    hits = {h["url"]: h for h in out["results"]}
    assert "full article text here" in hits["https://a.test/one"]["page_text"]
    assert "page_text" not in hits["https://b.test/two"]   # dead url: hit kept


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
    time.sleep(1.05)  # mtime granularity on some filesystems
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
    time.sleep(1.05)
    assert "mcp_hello_greet" in c.get("/discover_tools").json()["tools"]
    time.sleep(1.05)
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


def test_create_skill_writes_discovers_and_quotes(monkeypatch, tmp_path):
    """A declarative skill spec becomes a real tool module: shell template with
    shlex-quoted args, discovered by the normal sync path."""
    monkeypatch.setattr(sb, "TOOLS_DIR", tmp_path / "tools")
    monkeypatch.setattr(sb, "WORKSPACE", tmp_path)
    from fastapi.testclient import TestClient
    c = TestClient(sb.app)
    r = c.post("/create_skill", json={"name": "count_lines", "desc": "count file lines",
                                      "required": ["path"], "cmd": "wc -l < {path}"})
    assert r.status_code == 200, r.text
    assert r.json()["tool"] == {"file": "count_lines.py", "desc": "count file lines",
                                "required": ["path"]}
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n")
    out = c.post("/run_tool", json={"name": "count_lines",
                                    "args": {"path": "notes.txt with spaces"}}).json()
    assert out["rc"] != 0                    # quoted as ONE arg: no such file -> rc 2…
    out = c.post("/run_tool", json={"name": "count_lines",
                                    "args": {"path": "notes.txt"}}).json()
    assert out["rc"] == 0 and "3" in out["stdout"]
    bad = c.post("/create_skill", json={"name": "Bad Name", "cmd": "echo"})
    assert bad.status_code == 400
    bad2 = c.post("/create_skill", json={"name": "ok_name", "cmd": "ls {noarg}"})
    assert bad2.status_code == 400 and "noarg" in str(bad2.json())


def test_confined_rejects_same_prefix_sibling_escape(tmp_path, monkeypatch):
    """../workspace_evil/x resolved outside WORKSPACE but passed the old
    startswith("/workspace") check (path-prefix bug)."""
    from fastapi import HTTPException
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "ws_evil").mkdir()
    monkeypatch.setattr(sb, "WORKSPACE", ws)
    with pytest.raises(HTTPException):
        sb._confined("../ws_evil/x.txt")
    assert sb._confined("sub/../x.txt") == ws / "x.txt"
