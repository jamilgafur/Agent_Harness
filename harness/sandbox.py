"""Sandbox: the ONLY place code runs. Shell + file ops confined to /workspace,
plus outbound web access (search/fetch) so runs can gather references,
plus dynamic tools: drop a .py in /workspace/tools and it becomes a tool (P2).

Convenience boundary, not a security one (see plan.txt §7).
"""
import importlib.util
import json
import os
import re
import select
import subprocess
import time
from html import unescape
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
SEARX_URL = os.environ.get("SEARX_URL", "http://host.docker.internal:8095")
TOOLS_DIR = WORKSPACE / "tools"
MAX_OUT = 20_000
app = FastAPI(title="harness-sandbox")


class ShellReq(BaseModel):
    cmd: str
    timeout: int = 30


class WriteReq(BaseModel):
    path: str
    content: str


class PathReq(BaseModel):
    path: str = "."


class SearchReq(BaseModel):
    query: str
    max_results: int = 6


class FetchReq(BaseModel):
    url: str


def _confined(p: str) -> Path:
    full = (WORKSPACE / p).resolve()
    if not str(full).startswith(str(WORKSPACE)):
        raise HTTPException(400, f"path escapes workspace: {p!r}")
    return full


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + f"\n... <truncated {len(s) - MAX_OUT} chars>"


@app.post("/shell")
def shell(req: ShellReq) -> dict:
    try:
        r = subprocess.run(req.cmd, shell=True, cwd=WORKSPACE, capture_output=True,
                           text=True, timeout=min(req.timeout, 60))
        return {"rc": r.returncode, "stdout": _clip(r.stdout), "stderr": _clip(r.stderr)}
    except subprocess.TimeoutExpired:
        return {"rc": -1, "stdout": "", "stderr": f"timeout after {req.timeout}s"}


@app.post("/read_file")
def read_file(req: PathReq) -> dict:
    p = _confined(req.path)
    if not p.is_file():
        raise HTTPException(404, f"not a file: {req.path!r}")
    return {"content": _clip(p.read_text(errors="replace"))}


@app.post("/write_file")
def write_file(req: WriteReq) -> dict:
    p = _confined(req.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(req.content)
    return {"bytes": len(req.content)}


@app.post("/list_dir")
def list_dir(req: PathReq) -> dict:
    p = _confined(req.path)
    if not p.is_dir():
        raise HTTPException(404, f"not a dir: {req.path!r}")
    return {"entries": sorted(f"{e.name}/" if e.is_dir() else e.name for e in p.iterdir())[:500]}


# ---------- web tools ----------
def _strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"(?s)<[^>]+>", "", html))).strip()


def _html_to_text(html: str) -> str:
    h = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    h = re.sub(r"(?is)<br\s*/?>|</(p|div|li|tr|h[1-6]|section|article)>", "\n", h)
    h = re.sub(r"[ \t]{2,}", " ", _strip_tags(h))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", h).strip()


def _parse_searx_html(html: str, limit: int) -> list:
    """Scrape SearXNG's HTML results (its JSON format is off by default)."""
    out = []
    for art in html.split('<article class="result ')[1:]:
        m = re.search(r'<h3>\s*<a href="([^"]+)"[^>]*>(.*?)</a>', art, re.S)
        if not m:
            continue
        snip = re.search(r'<p class="content">(.*?)</p>', art, re.S)
        out.append({"title": _strip_tags(m.group(2)), "url": m.group(1),
                    "snippet": _strip_tags(snip.group(1)) if snip else ""})
        if len(out) >= limit:
            break
    return out


def _page_text(c: httpx.Client, url: str, cap: int = 1500) -> str:
    """Fetch one page and extract readable text (the 'actually read it' step)."""
    try:
        r = c.get(url, headers={"User-Agent": "harness-research/0.1"},
                  timeout=12, follow_redirects=True)
        text = r.text
        if "html" in r.headers.get("content-type", ""):
            text = _html_to_text(text)
        return text[:cap]
    except Exception:
        return ""


@app.post("/web_search")
def web_search(req: SearchReq) -> dict:
    limit = min(max(req.max_results, 1), 10)
    try:
        with httpx.Client(timeout=15) as c:
            r = c.get(f"{SEARX_URL}/search", params={"q": req.query, "format": "json"})
            if r.status_code == 200:  # instance has JSON enabled
                results = [{"title": x.get("title", ""), "url": x.get("url", ""),
                            "snippet": x.get("content", "")}
                           for x in r.json().get("results", [])[:limit]]
            else:
                r = c.get(f"{SEARX_URL}/search", params={"q": req.query})
                r.raise_for_status()
                results = _parse_searx_html(r.text, limit)
            # read the top pages: snippets alone aren't research
            for hit in results[:3]:
                body = _page_text(c, hit["url"])
                if body:
                    hit["page_text"] = body
            return {"results": results}
    except httpx.HTTPStatusError as e:
        return {"rc": -5, "stderr": f"search backend {e.response.status_code}"}
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return {"rc": -5, "stderr": f"search backend unreachable: {type(e).__name__}"}


@app.post("/http_fetch")
def http_fetch(req: FetchReq) -> dict:
    if not req.url.startswith(("http://", "https://")):
        raise HTTPException(400, "url must start with http:// or https://")
    try:
        with httpx.Client(timeout=25, follow_redirects=True) as c:
            r = c.get(req.url, headers={"User-Agent": "harness-research/0.1"})
            text = r.text
            if "html" in r.headers.get("content-type", ""):
                text = _html_to_text(text)
            return {"status": r.status_code, "url": str(r.url), "content": _clip(text)}
    except httpx.HTTPStatusError as e:
        return {"rc": -5, "stderr": f"fetch {e.response.status_code}"}
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return {"rc": -5, "stderr": f"fetch failed: {type(e).__name__}"}


# ---------- MCP stdio client (P2): manifest-driven, hot-reloaded ----------
MCP_MANIFEST = TOOLS_DIR / "mcp.json"   # {"servers": {"name": {"command": [...]}}}


class _MCP:
    """One MCP server subprocess speaking newline-delimited JSON-RPC over stdio."""

    def __init__(self, command: list) -> None:
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self._next_id = 0
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "harness-sandbox", "version": "0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg: dict) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict, timeout: float = 15) -> dict:
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": self._next_id,
                    "method": method, "params": params})
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not select.select([self.proc.stdout], [], [], deadline - time.time())[0]:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue                    # tolerate server banner noise
            if msg.get("id") != self._next_id:
                continue                    # skip notifications/other ids
            if "error" in msg:
                raise RuntimeError(str(msg["error"])[:200])
            return msg.get("result", {})
        raise TimeoutError(f"{method} timed out")

    def tools(self) -> list:
        return self._rpc("tools/list", {}).get("tools", [])

    def call(self, name: str, args: dict) -> dict:
        return self._rpc("tools/call", {"name": name, "arguments": args}, timeout=30)

    def close(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass


class _MCPMan:
    """Keeps MCP server processes in sync with the manifest (mtime = hot-reload)."""

    def __init__(self) -> None:
        self.manifest_mtime: float = 0.0
        self.servers: dict = {}   # name -> {"proc": _MCP, "sig": str}
        self.tools: dict = {}     # mcp_<srv>_<tool> -> {server, name, desc, required}

    def sync(self) -> None:
        try:
            raw = MCP_MANIFEST.read_text()
            mtime = MCP_MANIFEST.stat().st_mtime
        except OSError:
            if self.servers:
                for s in self.servers.values():
                    s["proc"].close()
                self.servers, self.tools, self.manifest_mtime = {}, {}, 0.0
            return
        if mtime == self.manifest_mtime:
            return
        try:
            wanted = {k: v for k, v in json.loads(raw).get("servers", {}).items()
                      if v.get("command")}
        except json.JSONDecodeError:
            return                          # broken manifest: keep current set
        for name in set(self.servers) - set(wanted) | {
                k for k in set(self.servers) & set(wanted)
                if self.servers[k]["sig"] != json.dumps(wanted[k], sort_keys=True)}:
            self.servers[name]["proc"].close()
            del self.servers[name]
        self.tools = {}
        for name, cfg in wanted.items():
            if name not in self.servers:
                try:
                    self.servers[name] = {"proc": _MCP(cfg["command"]),
                                          "sig": json.dumps(cfg, sort_keys=True)}
                except Exception:
                    continue               # server won't start: skip it, keep others
            try:
                for t in self.servers[name]["proc"].tools():
                    args = t.get("inputSchema", {})
                    self.tools[f"mcp_{name}_{t['name']}"] = {
                        "server": name, "name": t["name"],
                        "desc": t.get("description", "") or f"MCP {name}/{t['name']}",
                        "required": list(args.get("required", []))}
            except Exception:
                pass                        # list failed: keep server for retry next sync
        self.manifest_mtime = mtime

    def close_all(self) -> None:
        for s in self.servers.values():
            s["proc"].close()


MCP = _MCPMan()


# ---------- dynamic tools (P2): /workspace/tools/*.py → registry entries ----------
class RunToolReq(BaseModel):
    name: str
    args: dict = {}


def _tool_modules() -> dict:
    """Dynamic tools = /workspace/tools/*.py (TOOL dict + run()) + MCP tools
    from the manifest (prefixed mcp_<server>_<tool>)."""
    out = {}
    if TOOLS_DIR.is_dir():
        for f in sorted(TOOLS_DIR.glob("*.py")):
            if f.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"dyn_{f.stem}", f)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
            except Exception:      # a broken tool file must not break discovery
                continue
            meta = getattr(m, "TOOL", None)
            if (isinstance(meta, dict) and "name" in meta
                    and callable(getattr(m, "run", None))):
                out[meta["name"]] = {"file": f.name, "desc": meta.get("desc", ""),
                                     "required": list(meta.get("required", []))}
    MCP.sync()                   # hot-reload: manifest mtime drives respawn
    for name, t in MCP.tools.items():
        out[name] = {"file": f"mcp:{t['server']}", "desc": t["desc"],
                     "required": t["required"]}
    return out


@app.get("/discover_tools")
def discover_tools() -> dict:
    return {"tools": _tool_modules()}


@app.post("/run_tool")
def run_tool(req: RunToolReq) -> dict:
    MCP.sync()
    if req.name in MCP.tools:                    # MCP stdio tool (P2 hot-reload)
        t = MCP.tools[req.name]
        missing = [k for k in t["required"] if k not in req.args]
        if missing:
            return {"rc": -6, "stderr": f"missing required args: {missing}"}
        try:
            res = MCP.servers[t["server"]]["proc"].call(t["name"], req.args)
            text = "\n".join(c.get("text", "") for c in res.get("content", [])
                             if isinstance(c, dict) and c.get("type") == "text")
            text = text or json.dumps(res)[:MAX_OUT]
            if res.get("isError"):
                return {"rc": -6, "stderr": text[:300]}
            return {"text": text}
        except Exception as e:
            return {"rc": -6, "stderr": f"mcp {t['server']}: {type(e).__name__}: {e}"[:300]}
    mods = {k: v for k, v in _tool_modules().items() if not v["file"].startswith("mcp:")}
    if req.name not in mods:
        raise HTTPException(404, f"no dynamic tool {req.name!r} in {TOOLS_DIR}")
    f = TOOLS_DIR / mods[req.name]["file"]
    spec = importlib.util.spec_from_file_location(f"dyn_{f.stem}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    missing = [k for k in mods[req.name]["required"] if k not in req.args]
    if missing:
        return {"rc": -6, "stderr": f"missing required args: {missing}"}
    try:
        res = m.run(**req.args)
        return res if isinstance(res, dict) else {"output": res}
    except Exception as e:
        return {"rc": -6, "stderr": f"{type(e).__name__}: {e}"[:300]}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "workspace": str(WORKSPACE)}
