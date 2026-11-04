"""Sandbox: the ONLY place code runs. Shell + file ops confined to /workspace.

Convenience boundary, not a security one (see plan.txt §7).
"""
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
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


@app.get("/health")
def health() -> dict:
    return {"ok": True, "workspace": str(WORKSPACE)}
