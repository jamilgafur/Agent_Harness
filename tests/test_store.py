"""Store: WAL + zlib-compressed snapshots, legacy-plain-JSON fallback."""
import json
import sqlite3
import zlib

from harness.models import Node
from harness.store import Store


def _run(rid="r1") -> dict:
    return {"id": rid, "created": 1.0, "intent": "test intent", "phase": "planning",
            "constraints": ["c1"], "prompts": {}, "params": {},
            "nodes": {"root": Node(id="root", intent="root task"),
                      "root.1": Node(id="root.1", intent="a search", tool="web_search",
                                     result={"results": [{"title": "t", "url": "u",
                                                          "page_text": "x" * 20000}]})},
            "stats": {"llm_calls": 1, "execs": 0, "approvals": 0, "patches": 0}}


def test_roundtrip_compresses_and_restores(tmp_path):
    db = tmp_path / "t.db"
    Store(db).save_run(_run())
    # the nodes column must actually be compressed (zlib magic 0x78), not plain JSON
    raw = sqlite3.connect(db).execute("SELECT nodes FROM runs WHERE id='r1'").fetchone()[0]
    assert isinstance(raw, (bytes, bytearray)) and json.loads is not None
    assert raw[:1] == b"\x78", "nodes column should be a zlib stream"
    loaded = Store(db).load_runs()
    n = loaded["r1"]["nodes"]
    assert n["root.1"].tool == "web_search"
    assert n["root.1"].result["results"][0]["page_text"] == "x" * 20000
    assert loaded["r1"]["stats"]["llm_calls"] == 1
    assert loaded["r1"]["constraints"] == ["c1"]


def test_legacy_plain_json_rows_still_load(tmp_path):
    """DBs written before compression hold plain JSON text — no migration allowed."""
    db = tmp_path / "t.db"
    Store(db).save_run(_run())
    plain = json.dumps({"root": Node(id="root", intent="root task").model_dump()})
    sqlite3.connect(db).execute(
        "UPDATE runs SET nodes=? WHERE id='r1'", (plain,)).connection.commit()
    assert Store(db).load_runs()["r1"]["nodes"]["root"].intent == "root task"


def test_wal_mode_active(tmp_path):
    db = tmp_path / "t.db"
    Store(db).save_run(_run())
    assert sqlite3.connect(db).execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_compress_decompress_symmetry_is_lossless():
    blob = zlib.compress(json.dumps({"a": [1, 2, 3] * 100}).encode())
    assert json.loads(zlib.decompress(blob)) == {"a": [1, 2, 3] * 100}
