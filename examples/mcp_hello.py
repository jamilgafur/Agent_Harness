"""Minimal MCP stdio server (no SDK): newline-delimited JSON-RPC, one tool `greet`.

Registered in the sandbox via /workspace/tools/mcp.json, e.g.:
  {"servers": {"hello": {"command": ["python3", "/workspace/tools/mcp_hello.py"]}}}
"""
import json
import sys

TOOLS = [{"name": "greet",
          "description": "greet someone by name",
          "inputSchema": {"type": "object",
                          "properties": {"name": {"type": "string"}},
                          "required": ["name"]}}]


def reply(mid, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        continue                                   # notification: nothing to send
    if method == "initialize":
        reply(mid, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mcp-hello", "version": "0"}})
    elif method == "tools/list":
        reply(mid, {"tools": TOOLS})
    elif method == "tools/call":
        p = msg.get("params", {})
        if p.get("name") == "greet":
            who = p.get("arguments", {}).get("name", "world")
            reply(mid, {"content": [{"type": "text", "text": f"hello {who}"}]})
        else:
            reply(mid, {"content": [{"type": "text", "text": "unknown tool"}],
                        "isError": True})
