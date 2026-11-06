"""Doc writer (P2 dynamic tool). Drop this file into /workspace/tools/ and
POST /tools/sync on core — it becomes a tool every agent can use.

Assembles a markdown document from workspace text files.
"""
TOOL = {
    "name": "doc_writer",
    "desc": "concatenate workspace text files into one markdown document",
    "required": ["files", "output"],
}


def run(files, output):
    import os
    import pathlib
    ws = pathlib.Path(os.environ.get("WORKSPACE", "/workspace"))
    out = pathlib.Path(output)
    parts = []
    for f in files:
        p = (ws / f).resolve()
        if not str(p).startswith(str(ws)) or not p.is_file():
            return {"rc": -6, "stderr": f"not a workspace file: {f}"}
        parts.append(p.read_text(errors="replace"))
    text = "\n\n".join(parts)
    if not text.strip():
        return {"rc": -6, "stderr": "all source files empty"}
    (ws / out).parent.mkdir(parents=True, exist_ok=True)
    (ws / out).write_text(text)
    return {"bytes": len(text), "sections": len(parts), "output": output}
