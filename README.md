# Agent Harness

Research multi-agent harness: a growing task DAG decomposed by small local LLM
agents, all tool use gated through a shared sandbox with user approvals.
Everything runs in Docker. See `plan.txt` for the full design and roadmap.

## Run

```sh
docker compose --profile local-ollama up --build   # + local GPU LLM (pulls qwen2.5:3b once)
docker compose up --build                          # no local LLM (needs remote OLLAMA_URL)
# remote GPU box instead of local Ollama:
OLLAMA_URL=http://host.docker.internal:11434 MODEL=qwen2.5:7b docker compose up
# host port if 8000 is busy: PORT=8100 docker compose --profile local-ollama up
```

UI: http://localhost:8000 (live DAG, approval inbox, template DAG import).

- **Attended (default):** every tool call waits for an approval in the UI.
- **Unattended:** `HARNESS_CONFIG='{"policy":{"auto_approve":["shell","write_file"]}}' docker compose up -d`

## Layout

- `harness/models.py` — DAG node/state/patch/message models (policy defaults: depth 17)
- `harness/graph.py` — DAG validation, atomic patch application, ready set
- `harness/core.py` — FastAPI: store/checkpoint (SQLite), claim, exec+approvals, WS, UI
- `harness/agent.py` — worker: claim → exec/decompose → judge → report
- `harness/sandbox.py` — the only place code runs: shell + file ops in /workspace
- `harness/llm.py` — Ollama JSON-mode client with repair round-trip
- `tests/` — `docker run --rm -v "$PWD":/app harness pytest -q`

Scale agents: `docker compose up --scale agent=3` (claims are atomic in core).
Config: `harness.toml`, env wins (`HARNESS_CONFIG` JSON, or per-key env vars).
