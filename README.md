# Agent Harness

A Claude-Code-style agent you can **watch and audit**: you give it one sentence
("tell me about …", "write a research paper on …"), a local LLM decomposes it
into a task DAG, and agents traverse that graph — decomposing, researching,
writing — until every node settles. Unlike a chat log, the DAG is the audit
trail: every intent, dependency, tool call and result is persisted, validated,
and replayable.

UI: **http://localhost:8100** · Everything runs in Docker.

```
┌────────┐  sentence   ┌──────────┐  decompose   ┌─────────┐  claim   ┌────────┐
│  you   │ ──────────► │   core   │ ◄─────────── │  agent  │ ◄─────── │ agents │
└────────┘             │ (FastAPI)│ ── approve ► │ planner │          │ (LLM)  │
                       │ DAG repo │              └────┬────┘          └───┬────┘
                       └────┬────┘                   │ exec              │
                            ▼                        ▼                   │
                       ┌──────────┐            ┌──────────┐              │
                       │ SQLite   │            │ sandbox  │ ◄────────────┘
                       │ + events │            │ (tools)  │
                       └──────────┘            └──────────┘
```

## Quickstart

```sh
docker compose up -d --build core sandbox agent   # + '--profile local-ollama up' for local GPU
# open http://localhost:8100
```

1. Hit the **＋** tab, keep *intent*, type: `tell me about the difference between
   structured and unstructured pruning of neural networks`
2. The planner proposes a DAG: parallel `🔍 web_search` branches per subtopic,
   then one `✍️ llm` synthesis node depending on all of them. Bad plans are
   rejected server-side and the agent repairs them itself.
3. Agents claim ready nodes, gather material (read-only tools run unaided by
   default; `shell`/`write_file` still ask — click the flashing card to
   approve). Watch wires turn green.
4. The ✍️ node reads everything its ancestors produced (agents traverse the
   DAG **upward** for context) and writes the answer. Click the green
   **📄 RESULTS** node at the right of the canvas to read it.

Also try: `write a short research paper on mixture-of-experts routing, with references`.

## The two run shapes

| Shape | Intent example | What the planner builds |
|---|---|---|
| Research / "tell me about X" | `tell me about pruning` | parallel `web_search` branches → `llm` synthesis (the answer) |
| Task / "do X" | `add tests to the module and run them` | `write_file`/`shell` chain deps'd left→right |

`examples/` has recorded ones: `acceptance-demo.sh` (unattended, ~2s),
`write-paper.json`, `add-tests.json` (template-DAG tab).

## Tools

`shell`, `read_file`, `write_file`, `list_dir` (sandboxed to /workspace),
`web_search` (SearXNG), `http_fetch`, `mcp_<server>_<tool>` (dynamic),
and **`llm`** — a virtual, agent-side tool: the model answers using only the
results its DAG ancestors produced (`result.text`), so answers are grounded in
the graph, not memory.

## Policy (approval gates) — `harness.toml` `[policy]`

Defaults aim at *follow-through with gates on mutation*:

- `approve_dag_edits = false` — plans are validated server-side; invalid ones
  bounce into the agent's repair loop. Set `true` to eyeball every decomposition.
- `auto_approve = [web_search, http_fetch, read_file, list_dir, llm]` —
  read-only + synthesis tools run unaided.
- `confirm_actions = true` — anything not auto-approved (e.g. `shell`,
  `write_file`) parks a flashing card until you approve/reject (editable args).
- Budgets: `max_llm_calls`, `max_seconds`, `max_depth`, `max_nodes` — the
  governor terminates a run (work kept) when breached, ending in an
  ESCALATED node with the reason.

Runtime tweaks: `POST /policy` (`GET /policy` shows current).

## Reading the UI

- **Tab bar** — one tab per run; dot: blue=active, yellow=needs you, green=done.
  `📄 output` opens the run's aggregate result; `＋` starts a run.
- **Cards** — left rail = state (blue pulsing=running, green=done, amber=waiting/
  repairing, red=failed/blocked, purple=escalated). Icon = tool. Green dot top
  right = this node produced output. Flashing + `!` badge = pending approval
  (click to approve/reject with editable JSON).
- **Wires** — dependencies; green once the source finished.
- **Agents** — colored chips that ride the wires to the node they're working;
  label shows their phase (`decomposing`, `executing`, `reading upstream`,
  `writing`…). A `?` means they're parked on an approval.
- **Panels** — node detail (click a card; includes a readable `── result ──`),
  workspace files (click to read), event log (full audit stream).
- Pan: drag background · Zoom: wheel · `fit all` recenters.

## Layout

- `harness/models.py` — node/state/patch models, policy defaults
- `harness/graph.py` — DAG validation, atomic patches, ready set
- `harness/core.py` — FastAPI: store (SQLite), claim (Algorithm-2 routing),
  exec + approvals, budget governor, WS, static UI
- `harness/agent.py` — worker: claim → decompose/exec → upstream read → report
- `harness/sandbox.py` — the only place tools run: shell/files/web/MCP
- `harness/llm.py` — Ollama client (JSON mode + repair round-trip)
- `harness/static/index.html` — the entire UI (no build step)
- `plan.txt` — design history and deferred ideas · `tests/` — 46 tests

```sh
docker run --rm -v "$PWD":/app -w /app harness:latest python -m pytest tests/ -q
```

Scale agents: `docker compose up -d --scale agent=3` (claims are atomic).
Config: `harness.toml`; env wins (`HARNESS_CONFIG` JSON or per-key env vars,
e.g. `OLLAMA_URL=http://gpu-box:11434 MODEL=qwen2.5:7b`).
