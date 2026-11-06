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

## Plan first, then bottom-up (intent runs)

Intent runs are **plan-decoupled from execution** (a decision-tree you edit):

1. **Decompose** — agents break the request into atomic leaves; nothing executes.
   Tool nodes are held; only planner (abstract) nodes are claimed.
2. **Your checkpoint** (the 📋 plan bar) — review the tree and shape it:
   - constraints/instructions (`ignore methodology`, `discuss cost`) ride every
     decomposition + synthesis prompt from then on;
   - **＋ add a branch** (top-level or under any node) — an LLM fleshes it out
     using the neighbor context (parent + sibling branches), never duplicating them;
   - **✎ edit intent** rewrites a step; **✏️ break down further** re-plans a subtree.
3. **✔ start** — execution runs **bottom-up**: leaves gather, each writer condenses
   only its neighbors, summaries climb level by level — and the
   **head node ends holding the final answer** (branches tapped into by a user-added
   node are auto-wired into the head's writer, so nothing you add is left out).

Template-DAG runs skip the checkpoint: you already wrote the plan, they just run.

## Tools

`shell`, `read_file`, `write_file`, `list_dir` (sandboxed to /workspace),
`web_search` (SearXNG), `http_fetch`, `mcp_<server>_<tool>` (dynamic),
and **`llm`** — a virtual, agent-side tool: the model answers using only the
results its DAG ancestors produced (`result.text`), so answers are grounded in
the graph, not memory.

## Policy (approval gates) — `harness.toml` `[policy]`

Defaults aim at *follow-through with gates on mutation* (commented in the file itself):

- `approve_dag_edits = false` — plans are validated server-side; invalid ones
  bounce into the agent's repair loop. Set `true` to eyeball every decomposition.
- `auto_approve = [web_search, http_fetch, read_file, list_dir, llm]` —
  read-only + synthesis tools run unaided.
- `confirm_actions = true` — anything not auto-approved (e.g. `shell`,
  `write_file`) parks a flashing card until you approve/reject (args editable).
- Budgets: `max_llm_calls`, `max_seconds`, `max_depth`, `max_nodes` — the
  governor terminates a run (work kept) when breached, ending in an
  ESCALATED node with the reason.

Runtime tweaks: `POST /policy` (`GET /policy` shows current). File wins at startup.
⚠ Approvals gate *you*, not the network — this stack has no auth; keep the port
on a trusted LAN (see `compose.yaml` `ports:`).

## Parallel workers

One agent container = one node at a time. For parallel leaf work:

```sh
docker compose up -d --scale agent=3    # N workers, N leaves running at once
```

Claims are atomic (Alg-2 routing: an agent never re-claims a node it redirected);
each worker registers as `agent-<hostname>`; `max_agents` in the policy is the
concurrency gate, `GET /autosize` suggests a number from measured LLM latency.
Long LLM calls keep their claim alive via the 5s heartbeat — a node's claim only
requeues when the holder goes silent (`claim_lease_seconds`).

## Troubleshooting

- **UI looks stale / no ✕ on tabs / old cards:** hard-refresh (Ctrl+Shift+R)
  after every `docker compose up -d --build core`.
- **Worker stuck mid-call:** `docker compose kill -s USR1 agent` dumps a live
  traceback into `docker logs harness-agent-1`.
- **Where's the audit trail:** SQLite at `/data/harness.db` (docker volume
  `harness_data`) — tables `runs` (full snapshots), `approvals`, `events`
  (capped 500/run). Nothing is in-memory-only except agent liveness.
- **A run is stuck with purple nodes:** click one, read the hint, use
  *fix & retry*; or delete the whole run (tab ✕, active runs confirm first).
- **SearXNG:** `web_search` expects a local instance (host `:8095` via
  `SEARX_URL`); it also reads every hit's page itself (`page_text`), so answers
  cite pages, not snippets. Every writer appends a deterministic `Sources:` footer
  built from the URLs its material actually came from — citations can't be
  forgotten by the model. In the UI, search results mark `📖 page read` vs `🔗 link only`.

## Reading the UI

- **Tab bar** — one tab per run; dot: blue=active, yellow=needs you, green=done.
  `📄 output` opens the run's aggregate result; `＋` starts a run.
- **Cards** — left rail = state (blue pulsing=running, green=done, amber=waiting/
  repairing, red=failed/blocked, purple=escalated). Icon = tool. Green dot top
  right = this node produced output. Flashing + `!` badge = pending approval
  (click to approve/reject with editable JSON).
- **Wires** — dependencies; green once the source finished.
- **Agents** — colored chips that ride the wires to the node they're working;
  label shows their phase (`decomposing`, `executing`, `writing`…). A `?` means they're parked on an approval.
- **Panels** — node detail (click a card; includes a readable `── result ──`),
  workspace files (click to read), event log (full audit stream).
- Pan: drag background · Zoom: wheel · `fit all` recenters.

## Layout

- `harness/models.py` — node/state/patch models, policy defaults
- `harness/graph.py` — DAG validation, atomic patches, ready set
- `harness/core.py` — FastAPI: store (SQLite), claim (Algorithm-2 routing),
  exec + approvals, budget governor, WS, static UI
- `harness/agent.py` — worker: claim → decompose/exec → neighbor-read → report (train of thought: writers condense their own level only)
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
