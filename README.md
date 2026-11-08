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

`examples/` has recorded ones (table in `examples/README.md`): `acceptance-demo.sh`
(unattended, ~2s), `run-with-overrides.sh` (per-run prompt/param overrides),
`write-paper.json`, `add-tests.json`, `codebase-inventory.json` (template-DAG tab).

## Plan first, then bottom-up (intent runs)

Intent runs are **plan-decoupled from execution** (a decision-tree you edit):

1. **Decompose** — agents break the request into atomic leaves; nothing executes.
   Tool nodes are held; only planner (abstract) nodes are claimed.
2. **Your checkpoint** (the 📋 plan bar) — review the tree and shape it:
   - constraints/instructions (`ignore methodology`, `discuss cost`) ride every
     decomposition + synthesis prompt from then on;
   - ⚙ prompts & parameters — per-run prompt/param overrides (see 'Adapt it');
   - **＋ add a branch** (top-level or under any node) — an LLM fleshes it out
     using the neighbor context (parent + sibling branches), never duplicating them;
   - **✎ edit intent** rewrites a step; **✏️ break down further** re-plans a subtree.
3. **✔ start** — execution runs **bottom-up**: leaves gather, each writer condenses
   only its neighbors, summaries climb level by level — and the
   **head node ends holding the final answer** (branches tapped into by a user-added
   node are auto-wired into the head's writer, so nothing you add is left out).

Named-parts tasks ("…with an abstract, introduction, background and conclusion")
are documents: the named pieces **always** become branches (code tops up what the
planner merges), and each section's writer is **wired to the earlier sections** —

- cross-branch deps make later sections wait for, and cite, earlier summaries
  (a conclusion consumes background/introduction), giving a real interconnected
  document instead of isolated blurbs;
- 📋 shows those cross-branch edges; run `…with steps:` / `…including:` /
  `sections:`-style lists get the same treatment — it's list syntax, not keywords.

The run's accumulated text is queryable as a **knowledge tree**:
`GET /runs/{id}/knowledge` (nested by branch, char counts, 400-char previews) and
the UI's **🌳 knowledge** tab. Leaves are raw evidence, inner nodes are the
summaries that rose; writers and planners read it (`Known material` context) so
later decomposition builds on knowns instead of re-gathering them.

Two checkpoint helpers: the plan bar shows ⚠ **warnings** (a child restating its
parent instead of splitting it, work nobody consumes), and the 📄 output modal's
**⬇ template** button downloads the DAG reset as a reusable template
(`GET /runs/{id}/snapshot` — paste it into ＋ → template DAG).

Template-DAG runs skip the checkpoint: you already wrote the plan, they just run.

## Adapt it (prompts, parameters, models)

Small models suffice here: decomposition shrinks every LLM call to a small,
well-bounded job — plan one node, condense one neighborhood. Reply *shape* is
enforced by Ollama structured outputs (JSON schema, not prose), and the
deterministic shape scaffolding (sink/branch pinning) covers what the small
model gets wrong structurally, so the prompts carry only semantic rules. A 3b
on a home GPU can drive research runs that feel much bigger. For better
*prose*, point `OLLAMA_URL`/`MODEL` at a bigger box — the structure stays the
same.

The harness is domain-generic: research/search is just the default vocabulary.
Per run, you can override:

- `decompose` — the planner prompt: how a node breaks into children;
- `synthesize` — the writer prompt: how a node condenses its neighbors;
- `judge` — the judge prompt: how plan quality is scored;
- `model` — `""` = env default;
- `temperature` — 0-1;
- `child_cap` — 1-6;
- `gather_tools` — which tools count as evidence-gatherers for sink convergence;
- `sink_tool` — the virtual writer tool name;
- `max_results` — 3-10.

UI: the **⚙ prompts & parameters** editor in the 📋 plan bar; any node's args
are editable from its detail panel — both apply while planning. API:

    PUT /runs/{id}/config        {"prompts": {"decompose": "...", "synthesize": "...", "judge": "..."}, "params": {...}}   (blank/omitted = built-in default)
    PATCH /runs/{id}/nodes/{nid} {"intent": "...", "args": {...}}

Edits apply while planning; after `/go` the plan is frozen (fix & retry takes over):

```sh
curl -X PUT localhost:8100/runs/$RUN/config -H 'content-type: application/json' \
  -d '{"prompts": {"synthesize": "Write it as an executive brief, bullets only."},
       "params": {"child_cap": 3}}'
```

## Tools

`shell`, `read_file`, `write_file`, `list_dir` (sandboxed to /workspace),
`web_search` (SearXNG), `http_fetch`, `mcp_<server>_<tool>` (dynamic),
and **`llm`** — a virtual, agent-side tool: the model answers using only the
results its DAG ancestors produced (`result.text`), so answers are grounded in
the graph, not memory.

## Add your own tools (Python or MCP)

Both extension points live in the sandbox's `/workspace/tools/` and hot-reload —
no rebuild. After dropping files in, tell core to re-scan:

```sh
docker cp my_tool.py harness-sandbox-1:/workspace/tools/
curl -X POST localhost:8100/tools/sync   # GET /tools shows the live registry
```

**Python tool** (full example: `examples/doc_writer.py`) — a `.py` file in
`/workspace/tools/` (skip the `_` prefix; it hides the file) exporting a `TOOL`
dict and a `run()`:

```python
TOOL = {"name": "my_tool", "desc": "what it does", "required": ["query"],
        "check": "employee.id",               # optional: dotted result path that must
                                              # be truthy — judged without the LLM
        "ground": {"query": "{intent}"}}      # optional: arg template used when the
                                              # planner grounds a bare subtask on you

def run(query):
    ...
    return {"text": "..."}      # any dict; {"rc":..., "stderr":...} = failure
```

**Data flows into args too**: any arg value may be a binding
`{"$from": "<node-id>", "path": "a.b.c"}`, resolved server-side at exec time
from that dep's result (`path` digs into the dict; omit it for the whole
result). That's how chained workflows pass real data — `create_employee`'s id
into `assign_shift`, a candidate record into each requirement check. The
binding must ride a declared dep (validation enforces the ordering), and the
approval card shows the *resolved* args.

**MCP server** (full example: `examples/mcp_hello.py`, a no-SDK stdio server) —
register any stdio MCP server in `/workspace/tools/mcp.json`:

```json
{"servers": {"hello": {"command": ["python3", "/workspace/tools/mcp_hello.py"]}}}
```

The sandbox spawns the server (newline-delimited JSON-RPC), discovers its tools,
and exposes them as `mcp_<server>_<tool>`; editing the manifest restarts only the
changed servers (mtime-driven). Broken tool files / unstartable servers are
skipped without breaking discovery.

New tools behave like built-ins: the planner can decompose onto them the moment
`/tools/sync` succeeds, they park an approval card unless listed in
`auto_approve` — **one exception**: a tool synced *after* startup is "unvetted",
so its first use parks a card even if `auto_approve` names it (a dropped file
must not silently widen the unaided surface); approving vets it — and
evidence-gathering ones can join a run's `gather_tools`
override (see 'Adapt it') so the research shape converges on them.

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
  ESCALATED node with the reason. `event_cap` bounds retained audit events
  per run.
- ⚠ Approval args are persisted to SQLite for audit — don't paste secrets
  into tool args (bind them from results with `$from` instead).

Failures scope, not sink: a node that exhausts its repairs escalates only
itself and blocks its dep-consumers; the parent keeps a pointer hint, and
*fix & retry* on the leaf reopens the branch. Late approvals can't replay:
a card is pinned to the claim that parked it, and once the node moved on the
card goes `stale` without executing.

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
- **LLM runner wedged (every call hangs, then 500s):** don't kill LLM clients
  mid-generation — killing the client does NOT cancel the server-side
  generation (orphans pile behind the single slot until the runner crashes;
  the restart policy recovers it, but work still piles up). Recover instead:
  `docker restart harness-ollama-1`. Agents self-protect: LLM transport
  failures release the node for a lease requeue (agent shows `llm-down`)
  instead of spending its repair attempts, and give up with a clear
  `llm unreachable` hint once attempts are exhausted. Knobs:
  `OLLAMA_MAX_QUEUE=8` (compose: excess requests 503 fast),
  `LLM_JSON_NUM_PREDICT=768` (env: cap for structured replies — runaway JSON
  can't hold the slot for minutes).
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
- `harness/core.py` — FastAPI: claim (Algorithm-2 routing), exec + approvals,
  budget governor, WS, static UI (request bodies live in `models.py`)
- `harness/store.py` — SQLite: run snapshots, approvals, retained event log
- `harness/agent.py` — worker: claim → decompose/exec → neighbor-read → report (train of thought: writers condense their own level only)
- `harness/sandbox.py` — the only place tools run: shell/files/web/MCP
- `harness/llm.py` — Ollama client (JSON mode + repair round-trip; per-call model/temperature)
- `harness/prompts.py` — single source of truth for the decompose/synthesize/judge
  prompts and params; per-run overrides resolve through here
- `harness/static/index.html` — the entire UI (no build step)
- `plan.txt` — design history and deferred ideas · `tests/` — 117 tests (also run in CI, push/PR)

```sh
docker run --rm -v "$PWD":/app -w /app harness:latest python -m pytest tests/ -q
```

Scale agents: `docker compose up -d --scale agent=3` (claims are atomic).
Config: `harness.toml`; env wins (`HARNESS_CONFIG` JSON or per-key env vars,
e.g. `OLLAMA_URL=http://gpu-box:11434 MODEL=qwen2.5:7b`).
Agent env: `POLL_INTERVAL` seconds between claims (default 2),
`APPROVAL_TIMEOUT` seconds before an unanswered card frees the agent (default 1800).
The core UI/API binds `127.0.0.1` only — no auth exists; do not expose the port.

## License

MIT — see `LICENSE`.
