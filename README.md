# Agent Harness

<p align="center">
  <img alt="CI" src="https://github.com/jamilgafur/Agent_Harness/actions/workflows/tests.yml/badge.svg">
  <img alt="License: MIT + BSL 1.1" src="https://img.shields.io/badge/license-MIT%20%2B%20BSL--1.1-blue.svg">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-blue.svg">
  <img alt="Docker" src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white">
</p>

A Claude-Code-style agent you can **watch and audit**: you give it one sentence
("tell me about …", "write a research paper on …"), a local LLM decomposes it
into a task DAG, and agents traverse that graph — decomposing, researching,
writing — until every node settles. Unlike a chat log, the DAG *is* the audit
trail: every intent, dependency, tool call and result is persisted, validated,
and replayable. Works fully offline on a small local model, or pointed at a
bigger one for better prose — the structure is the same.

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

## Features

- **Plan first, then execute** — intent runs decompose into a DAG you review and
  edit at a checkpoint (constraints, add branches, re-plan subtrees) before
  anything executes; execution then runs **bottom-up** until the head node
  holds the final answer.
- **The DAG is the audit trail** — every node, patch, approval, exec and result
  is persisted (SQLite) and replayable; an event log per run; run snapshots
  resume exactly after a restart.
- **Real multi-agent mesh** — workers claim leaves atomically (Algorithm-2
  routing with visited-set exclusion, claim leases, redirect handoffs);
  scale with `--scale agent=N`.
- **Approval gates with editable args** — anything not auto-approved parks a
  card you can approve, edit-and-resubmit, or reject; stale cards can't replay.
- **Extensible tools, hot-reloaded** — built-in shell/files/web tools plus
  drop-in **Python tools** and **MCP servers** discovered from the sandbox;
  data flows into tool args via `{"$from": node, "path": …}` bindings.
- **Built for small models** — decomposition shrinks every LLM call to one
  small bounded job; reply *shape* is enforced by structured outputs;
  deterministic scaffolding fixes what 3b models get wrong structurally.
- **Sampling + debate + voting** — per-run params `samples` (1–5) and
  `debate_rounds` (0–3) coordinate several attempts at the *same* node:
  K sampled draft answers/plans, rival-aware revision rounds, one judge
  ballot picks the winner; judges become a strict-majority panel. Off by
  default (samples=1, rounds=0 = exactly today's call pattern).
- **Budget governor** — per-run caps (LLM calls, seconds, depth, nodes) that
  terminate a run (work kept) instead of letting it run away.
- **Runs without any LLM** — template-DAG runs of auto-approved tools execute
  unaided (the offline acceptance test: 5/5 DONE in ~2s, 0 LLM calls).

## Prerequisites

- **Docker** + Docker Compose (v2). That's the only hard requirement.
- **An LLM via [Ollama](https://ollama.com)** — either the bundled GPU profile
  (`--profile local-ollama`) or any reachable Ollama endpoint
  (`OLLAMA_URL`). Small models (3b) work; bigger is better for prose.
- Optional: a **SearXNG** instance for the `web_search` tool.

## Quickstart

```sh
git clone https://github.com/jamilgafur/Agent_Harness.git
cd Agent_Harness
docker compose up -d --build core sandbox agent        # + '--profile local-ollama up' for a bundled GPU ollama
# open http://localhost:8100
```

1. Hit the **＋** tab, keep *intent*, type: `tell me about the difference
   between structured and unstructured pruning of neural networks`
2. The planner proposes a DAG: parallel `🔍 web_search` branches per subtopic,
   then one `✍️ llm` synthesis node depending on all of them. Bad plans are
   rejected server-side and the agent repairs them itself.
3. Approve the plan in the 📋 bar (**✔ start**) — or watch it decompose first:
   nothing executes until you approve.
4. Agents claim ready nodes and gather material (read-only tools run unaided
   by default; `shell`/`write_file` still ask — click the flashing card).
   Watch wires turn green as the train of thought climbs level by level.
5. Click the green **📄 RESULTS** node (or the head node) to read the answer —
   download it as Markdown, copy it, or re-run the plan as a template.

Also try: `write a short research paper on mixture-of-experts routing, with references`.

### The two run shapes

| Shape | Intent example | What the planner builds |
|---|---|---|
| Research / "tell me about X" | `tell me about pruning` | parallel `web_search` branches → `llm` synthesis (the answer) |
| Task / "do X" | `add tests to the module and run them` | `write_file`/`shell` chain deps'd left→right |

`examples/` has runnable ones: `acceptance-demo.sh` (unattended, ~2s),
`run-with-overrides.sh` (per-run prompt/param overrides), and template DAGs
(`write-paper.json`, `add-tests.json`, `codebase-inventory.json` — paste into
**＋ → template DAG**).

## How a run works

1. **Decompose** — agents break the request into atomic leaves; nothing
   executes. Tool nodes are held; only planner (abstract) nodes are claimed.
2. **Your checkpoint** (the 📋 plan bar) — review the tree and shape it:
   - constraints/instructions (`ignore methodology`, `discuss cost`) ride every
     decomposition + synthesis prompt from then on;
   - **⚙ prompts & parameters** — per-run prompt/param overrides (see below);
   - **＋ add a branch** (top-level or under any node) — the LLM fleshes it out
     using neighbor context (parent + sibling branches), never duplicating them;
   - **✎ edit intent** rewrites a step; **✏️ break down further** re-plans a subtree.
3. **✔ start** — execution runs **bottom-up**: leaves gather, each writer
   condenses only its neighbors, summaries climb level by level — and the
   **head node ends holding the final answer**.

Document-shaped tasks ("…with an abstract, introduction, background and
conclusion") are treated as documents: the named pieces **always** become
branches, and each section's writer is **wired to the earlier sections**
(cross-branch deps) — a real interconnected document, not isolated blurbs. The
run's accumulated text is queryable as a **knowledge tree**
(`GET /runs/{id}/knowledge`, UI 🌳 tab); writers and planners read it so later
decomposition builds on knowns instead of re-gathering them.

Template-DAG runs skip the checkpoint: you already wrote the plan, they just run.

### The model proposes, the harness decides

Deterministic shaping sits between the model's plan and execution — code, not
prompt text (`DEFAULT_PROMPTS` is untouched; the regression suite replays every
measured failure this list fixes):

- **The head always answers.** Synthesis writers survive dead siblings and run
  once their deps are terminal; a parent with one dead child partial-settles
  with the writer's text instead of going up answerless.
- **Named deliverables land.** "…write X to `path`" gains a read → sink →
  write chain whose write-back binds the sink's text; "write the corrected
  contents of `path`" writes the file — "summarize the contents of" never does.
- **Code files land as code.** A `*.py` write must parse and contain a real
  statement: task echoes, `{{templates}}` and status lines are dropped at plan
  time and refused by the judge, and when analysis prose wraps the fix in
  fenced code the largest valid block is what lands.
- **Invented scripts get created.** A shell running a bare `X.py` nobody writes
  is split into write-then-run.
- **Runtime errors reach the author.** A terminal failure whose error names a
  workspace file re-synthesizes the content source with the traceback in hand,
  re-lands the write, and re-runs the failure behind it (bounded, once per
  source). Predicate echoes never trigger it.
- **Plans stay runnable.** No forward deps (the sink moves last), reads never
  count as writes, named input files get read whole, and failed hints carry
  rc + stderr so repair sees why the call died.

## Adapt it (prompts, parameters, models)

Small models suffice here: decomposition shrinks every LLM call to a small,
well-bounded job — plan one node, condense one neighborhood. The harness is
domain-generic (research/search is just the default vocabulary). Per run, you
can override:

- `decompose` / `synthesize` / `judge` — the planner / writer / judge prompts;
- `model` ("" = env default), `temperature` (0–1), `child_cap` (1–6),
  `gather_tools` (which tools count as evidence-gatherers), `sink_tool`
  (the virtual writer's name), `max_results` (3–10);
- `samples` (1–5) and `debate_rounds` (0–3) — ensemble coordination
  (see `harness/ensemble.py`): candidate answers/plans are sampled, debated
  and voted; calls are sequential on the one local runner, so cost is
  additive per node.

UI: the **⚙ prompts & parameters** editor in the 📋 plan bar; any node's args
are editable from its detail panel — both apply while planning. API:

```sh
curl -X PUT localhost:8100/runs/$RUN/config -H 'content-type: application/json' \
  -d '{"prompts": {"synthesize": "Write it as an executive brief, bullets only."},
       "params": {"child_cap": 3}}'
```

## Measurements: what the ensemble buys you

Small local models are brittle in two specific ways the ensemble addresses:
**confident-wrong answers** (rate problems, trick qualifiers) and **flaky
verdicts** (a single judge misreading a fine execution). Sampling draws K
independent attempts, debate lets attempts correct each other, and one judge
ballot picks the winner — at an **exact, deterministic cost**:

| config | LLM calls per node | formula |
|---|---|---|
| baseline (`samples=1`) | 1 | — |
| sampled (`samples=K`) | K+1 | K drafts + 1 vote |
| debated (`samples=K, debate_rounds=R`) | K×(1+R)+1 | K drafts + K·R revisions + 1 vote |

Measured on this machine with `examples/benchmark_ensemble.py` (local Ollama
0.40, GTX 1660 Ti, 2026-11-09 — your numbers will differ with hardware and
model; the calls column never does):

**`llama3.1:8b` and `qwen2.5:3b`, full 8-task suite each, temperature 0.2:**

| model | config | accuracy | LLM calls/task | wall s/task |
|---|---|---|---|---|
| llama3.1:8b | baseline | 8/8 | 1.0 | 25.3 |
| llama3.1:8b | sampled (`samples=3`) | 8/8 | 4.0 | 97.6 |
| llama3.1:8b | debated (`samples=3, debate_rounds=1`) | 8/8 | 7.0 | 181.5 |
| qwen2.5:3b | baseline | 8/8 | 1.0 | 13.5 |
| qwen2.5:3b | sampled (`samples=3`) | 8/8 | 4.0 | 48.6 |
| qwen2.5:3b | debated (`samples=3, debate_rounds=1`) | 8/8 | 7.0 | 96.2 |

Reading it honestly: **both models already ace checkable word problems — even
the rate traps — so there was no accuracy headroom to buy.** The mechanism
(sampling, rival revision, one ballot) ran end-to-end on both, measured calls
matched the cost model in all 48 task-runs, and debating cost 7.1× the wall
time to gain 0 accuracy. That is exactly why `samples=1, debate_rounds=0`
ships **off**: on well-specified tasks an ensemble is pure spend. Where it
pays: genuinely flaky models/tasks, and the **judge panel** (`samples>1` on
judging only), because a single small-model verdict is the brittliest call in
the system (`harness/agent/judging.py` documents the failure mode; agent-side
acceptance had to go deterministic because of it).

Reproduce with your own model/hardware (numbers are hardware- and model-bound):

```sh
pip install -r requirements.txt
python examples/benchmark_ensemble.py --samples 3 --rounds 1        # full 8-task suite
python examples/benchmark_ensemble.py --skip 4                      # the trap-question half
MODEL=qwen2.5:3b python examples/benchmark_ensemble.py --limit 2    # smoke
```

The script also asserts the measured call count matches the cost model —
the *accuracy* column varies with model and task; the *cost* column never
surprises. Treat suite accuracy as a smoke signal: 4–8 problems is enough to
show the mechanism works, not to rank models — rerun it on tasks from your
own domain (the suite is a plain list in the script).

## Measurements: does decomposition beat one flat call?

The product hypothesis is that decomposing a multi-part question into a tree
(gather evidence per part → branch writers → head synthesis) beats one
monolithic call. `examples/benchmark_decomposition.py` measures it honestly:

- **12 tasks × 4 extractable parts** on invented-fact corpora (memory cannot
  cheat: the facts exist nowhere else), identical evidence on both arms —
  flat gets it inline, the tree gets it through canned tool results *whatever
  retrieval tool its plan picks* (tool choice is not the measured variable).
- **Scoring is deterministic** substring-per-part — the eval never asks an LLM
  to grade itself. The tree arm is audited for nodes/depth/grounding.
- The report separates **pipeline reliability** (did the run settle at all)
  from **content quality** (parts a *settled* run kept), because pre-fix runs
  conflated the two.

Measured on this machine (Ollama 0.40, GTX 1660 Ti, temperature 0.2):

| model | arm | parts covered (48) | tasks fully correct (12) | calls/task | s/task | tree settled |
|---|---|---|---|---|---|---|
| qwen2.5:3b | flat | 46 (95%) | 10 | 1.0 | 7 | — |
| qwen2.5:3b | tree | 26 (54%) | 5 | 5.2 | 84 | 10/12 |
| llama3.1:8b | flat | 41 (85%) | 8 | 1.0 | 14 | — |
| llama3.1:8b | tree | 7 (14%) | 1 | 8.0 | 211 | 3/12 |

**The honest reading: on this suite, at 3b–8b scale, the claim does NOT
hold — flat wins, and it's not close.** 4–5 sentence corpora fit in every
context; decomposition can't add evidence, only move it around, and every
funnel (branch summary → parent) instead loses parts (settled qwen tasks:
tree 26/40 vs flat 38/40). Two hard-won facts about *where* the tree dies:

1. **Reliability dominates quality.** Across the fix campaign, qwen settle
   rate went 4/12 → 10/12; llama stayed 3/12 because 8b produces bigger,
   redundant trees (13.9 nodes/task) whose branches more often burn their
   3-attempt repair budget. Every unsettled run now terminates *honestly*
   (escalated, diagnosed) instead of spinning — before the fixes, stalled
   runs silently returned empty heads and scored 0 anyway.
2. **The ground truth held everywhere**: grounded 12/12 on both models —
   every citation the tree emitted traced to gathered pages, even in failed
   runs. Whatever the tree writes, it doesn't invent citations; the funnel
   loses parts but never their provenance.

What broke along the way (all fixed, covered by tests): LLM judges flunking
correct shell output, no-material "answer cards" passing acceptance checks,
oblivious retry loops, dead branches holding runs active forever, and the
budget governor re-killing user-resumed runs within a second. The remaining
gap is plan-shape variance at small scale — not a harness hang.

Reproduce: `python examples/benchmark_decomposition.py --limit 3` (smoke, one
model) — flags `--model --skip/--limit --out`. Suite + report live in the
script; the settled-vs-reliability split prints per run.

## Live intent E2E (`examples/intent-e2e.sh`)

Five intent-only tests against the live stack with a small (3B) model — a
calculator app, a seeded-bug fix, a CSV→report, quicksort-vs-mergesort, and an
invented-facts QA — scored on both the decomposition (plan shape) and the head
answer (part coverage). Across 19 measured rounds: the calculator and the
parallel-comparison tests pass every round; the seeded-bug fix passes every
round once the mutation pipeline landed; the invented-facts test passes except
for model name-transcription slips. The CSV→report test lands its report and
the top product every time; its exact grand total is the model's arithmetic
ceiling — 3B models can't sum ten rows of decimals in-head, and their
generated aggregation code breaks. `./examples/intent-e2e.sh` runs it
(`t1`…`t5` selects one; `--selfcheck` validates the checkers offline).

## Audit a run

The claim is the pipeline, not the calls: sentence in → decomposed DAG → human
checkpoints → grounded answer out, every step inspectable.
`examples/audit_run.py` drives that whole loop in-process (real core, real
worker, LLM swapped for a deterministic double) and then audits what came out:

```console
$ python examples/audit_run.py
AUDIT — offline (scripted model + canned tools)
nodes: 8 · repairs: 1 · delegation depth: 2

root       [DONE 1x 0c plan] How transformers scale: attention cost vs sequence l
  root.1     [DONE 1x 0c plan] branch A: find primary sources on attention's quadra
    root.1.1   [DONE 2x 0c web_search] gather evidence: an n x n affinity matrix per layer
    root.1.2   [DONE 1x 1c llm] summarize the quadratic cost evidence with citations
  root.2     [DONE 1x 0c plan] branch B: find primary sources on transformer scalin
    root.2.1   [DONE 1x 0c web_search] gather evidence: power-law fits and compute-optimal
    root.2.2   [DONE 1x 1c llm] summarize the scaling-law evidence with citations
  root.3     [DONE 1x 2c llm] write the answer from both branches

CHECKS
  [PASS] settled: every node DONE: True
  [PASS] first-pass success: 7/8
  [PASS] grounding: cited URLs are gathered URLs: True
  [PASS] persistence: restart reload identical: True

HUMAN LEDGER (4 decisions):
  - approve
  - approve (web_search)
  - approve (web_search)
  - approve (web_search) [edited args]

GROUNDING: 4 cited ⊆ 4 gathered — no invented citations
BUDGET: 4 accounted of 8 observed model calls, cap 500 (successful decompose
calls ride no report) · stats: {"llm_calls": 4, "execs": 3, "approvals": 3, "patches": 3}
TRAIL: 27 events persisted
```

What each check proves:

- **settled / first-pass** — the tree converged bottom-up; one scripted flake
  (empty search results) shows the judge rejecting and the repair loop
  recovering (`root.1.1` done on attempt 2) instead of a silently bad run.
- **grounding** — every URL in the final answer is a URL the tree actually
  gathered. An invented citation flips this to FAIL. This is the check that
  makes "grounded in the graph, not memory" falsifiable.
- **persistence** — a simulated restart reloads state from SQLite and it is
  byte-identical; 27 events form the replayable trail.
- **human ledger** — the HIL story as a number: one plan checkpoint, three
  exec cards, one edited-and-resubmitted query. `--live --interactive` puts a
  real (timed) human at those cards.
- **budget** — accounted vs observed model calls are both shown, including
  the honest quirk that successful decompose calls ride no result report.

Offline mode is deterministic and runs in CI (3 tests in
`tests/test_audit_run.py` assert every invariant); `--live` swaps in your
Ollama model against the same pipeline.

## Tools

`shell`, `read_file`, `write_file`, `list_dir` (sandboxed to /workspace),
`web_search` (SearXNG), `http_fetch`, `mcp_<server>_<tool>` (dynamic), and
**`llm`** — a virtual, agent-side tool: the model answers using only the
results its DAG ancestors produced (`result.text`), so answers are grounded in
the graph, not memory.

### Add your own tools (Python or MCP)

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
from that dep's result. That's how chained workflows pass real data — a
`create_employee` id into `assign_shift`, a candidate record into each
requirement check. The binding must ride a declared dep (validation enforces
the ordering), and the approval card shows the *resolved* args.

**MCP server** (full example: `examples/mcp_hello.py`, a no-SDK stdio server) —
register any stdio MCP server in `/workspace/tools/mcp.json`:

```json
{"servers": {"hello": {"command": ["python3", "/workspace/tools/mcp_hello.py"]}}}
```

The sandbox spawns the server (newline-delimited JSON-RPC), discovers its tools,
and exposes them as `mcp_<server>_<tool>`; editing the manifest restarts only
the changed servers (mtime-driven). Broken tool files / unstartable servers are
skipped without breaking discovery.

New tools behave like built-ins: the planner can decompose onto them the moment
`/tools/sync` succeeds, and they park an approval card unless listed in
`auto_approve` — **one exception**: a tool synced *after* startup is
"unvetted", so its first use parks a card even if `auto_approve` names it (a
dropped file must not silently widen the unaided surface); approving vets it.

### Register a skill at runtime (no file drop) — `POST /skills`

The AI (or the UI's *Tools & skills* panel) can register a declarative skill
while the stack is running — a name, a description, required args, and a shell
template with `{arg}` placeholders:

```sh
curl -X POST localhost:8100/skills -H 'content-type: application/json' -d '{
  "name": "count_lines", "desc": "count the lines of a workspace file",
  "required": ["path"], "cmd": "wc -l < {path}"}'
```

Core validates the spec, the sandbox turns it into a `/workspace/tools/*.py`
module (args shlex-quoted), the registry re-syncs, and the planner can pick the
skill at its **very next claim**. It lands as *unvetted*, so the human gate
above still applies. For anything richer than a shell template, drop a Python
`.py` directly — same directory, same discovery.

## Configuration

### `harness.toml` — `[policy]`

Defaults aim at *follow-through with gates on mutation* (commented in the file
itself). Runtime-tweakable keys: `POST /policy` (`GET /policy` shows current);
the file wins at startup.

- `auto_approve = [web_search, http_fetch, read_file, list_dir, llm]` —
  read-only + synthesis tools run unaided.
- `confirm_actions = true` — anything not auto-approved (e.g. `shell`,
  `write_file`) parks a flashing card until you approve/reject (args editable).
- `approve_dag_edits = false` — plans are validated server-side; invalid ones
  bounce into the agent's repair loop. `true` = eyeball every decomposition.
- Budgets: `max_llm_calls`, `max_seconds`, `max_depth` (17), `max_nodes` (300) —
  the governor terminates a run (work kept) when breached, leaving ESCALATED
  nodes with the reason. `max_repairs` bounds the repair loop; `event_cap`
  bounds retained audit events per run; `claim_lease_seconds` requeues nodes
  whose holder went silent.

### Environment variables (compose)

Every value has a default; `.env` (see `.env.example`) is only for the knobs
you'd actually change. `HARNESS_CONFIG` overrides any config/policy key as JSON.

| Variable | Default | Where | What |
|---|---|---|---|
| `PORT` | `8100` | core | Host port for UI/API (bound to `127.0.0.1`) |
| `MODEL` | `qwen2.5:3b` | core, agent, ollama | Ollama model name |
| `OLLAMA_URL` | `http://ollama:11434` | core, agent | Any Ollama endpoint (remote GPU box supported) |
| `SEARX_URL` | `http://host.docker.internal:8095` | sandbox | SearXNG for `web_search` |
| `HARNESS_CONFIG` | — | core, agent | JSON override of any config/policy key |
| `COMPOSE_PROFILES` | — | host | `local-ollama` to bundle a GPU ollama |
| `POLL_INTERVAL` | `2` | agent | Seconds between claims |
| `APPROVAL_TIMEOUT` | `1800` | agent | Seconds before an unanswered card frees the agent |
| `OLLAMA_MAX_QUEUE` | `8` | ollama | Excess requests 503 fast (protects a single-slot runner) |
| `LLM_JSON_NUM_PREDICT` | `768` | agent | Token cap for structured (JSON) replies |
| `OLLAMA_NUM_CTX` | `4096` | ollama | Context window budget |

## API (REST + WebSocket)

Core is a plain-JSON REST API (see `harness/models.py` for request bodies):

| Endpoint | Purpose |
|---|---|
| `POST /runs` `{intent}` or `{nodes:[…]}` | start a run (intent or template DAG) |
| `GET /runs` · `GET /runs/{id}` · `DELETE /runs/{id}` | list / full state / delete |
| `POST /claim` | atomic ready-node handout (Alg-2 routing, locality) |
| `POST /runs/{id}/nodes/{nid}/decompose` | validated children (repair-loop friendly) |
| `POST /runs/{id}/nodes/{nid}/result` | DONE / FAILED — settles parents, cascades |
| `POST /runs/{id}/nodes/{nid}/redirect` | agent handoff with reason |
| `POST /runs/{id}/exec` · `GET /exec/{id}` | tool call (approval-gated) + status poll |
| `GET /approvals` · `POST /approvals/{id}` | approve (edit-and-resubmit) / reject |
| `POST /runs/{id}/patch` | graph ops (`add_node`, `add_edge`, `replace_tool`, `retry_node`) |
| `GET /runs/{id}/events?after_id=` | persisted event log (capped) |
| `GET /runs/{id}/knowledge` · `…/snapshot` | knowledge tree · save-as-template |
| `GET /tools` · `POST /tools/sync` · `POST /skills` | tool registry, hot-reload sync, runtime skill registration |
| `GET /metrics` · `GET /policy` · `POST /policy` · `GET /autosize` | paper §V metrics, runtime policy, sizing hint |
| `WS /ws` | live state / approvals / log / agents push to the UI |

## Parallel workers

One agent container = one node at a time. For parallel leaf work:

```sh
docker compose up -d --scale agent=3    # N workers, N leaves running at once
```

Claims are atomic (an agent never re-claims a node it redirected); each worker
registers as `agent-<hostname>`; `max_agents` is the concurrency gate,
`GET /autosize` suggests a number from measured LLM latency. Long LLM calls
keep their claim alive via the 5s heartbeat.

## Security

Read [SECURITY.md](SECURITY.md) before deploying. The short version:

- **No auth exists** — approvals gate *you* at the UI, they are not a security
  boundary. The compose file therefore binds the API to `127.0.0.1` only.
  Keep it that way; if you must expose the port, put auth in front.
- **The sandbox is a convenience boundary, not isolation** — code runs in a
  dedicated container with a scratch volume, docker-exec level. Fine for
  research; don't point it at untrusted tool code.
- **Audit by default** — runs, approvals (with args) and events persist to
  SQLite; don't paste secrets into tool args (bind them from results with
  `$from` instead).
- Dependencies are **pinned** in `requirements.txt`; CI runs lint, tests,
  dependency auditing and CodeQL on every push/PR (see below).

## Troubleshooting

- **UI looks stale / old cards:** hard-refresh (Ctrl+Shift+R) after every
  `docker compose up -d --build core`.
- **LLM runner wedged (every call hangs, then 500s):** don't kill LLM clients
  mid-generation — killing the client does NOT cancel the server-side
  generation (orphans pile behind the single slot until the runner crashes).
  Recover with `docker restart harness-ollama-1`. Agents self-protect: LLM
  transport failures release the node for a lease requeue (agent shows
  `llm-down`) instead of spending repair attempts.
- **Worker stuck mid-call:** `docker compose kill -s USR1 agent` dumps a live
  traceback into `docker logs harness-agent-1`.
- **Where's the audit trail:** SQLite at `/data/harness.db` (volume
  `harness_data`) — tables `runs` (snapshots), `approvals`, `events`.
- **A run is stuck with purple nodes:** click one, read the hint, use
  *fix & retry*; or delete the run (tab ✕, active runs confirm first).
- **SearXNG:** `web_search` expects a local instance (`SEARX_URL`); it also
  reads every hit's page itself, so answers cite pages, not snippets. In the
  UI, search results mark `📖 page read` vs `🔗 link only`.

## Reading the UI

- **Tab bar** — one tab per run; dot: blue=active, yellow=needs you, green=done.
- **Cards** — left rail = state (blue pulsing=running, green=done, amber=
  waiting/repairing, red=failed/blocked, purple=escalated). Icon = tool. Green
  dot top-right = this node produced output. Flashing `!` badge = pending
  approval (click to approve/reject with editable JSON).
- **Wires** — dependencies; green once the source finished.
- **Agents** — colored chips that ride the wires to the node they're working;
  the label shows their phase. A `?` means they're parked on an approval.
- **Panels** — the rail under the canvas: node inspector (click a card),
  **tools & skills** (registry plus the runtime skill form), workspace files.
  The developer panel (agents + event log) folds away at the bottom.
- Pan: drag background · Zoom: wheel · `fit all` recenters.

## Development

```sh
# tests (no GPU, no ollama needed — the sandbox HTTP layer is mocked):
docker run --rm -v "$PWD":/app -w /app harness:latest python -m pytest tests/ -q
# lint (same config as CI):
docker run --rm -v "$PWD":/app -w /app harness:latest sh -c "pip install -q ruff && ruff check harness tests"
# full local stack (dev):
docker compose up -d --build
```

- Python 3.12 (3.9–3.10 still import cleanly via the `tomllib` fallback in
  `config.py`, but 3.12 is the supported target).
- `ruff.toml` configures lint (`E/F/I/UP`); keep `ruff check harness tests`
  green before pushing.
- CI runs the same checks plus dependency audit, CodeQL and a Docker build
  ([`.github/workflows/`](.github/workflows/)).

### Layout

- `harness/models.py` — node/state/patch models, policy defaults
- `harness/graph.py` — DAG validation, atomic patches, ready set, lint
- `harness/core.py` — FastAPI: claim (Algorithm-2 routing), exec + approvals,
  budget governor, WS, static UI
- `harness/store.py` — SQLite: WAL-journaled, zlib-compressed run snapshots;
  approvals; capped event log
- `harness/agent/` — the worker, split by duty: `base.py` (transport,
  heartbeat, approvals), `planning.py` (decompose + deterministic shaping),
  `writing.py` (grounded synthesis), `judging.py` (verdict panels),
  `worker.py` (`Agent` = the mixins composed + claim loop)
- `harness/ensemble.py` — sampling → debate → vote coordination for
  answers and plans (per-run `samples`/`debate_rounds` params)
- `harness/sandbox.py` — the only place tools run: shell/files/web/dynamic/MCP
- `harness/llm.py` — Ollama client (structured outputs + repair round-trip)
- `harness/prompts.py` — single source of truth for prompts/params; per-run
  overrides resolve through here
- `harness/static/index.html` — the entire UI (no build step)
- `plan.txt` — design history & decisions · `CHANGELOG.md` — release notes ·
  `ROADMAP.md` — where this is going · `tests/` — the pytest regression suite

## License

This project is **dual-licensed** — you choose which license governs your
use ([LICENSE](LICENSE)):

1. **[MIT](LICENSE-MIT)** — unrestricted: use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell, no warranty or liability, keep the
   copyright notice. This is the default choice for most people.
2. **[Business Source License 1.1](LICENSE)** — the source-available path:
   you may copy, modify, create derivative works, redistribute and use the
   work, but **not** offer it as a competing commercial hosted service during
   the Change Period (until **2030-11-07**). Each version released under the
   BSL also converts to MIT on the earlier of the Change Date or the fourth
   anniversary of that version's first distribution.

Because the MIT option is available for any use, the BSL path functions as
the author's on-record commercial-service reservation plus a fixed,
automatic conversion schedule — if you're just building on top of this
harness, take MIT and don't think about it. See [CONTRIBUTING.md](CONTRIBUTING.md)
for how to contribute.
