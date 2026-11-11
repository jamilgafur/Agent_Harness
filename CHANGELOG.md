# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Ensemble coordination (`harness/ensemble.py`)**: sampling → debate →
  voting over several attempts at the same node. Per-run params `samples`
  (1–5) and `debate_rounds` (0–3): llm-writer synthesis draws K drafts,
  revises each against its rivals, and a judge ballot picks the winner;
  decomposition samples K plans and votes on shaped plan digests; the judge
  becomes a strict-majority panel of K. New `debate`/`vote` prompts are
  per-run overridable like the rest. Off by default — samples=1,
  debate_rounds=0 keeps the exact pre-ensemble call pattern (all coordination
  calls ride the budget governor's `llm_calls` accounting).
- **`examples/audit_run.py`**: the product loop demonstrated and audited —
  intent → decomposed DAG → human checkpoints (plan approval, exec cards,
  edit-and-resubmit) → grounded answer, in-process. Checks: tree shape,
  first-pass success + repair recovery, grounding (cited URLs ⊆ gathered
  URLs), simulated-restart persistence, human-attention ledger, budget
  accounting. Deterministic offline mode is CI-verified
  (`tests/test_audit_run.py`); `--live --interactive` measures a real model
  and a real human at the cards.
- **`examples/benchmark_ensemble.py`**: measured accuracy / LLM-calls /
  wall-time of baseline vs sampled vs debated configs on a checkable
  arithmetic suite, as markdown — verifies measured calls against the exact
  cost model `samples × (1 + debate_rounds) + vote`; results reproduced in
  the README's Measurements section.
- **`examples/benchmark_decomposition.py`**: the product claim measured —
  flat (one call must answer every part of a multi-part question) vs tree
  (decompose → branch evidence → branch writers → head synthesis), same
  model on both arms, invented-fact corpora both arms receive in full
  (memory cannot cheat the comparison), deterministic per-part substring
  scoring. The canned sandbox serves the corpus through whichever
  retrieval tool the plan picks, so tool choice is not the measured
  variable. Tree arm audited (nodes, depth, grounding) like
  `audit_run.py`; results in the README's decomposition section.
- **`examples/intent-e2e.sh`**: live intent-run acceptance suite — five diverse
  tasks (calculator app, seeded-bug fix, CSV report, parallel-decomposition
  benchmark, invented-facts QA), each checked on BOTH product claims: plan
  shape after `plan_ready` (deps, branches, depth, tool mix, synthesis sink)
  and part-coverage scoring of the head answer. `--selfcheck` validates the
  checkers offline; runs pass/fail with a wedge dump when a run never settles.
  Suite grown from 8 to 12 tasks (+4 distractor-rich corpora) and the
  report now separates **pipeline reliability** (did the run settle) from
  **content quality** (parts a settled run kept) with a per-task matrix.
- **Runtime skills (`POST /skills`)**: the AI or the UI registers a new
  tool while the stack runs — a declarative spec (name, desc, required
  args, shell template) becomes a `/workspace/tools/*.py` module (sandbox
  generates it, args shlex-quoted), the registry hot-re-syncs, and the
  planner can pick it at the very next claim. First use keeps the human
  gate (unvetted card). Custom python tools unchanged: drop a `.py` in
  `/workspace/tools`.

### Changed

- **Escalation now collapses dead branches** (`core._collapse_dead_branch`):
  a WAITING parent whose children all went terminal can never settle
  (settle requires all-DONE) — it escalates too, bubbling to the root, so
  a failed branch ends the run honestly instead of leaving unclaimable
  WAITING nodes spinning (measured: 3/8 benchmark runs never terminated).
- **Writer acceptance is honest about failure** (`writing._llm_ok`): whole-
  answer no-material cards like `[Missing specific recipe details]` fail
  the deterministic check, and the Sources footer no longer pads a thin
  body past the length bar — junk chains can no longer pass judging into
  run heads. Repair retries now also see WHY the previous attempt was
  rejected (hint rides the synthesis material).
- **Shell verdicts are deterministic** (`agent/judging.py`): `rc==0` is the
  only honest signal — a 3b LLM judge flunks large stdout dumps, which
  escalated entire benchmark runs over correct greps. Content predicates
  stay as planner guidance, same as writers.
- **User patches re-arm expired budgets** (`core.patch_run`): `max_seconds`
  counts from `created`, so fix&retry on a budget-terminated run died
  again within a second; a user patching the run is the supervision the
  budget stands in for. The plan bar also gains a "start partial plan
  anyway" escape hatch for planning stalls (`plan_ready` never firing).
- **UI redesigned** (`static/index.html`): design-token theme, header/
  tab chrome, three-column rail (node inspector · tools & skills panel ·
  workspace files), folded developer section, blur-backdrop modal. No
  build step; same single file.

- **Agent is a package (`harness/agent/`)**, split by duty with a
  mixin-inheritance structure — `base.py` (BaseAgent: transport, heartbeat,
  approvals), `planning.py` (PlannerMixin: decompose + deterministic
  shaping), `writing.py` (WriterMixin: grounded synthesis), `judging.py`
  (JudgeMixin), `worker.py` (`Agent` = mixins composed + claim loop).
  `python -m harness.agent` and the old function-level imports are unchanged;
  loop tunables (POLL, APPROVAL_TIMEOUT, …) live in `harness.agent.base`.

### Fixed

Each entry replays a measured live failure; the regression suite pins it.

#### Plan shaping

- **Plans no longer use one tool for everything.** The decomposition example now
  shows two different tools instead of repeating the registry's first one — the
  model imitates the example literally, and the single-tool example produced
  shell-for-everything plans (file reads as fragile greps, fixes planned as test
  runs).
- **Plans no longer route local-file work to the web.** Bare subtasks in a task
  naming a workspace file now ground onto the registry's path-taking gatherer
  (`read_file`) instead of `gather_tools[0]` — grounding produced web queries
  like "Extract total revenue" for a local CSV that return nothing usable. A
  path-required tool with no `ground` template now grounds to its `path` field
  instead of an invalid `query`.
- **Invented scripts get created before they run.** A shell invoking a bare
  script nobody writes (`python analyze_sales.py`) is split: the child becomes
  the script write (abstract — the sub-plan supplies real code) and the run is
  appended after it. The model regenerates invented scripts 3x per child even
  with `can't open file` in the hint (live T3), so repair alone never saved
  these steps. `-c`/`-m` invocations and scripts a sibling writes are untouched,
  and the split never fires inside a file-producing task (it recursed to 35
  nodes / depth 6 live).
- **The mutation sub-plan's sink produces the file.** A task promising corrected
  contents reframes the sink child to 'write the corrected contents of X' —
  framed as analysis ('locate the failing test') the sink answered prose and the
  write-back shipped it over the source (live T2). A 'to `path`' deliverable
  keeps the analysis framing (its text is prose for the report).
- **Plans never contain forward deps.** The synthesis sink moves to the end of
  the child list before its 'deps = all siblings' wiring — a top-up appended
  after the model's own sink pointed forward and core rejected the whole plan
  (live T3: `cannot decompose` x3, 1-node run).
- **Mutation sub-plans actually write the fix.** 'write the corrected contents
  of `X`' (the mutation top-up's own child intent) now shapes into read → sink →
  write of `X`, bound to the sink's text; previously the sub-plan ran a read and
  stopped (live T2: the bug survived). The trigger is narrow: 'summarize/read
  the contents of `X`' never marks `X` for rewriting.
- **Reads don't count as writes.** The mutation top-up ("fix X in `path`") only
  skips when some child actually writes the path; a `read_file` child carrying
  the path in its args used to suppress it and the bug survived.
- **Named input files get read whole.** A task naming a workspace file gains a
  read child on the registry's path-taking gatherer when nothing else reads it —
  field-extraction shells ('grep X | awk') left the sink with a guessed field and
  none of the requested facts. Deliverables are never read back.
- **Named deliverables get planned as writes.** An intent promising a file
  ("write a summary report to `data-lab/report.md`") gains a `write_file` child
  bound to the sink's text (`$from`), and an intent naming an in-place change
  ("fix the bugs in `bugfix-lab/analyze.py`") gains a mutation child — live runs
  computed answers while the promised file never appeared and the bug survived.
  Mutation targets stay abstract so the sub-plan writes the real content;
  read-only files named in the intent are never flagged for rewriting.

#### Write integrity

- **A source-file write needs a real statement.** The code gate requires a
  parsed module with a statement of substance — a bare placeholder like
  `fixed_content` parses as a lone expression and live T2 shipped exactly that
  into `analyze.py`, waving echo/template/parse checks on the way.
- **A `*.py` write lands as code, not analysis.** Synthesis text wraps the
  corrected file in prose and markdown fences — live T2's complete fix sat
  inside ` ```python … ``` ` while the write carried the analysis and every
  attempt was (correctly) refused. The payload now resolves `$from` bindings
  and, when the text doesn't parse, takes the largest fenced block that does.
- **A `*.py` write must be valid Python.** Unparseable content is dropped at
  plan time (so the fix top-up fires) and fails the judge at exec time (so
  repair retries instead of shipping it): live T2 wrote the status line
  `updated code in bugfix-lab/analyze.py` over the source and pytest died on it
  (rc=2) while the bytes check called it DONE.
- **Echo and template writes are dropped.** A `write_file` whose content is the
  task's own intent text or an unfilled `{{placeholder}}` template is not
  content — live T2 wrote instruction prose over source code (breaking it) and
  live T3 wrote `{{total_revenue}}` into the deliverable, both suppressing the
  top-ups that would have planned real writes. All-garbage plans get re-planned
  through the existing repair loop.

#### Answer assembly

- **The head node always holds the answer.** Root-level decompositions converge
  on one synthesis sink even when the plan is pure task shape (write_file/shell);
  file-based plans previously produced no writer child, so `root.result` ended as
  a dict of raw tool dumps with no `text` — `run-with-overrides.sh`'s
  `root.result.text` was empty.
- **The head answer carries tool output, not just sibling prose.** Head
  assembly collected only `text`-shaped dep results (other sinks) — shell
  stdout and read_file content were dropped outright, so a tool-only run found
  zero sections and the answer collapsed to the model's first sentence (live
  T5: 2 of 3 requested facts sat verbatim in the dep results and were lost).
  Any surviving material now becomes a verbatim section — including a single
  one, which regeneration paraphrased away (live T5: the lone read_file result
  held `Petra Lund` and the chat answer dropped the name). Only a head with no
  material at all falls back to one plain synthesis call.
- **A failed leaf no longer empties the head answer.** The synthesis sink is
  exempt from failure cascades and runs once its deps are terminal (resilient
  claim), answering from surviving material; when some siblings fail, the parent
  partial-settles with the writer's text instead of escalating answerless
  (live: 4/5 runs ended `ESCALATED` with 0-char answers). `$from` deliverable
  writers get the same survival — live T3's report write was cascade-blocked.

#### Liveness & repair

- **Runs can no longer deadlock after a branch dies.** A parent escalated by
  `_collapse_dead_branch` now blocks its dep-consumers too, and the collapse
  re-check fires on the DONE path when the last live sibling finishes after an
  earlier failure — both gaps left `WAITING` parents with all-terminal kids,
  holding `active` true until the budget governor's 2-hour cap
  (`examples/intent-e2e.sh` caught both variants live).
- **Repair retries actually repair.** A tool node whose execution failed now
  re-derives its `args` (and `success_predicate`) from the failure hint before
  re-executing — retries used to replay byte-identical args, so a wrong command
  ran three times and escalated. Repair generation is best-effort: an LLM hiccup
  replays the old args rather than stalling the node.
- **State predicates no longer fail on written files.** `read_file`, `write_file`
  and `list_dir` declare a `check` result path and are judged deterministically
  from the result — an LLM judge shown `{bytes: N}` rejected correctly written
  files ("the file contains …" is not visible in the result).
- **Runtime errors refine the code that caused them.** A terminal failure whose
  error names a workspace file reopens the chain that authored it: the content
  source re-synthesizes with the traceback in hand, the write re-lands, and the
  failed node re-runs behind the write (dep-ordered). Generated code failed at
  run time (`ModuleNotFoundError`, wrong column names) while its writer sat
  DONE and every runner burned retries against the broken artifact — the error
  never reached the author. Bounded: one refine per source, one retry per node.
  The runner's re-run is deferred behind the write's DONE report — no dep
  edges are touched: the runner's output feeds the sink, so an edge back closed
  a cycle and deadlocked a whole run (live: 1800s timeout). Only an execution
  error tail (`| rc=…`) attributes a failure to a file — the success predicate
  names every task path, and a predicate echo once rewrote a working fix into
  test code (live T2).
- **Failed tool hints carry the real error.** A judged-FAIL report now appends
  rc and the stderr/stdout tail — repair re-derives args from that hint, and
  without it the model regenerated the same invented script 3x while
  `can't open file 'analyze_sales.py'` never surfaced (live T3).
- **Repair never touches `$from` bindings.** They are structural; the repair
  model mangled them (live T2: bind id `root/analyze.py`) and the exec was then
  rejected outright, burning every retry.
- **`cannot decompose` hints show the real reason.** The repair loop's final
  hint kept the last rejection detail instead of overwriting it with `None`.
- **`POST /policy` depth bound matches the governor's own config.** `max_depth`
  clamps to `DEFAULT_MAX_DEPTH` (17) instead of 12, so a UI settings save can no
  longer shrink a configured depth.

## [0.1.0] — 2026-11-07

Initial public release.

### Added

- **Core** (FastAPI): DAG store + validation, plan checkpoint, atomic claim
  handout (Algorithm-2 routing with visited-set exclusion, claim leases,
  locality preference), budget governor (LLM calls / seconds / depth / nodes),
  approvals with edit-and-resubmit, patch validation, event log, WebSocket
  feed, single-page UI with live DAG canvas.
- **Agents**: claim → decompose (LLM, structured outputs + repair loop) or
  exec → judge → report; redirect handoffs; heartbeat liveness; llm-down
  release path (a wedged runner doesn't spend repair attempts).
- **Sandbox**: the only place tools run — shell/files (workspace-confined,
  clipped, timed), `web_search` (SearXNG, reads every hit's page text),
  `http_fetch`, hot-reload **Python tools**, **MCP** stdio servers
  (mtime-watched manifest), dynamic tool discovery.
- **Deterministic small-model scaffolding**: decomposition shape pinned in
  code (synthesis sink convergence, branch deepening, named-part coverage),
  so 3b-class models drive real runs; per-run prompt/param overrides.
- **Document-shaped runs**: named-parts detection, cross-branch writer
  wiring, knowledge tree (`GET /runs/{id}/knowledge` + UI tab), doc-aware
  head assembly.
- **Data flow**: `{"$from": node, "path": …}` arg bindings resolved
  server-side at exec time; `check` (dotted result path, judged without the
  LLM) and `ground` (arg template) tool declarations.
- **Checkpointing**: SQLite run snapshots (WAL journal, zlib-compressed
  nodes column, legacy-plain-JSON fallback), approvals, capped event log —
  core resumes exactly after restart.
- **Loop nodes** with bounded requeue + `max_iterations` fallback to
  escalation; **metrics** endpoint (latency, TSR, FRT, delegation depth,
  first-pass success); **save-as-template** snapshots; unattended
  template-DAG acceptance path (0 LLM calls, 0 approvals).

### Security

- API binds `127.0.0.1` only — there is no auth; approvals gate the UI, not
  the network (see SECURITY.md).
- Path confinement (`Path.is_relative_to`), output clipping, per-command
  timeouts in the sandbox.
- Stale-approval replay blocked (attempts epoch + terminal-state gate);
  unvetted (post-startup-synced) tools park a first-use approval card.
- Pinned dependencies; CI runs ruff, pytest, pip-audit, CodeQL, and a
  Docker build on every push/PR.
