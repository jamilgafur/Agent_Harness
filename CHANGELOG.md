# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
