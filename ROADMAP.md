# Roadmap

Open items, roughly ordered by value. Shipped history: `CHANGELOG.md` and
`plan.txt` (the design log, including rejected ideas and their reasons).

## Performance (next)

- **"Fog" on the WebSocket state push** — broadcast the canvas view per node
  (state/intent/depth/… + a result char-count) and fetch the full node on
  demand (detail card, output modal). A research run's state push drops from
  ~200KB to ~5KB; `model_dump` stops walking page-text every broadcast.
- **Claim sweep** — one lean pending-approvals query per sweep instead of one
  per active run, and a single pass over each run's nodes (two today).
- **Pooled HTTP client** for core's sandbox/ollama calls (fresh
  `httpx.AsyncClient` per call today, including the UI's per-push file proxy).
- Deferred (after the above pays for itself): run results in their own table
  (true delta persistence) — only worth it if `/data` moves to slow disk.

## Features

- **Loop bodies as multi-node subgraphs** — loops are requeue-per-iteration
  today (`shortcut:` comment in `core.report`); unroll when a body needs
  several nodes.
- **Repair creativity** — on the last repair attempt, convert a failing tool
  node into an abstract node so the next claim re-plans instead of
  re-executing identical args (useless for "make the failing tests pass").
- **Autosizer upgrade** — queue-depth + VRAM-aware agent sizing (today:
  ollama-ping latency bands).
- **Paper metrics not yet implemented**: EWC (needs learned edge weights),
  sustained-throughput (needs multi-run load).

## Hygiene / deferred

- Split `requirements.txt` into runtime vs dev (pytest ships in the image —
  the documented `docker run … pytest` flow depends on it).
- SearXNG `formats: [json]` → drop the HTML-scrape fallback.
- Point the MCP manifest at real servers (filesystem/github) when the sandbox
  image needs them.
- The UI is one 700-line `index.html`; it predates the document-shape
  features and is the likeliest extraction target for a new contributor.

## Non-goals (deliberate — see plan.txt §7 and SECURITY.md)

- Auth / public exposure: localhost-only by design; put auth in front if you
  must expose it.
- The sandbox as a security isolation boundary: it is a docker-exec-level
  convenience boundary, not a hostile-input boundary.
- Redis/NATS-style brokers: HTTP + WebSocket is enough at this scale.
