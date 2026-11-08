# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `main` (latest release) | ✅ |
| Older tags | ❌ (fixes are not backported) |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Use **[GitHub Security Advisories](https://github.com/jamilgafur/Agent_Harness/security/advisories/new)**
to report privately. I aim to acknowledge reports within 48 hours and to
patch confirmed issues promptly. If you don't hear back, follow up on the
advisory.

## Threat model (read this before deploying)

Be explicit about what this project is *not*, so you don't get surprised:

- **There is no authentication.** The approval UI gates *you*, the human
  operator — it is not an auth boundary. Anyone who can reach the API can
  create runs, approve tool executions, and read all results. The compose
  file therefore binds the API to `127.0.0.1` only. **If you expose the port,
  you must put real authentication in front of it** (reverse proxy, VPN,
  tailnet) and understand what that unlocks.
- **The sandbox is a convenience boundary, not isolation.** Tool code runs in
  a dedicated container with a scratch `/workspace` volume, at docker-exec
  level of containment. This is fine for research on your own machine; it is
  *not* a boundary against untrusted tool code or hostile LLM output.
- **The LLM can steer the DAG, not the box.** The model proposes
  decompositions and tool calls, but every tool call passes the same
  validation, budget governor, and approval gates as a user's. An
  "unvetted" (dynamically synced) tool always parks a first-use approval
  card.
- **Audit artifacts are on disk.** Runs, approval args, and events persist to
  SQLite (`/data/harness.db`). Don't paste secrets into tool args — use
  `{"$from": node, "path": …}` bindings so data flows from results instead.

### Defense-in-depth already in place

- API bound to `127.0.0.1` by default in `compose.yaml`.
- Dependencies pinned in `requirements.txt`; CI runs `pip-audit` and
  CodeQL on every push/PR; Dependabot covers pip and GitHub Actions.
- Path confinement in the sandbox (`Path.is_relative_to`), output clipping,
  and per-command timeouts.
- Claim leases + attempts-epoch staleness gates: a slow or crashed agent can't
  double-spent a node or replay a late approval.
- Budget governor: per-run caps on LLM calls, seconds, depth, and node count
  terminate runaway runs.

## Known limitations

- The sandbox container runs as root (volume ownership trade-off); keep it on
  a trusted network.
- `web_search`/`http_fetch` follow redirects and render pages client-side;
  treat fetched content as untrusted input to the LLM, which it is.
- `HARNESS_CONFIG`/`harness.toml` are read at startup with no integrity
  checks — protect them like configuration (they're on disk next to the app).
