# Contributing

Thanks for helping — this is a research project that's used daily, so
pragmatic, well-tested changes are welcome.

## Getting started

The whole stack runs in Docker; you don't need a local Python environment:

```sh
git clone https://github.com/jamilgafur/Agent_Harness.git
cd Agent_Harness
docker compose up -d --build          # UI at http://localhost:8100
```

Run the tests and lint the same way CI does:

```sh
docker run --rm -v "$PWD":/app -w /app harness:latest python -m pytest tests/ -q
docker run --rm -v "$PWD":/app -w /app harness:latest sh -c "pip install -q ruff && ruff check harness tests"
```

Both must be green before you push. (No GPU or Ollama needed — the sandbox
HTTP layer is mocked in tests.)

## Before you open a PR

1. **Read `plan.txt`** first — it's the design history, and several
   "obvious" simplifications are already-rejected ideas (with the reasons why).
2. Keep the small-model assumption: the LLM is a *proposer*, and shape is
   enforced deterministically in code (`_force_synthesis`, `apply_patch`
   validation). Prefer fixing shape in code over preaching it in prompts.
3. Don't touch the approval/lease/state-machine semantics without reading
   `tests/test_core_api.py` — the edge cases (stale reports, stale
   approvals, double handout) are pinned by tests on purpose.
4. If you change the wire format or the UI contract, update the affected
   tests and the README's API table in the same PR.
5. No new dependencies without a strong reason (see `requirements.txt` —
   everything is pinned and everything ships in one small image).

## Style

- `ruff check harness tests` (config in `ruff.toml`) must pass.
- Comments explain *why* (invariants, incident history), not *what*.
- Shortcuts that defer a known limitation get a comment:
  `shortcut: <the limit>, <when to upgrade>` — the audit trail matters here.
- Commit messages: imperative subject + a body that says what/why. The
  history doubles as documentation.

## Good first places

- `ROADMAP.md` lists open items ordered by value.
- Test coverage for the dynamic-tool/MCP paths is thin.
- The UI is one 700-line `index.html` — it works, but it predates the
  document-shape features; extraction is fair game.

## Licensing your contributions

By submitting a PR you agree that your contributions are licensed under the
project's dual license — [MIT or BSL 1.1](LICENSE), the user's choice.
