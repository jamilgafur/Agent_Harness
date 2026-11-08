## What changed

<!-- One paragraph: what this PR does and why. Link the issue it closes. -->

Closes #

## Why this shape

<!-- The codebase makes specific bets (small-model assumption: deterministic
     code fixes what the model gets wrong structurally; approvals gate the UI,
     not the network; one image, three services). If your change leans on one
     of them — or against one — say so here. See plan.txt for the history. -->

## Tests

<!-- What you ran. CI (ruff + pytest + pip-audit + CodeQL + docker build)
     must be green. New behavior needs a test; edge cases in the
     approval/lease/state-machine paths are pinned in tests/test_core_api.py
     on purpose. -->

- [ ] `docker run --rm -v "$PWD":/app -w /app harness:latest python -m pytest tests/ -q`
- [ ] `ruff check harness tests`
- [ ] New/changed behavior is covered by a test
- [ ] README / API table / plan.txt updated if the wire format or UX changed
