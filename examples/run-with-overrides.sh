#!/bin/bash
# Domain-generic proof: an intent-only run with per-run prompt/param overrides.
# The planner + synthesizer obey whatever prompts and params you give them —
# here: audit for debug prints, answered as a terse table. No web tools needed.
# Usage from repo root (core + sandbox + one agent up): ./examples/run-with-overrides.sh
set -euo pipefail
BASE="${BASE:-http://localhost:8100}"

# a. intent-only run — core starts in the planning phase, an agent decomposes it.
RUN=$(curl -s -X POST "$BASE/runs" -H 'content-type: application/json' \
  -d '{"intent": "audit this workspace for leftover debug prints"}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["run_id"])')
echo "run_id=$RUN  (watch it live at $BASE)"

# b. per-run config override: terse table-style synthesize prompt + tight params.
curl -s -X PUT "$BASE/runs/$RUN/config" -H 'content-type: application/json' -d '{
  "prompts": {"synthesize": "Answer as a terse markdown table only: file | line | debug-print snippet. No prose, no headers outside the table."},
  "params": {"child_cap": 3, "temperature": 0.1}
}' >/dev/null

# c. wait for the planner to finish (max ~3 min)
READY=False
for i in $(seq 1 45); do
  READY=$(curl -s "$BASE/runs/$RUN" | python3 -c 'import json,sys;print(json.load(sys.stdin)["plan_ready"])')
  [ "$READY" = "True" ] && break
  sleep 4
done
[ "$READY" = "True" ] || { echo "plan never became ready"; exit 1; }

# d. approve the plan
curl -s -X POST "$BASE/runs/$RUN/go" >/dev/null

# e. wait for execution to run out
for i in $(seq 1 45); do
  ACTIVE=$(curl -s "$BASE/runs/$RUN" | python3 -c 'import json,sys;print(json.load(sys.stdin)["active"])')
  [ "$ACTIVE" = "False" ] && break
  sleep 4
done

# f. head node synthesizes the whole DAG
curl -s "$BASE/runs/$RUN" | python3 -c '
import json, sys
print(json.load(sys.stdin)["nodes"]["root"]["result"]["text"])'
