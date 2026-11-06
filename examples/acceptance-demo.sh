#!/bin/bash
# Acceptance demo (answer 20 / plan P5): "add tests to repo X, make them pass"
# fully unaided — template DAG + auto_approve, zero human clicks, zero LLM calls.
# Usage from repo root: ./examples/acceptance-demo.sh | tee examples/acceptance-demo.txt
set -u
BASE="http://localhost:${PORT:-8100}"
say() { printf '\n=== %s ===\n' "$*"; }

say "0. stack health"
curl -sf "$BASE/health" || { echo "core not reachable at $BASE"; exit 1; }

say "1. enable unattended policy (auto_approve shell + write_file)"
curl -s -X POST "$BASE/policy" -H 'content-type: application/json' \
  -d '{"auto_approve": ["shell", "write_file"]}' | python3 -m json.tool

say "2. import template (examples/add-tests.json)"
RUN=$(curl -s -X POST "$BASE/runs" -H 'content-type: application/json' \
  -d @examples/add-tests.json | python3 -c 'import json,sys;print(json.load(sys.stdin)["run_id"])')
echo "run_id=$RUN  (watch it live at $BASE)"

say "3. agent works the DAG unattended (60s cap)"
for i in $(seq 1 30); do
  ACTIVE=$(curl -s "$BASE/runs/$RUN" | python3 -c 'import json,sys;print(json.load(sys.stdin)["active"])')
  [ "$ACTIVE" = "False" ] && break
  sleep 2
done

say "4. final node states"
curl -s "$BASE/runs/$RUN" | python3 -c '
import json, sys
r = json.load(sys.stdin)
for k, n in sorted(r["nodes"].items()):
    print("  %-16s %-10s attempts=%s" % (k, n["state"], n["attempts"]))
print("  stats:", r["stats"])'

say "5. repo-x on disk (sandbox workspace)"
curl -s "$BASE/workspace/file?path=repo-x/calc.py" | python3 -c 'import json,sys;print(json.load(sys.stdin)["content"])'
curl -s "$BASE/workspace/files?path=repo-x"

say "6. metrics snapshot"
curl -s "$BASE/metrics" | python3 -c 'import json,sys;print(json.dumps(json.load(sys.stdin)["total"],indent=2))'

say "7. restore confirm-everything policy"
curl -s -X POST "$BASE/policy" -H 'content-type: application/json' -d '{"auto_approve": []}' >/dev/null
echo done
