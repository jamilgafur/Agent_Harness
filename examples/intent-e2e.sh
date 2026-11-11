#!/bin/bash
# End-to-end intent tests: the two product claims measured directly.
#   1. the planner DECOMPOSES  -> every plan is shape-checked (nodes, branches,
#      depth, tool mix) after plan_ready, before /go
#   2. the head node HOLDS THE ANSWER -> part-coverage scoring on the root
#      result text (substring checks, benchmark_decomposition.py style)
#
# Five tests: t1 calculator app (codegen) · t2 fix seeded bugs (debug) ·
# t3 sales CSV report (data + synthesis) · t4 quicksort vs mergesort (parallel
# decomposition) · t5 invented-facts QA (grounded answer, memorization-proof).
#
# Usage from repo root (core + sandbox + >=1 agent up):
#   ./examples/intent-e2e.sh                 # all five
#   ./examples/intent-e2e.sh t3              # one test
#   ./examples/intent-e2e.sh --selfcheck     # verify the checkers, no stack needed
# Exits non-zero if any test fails. Budget 30-60 min on one agent with a local
# 3B model (`--scale agent=N` to parallelize); per-phase caps: PLAN_WAIT/EXEC_WAIT.
set -u
BASE="${BASE:-http://localhost:8100}"
PLAN_WAIT="${PLAN_WAIT:-300}"    # seconds to reach plan_ready
EXEC_WAIT="${EXEC_WAIT:-1800}"   # seconds to run to completion (sink-carrying trees are slow on a 3B)
POLL=4

say() { printf '\n=== %s ===\n' "$*"; }
T_FAIL=0
report() { if [ "$1" -eq 0 ]; then echo "  [PASS] $2"; else echo "  [FAIL] $2"; T_FAIL=1; fi; }

api() {  # api <METHOD> <path> [json-body]
  if [ $# -ge 3 ]; then
    curl -s -X "$1" "$BASE$2" -H 'content-type: application/json' -d "$3"
  else
    curl -s -X "$1" "$BASE$2" -H 'content-type: application/json'
  fi
}

# ---------- plan shape: deps resolve, size, parallel branches, depth, tool mix ----------
plan_check() {  # plan_check <nodes_min> <nodes_max> <branches_min> <depth_min> <parallel 0|1> <"tool>=n,...">
  python3 /dev/fd/3 "$@" 3<<'PY'
import json, sys
nmin, nmax, bmin, dmin, par = (int(x) for x in sys.argv[1:6])
spec = sys.argv[6]
run = json.load(sys.stdin)
nodes, errs = run["nodes"], []
for n in nodes.values():
    for d in n.get("deps", []):
        if d not in nodes:
            errs.append(f"node {n['id']} depends on missing {d!r}")
cnt = len(nodes)
if not nmin <= cnt <= nmax:
    errs.append(f"{cnt} nodes, expected {nmin}..{nmax}")
for part in filter(None, spec.split(",")):
    tools, need = part.split(">=")
    have = sum(1 for n in nodes.values() if n.get("tool") in tools.split("|"))
    if have < int(need):
        errs.append(f"tool mix {part}: found {have}")
tops = [n for n in nodes.values()
        if n["id"] != "root" and n.get("parent") in ("root", "")
        and n.get("tool") != "llm"]   # sink nodes converge by design, not branches
if len(tops) < bmin:
    errs.append(f"{len(tops)} top-level branches, expected >= {bmin}")

def ancestors(nid):          # transitive deps = execution-order upstream
    out, stack = set(), list(nodes[nid].get("deps", []))
    while stack:
        cur = stack.pop()
        if cur in nodes and cur not in out:
            out.add(cur)
            stack.extend(nodes[cur].get("deps", []))
    return out

if par:
    free = [t for t in tops
            if not any(t["id"] in ancestors(o["id"]) or o["id"] in ancestors(t["id"])
                       for o in tops if o is not t)]
    if len(free) < bmin:
        errs.append(f"only {len(free)} mutually independent top branches (want parallel)")

def depth(n):
    d, cur, seen = 0, n, set()
    while cur.get("parent") and cur["parent"] in nodes and cur["parent"] not in seen:
        seen.add(cur["parent"])
        cur = nodes[cur["parent"]]
        d += 1
    return d

dmax = max((depth(n) for n in nodes.values()), default=0)
if dmax < dmin:
    errs.append(f"depth {dmax}, expected >= {dmin}")
mix = {}
for n in nodes.values():
    if n.get("tool"):
        mix[n["tool"]] = mix.get(n["tool"], 0) + 1
print(f"  plan: {cnt} nodes, depth {dmax}, top branches {len(tops)}, tools {mix}")
if errs:
    print("  plan problems: " + "; ".join(errs))
    sys.exit(1)
PY
}

# ---------- end answer: root DONE + part coverage ("a|b" = any alternative) ----------
answer_check() {  # answer_check <"part|alt,..." ...>
  python3 /dev/fd/3 "$@" 3<<'PY'
import json, sys
run = json.load(sys.stdin)
root = run["nodes"].get("root", {})
res = root.get("result")
text = res.get("text", "") if isinstance(res, dict) else (res or "")
norm = text.lower().replace(",", "")
errs = []
if root.get("state") != "DONE":
    errs.append(f"root state is {root.get('state')}, not DONE")
for spec in sys.argv[1:]:
    if not any(a.lower() in norm for a in spec.split("|")):
        errs.append(f"answer missing {spec!r}")
print(f"  answer: {len(text)} chars - {text[:160].replace(chr(10), ' ')}...")
if errs:
    print("  answer problems: " + "; ".join(errs))
    sys.exit(1)
PY
}

# ---------- run lifecycle ----------
create_run() {
  api POST /runs "{\"intent\": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$1")}" \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["run_id"])'
}

seed_file() {  # seed_file <run> <path> <content> — write via /exec: auto-approved, executes at once
  local payload out bytes
  payload=$(python3 -c 'import json,sys;print(json.dumps({"node_id":"root","tool":"write_file",
             "args":{"path":sys.argv[1],"content":sys.argv[2]}}))' "$2" "$3")
  out=$(api POST "/runs/$1/exec" "$payload")
  bytes=$(echo "$out" | python3 -c 'import json,sys
d=json.load(sys.stdin); print((d.get("result") or {}).get("bytes") or "")' 2>/dev/null)
  if [ -n "$bytes" ]; then report 0 "seed $2 ($bytes bytes)"; else report 1 "seed $2: $out"; fi
}

run_shell() {  # run_shell <run> <cmd> -> sandbox result as one JSON object
  local payload
  payload=$(python3 -c 'import json,sys;print(json.dumps({"node_id":"root","tool":"shell",
             "args":{"cmd":sys.argv[1]}}))' "$2")
  api POST "/runs/$1/exec" "$payload" | python3 -c 'import json,sys
d=json.load(sys.stdin); print(json.dumps(d.get("result") or {}))'
}

wait_json() {  # wait_json <run> <python expr over r> <seconds>
  local run="$1" expr="$2" deadline=$(( $(date +%s) + $3 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ "$(api GET "/runs/$run" | python3 -c "import json,sys;r=json.load(sys.stdin);print($expr)")" = "True" ]; then
      return 0
    fi
    sleep "$POLL"
  done
  return 1
}

finish_gate() {  # finish_gate <run> — completion within EXEC_WAIT, else dump the wedge
  if wait_json "$1" 'not r["active"]' "$EXEC_WAIT"; then return 0; fi
  report 1 "run finished in ${EXEC_WAIT}s (still active — unfinished nodes below)"
  api GET "/runs/$1" | python3 /dev/fd/3 3<<'PY'
import json, sys
for k, n in sorted(json.load(sys.stdin)["nodes"].items()):
    if n["state"] != "DONE":
        print(f"  unfinished {k:10} {n['state']:10} tool={str(n.get('tool')):10} hint={n.get('hint','')[:70]}")
PY
  return 1
}

ws_file() { api GET "/workspace/file?path=$1" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("content",""))'; }

# ---------- tests ----------
t1() {  # calculator app: greenfield codegen + verify
  say "T1 create a calculator app in python"
  local run; run=$(create_run "Create a calculator app in python")
  echo "  run_id=$run (watch at $BASE)"
  if ! wait_json "$run" 'r["plan_ready"]' "$PLAN_WAIT"; then report 1 "plan ready in ${PLAN_WAIT}s"; return; fi
  api GET "/runs/$run" | plan_check 2 15 1 1 0 "write_file|shell>=1,llm>=1"; report $? "plan shape (work step + answer sink)"
  api POST "/runs/$run/go" >/dev/null
  finish_gate "$run"    # non-fatal: answer/artifact checks below report their own evidence
  api GET "/runs/$run" | answer_check "calc" "python|run|import|usage|use"; report $? "answer names the calculator and how to run it"
  local hit
  hit=$(run_shell "$run" 'grep -rlE "def (add|subtract|sub|mul|mult|div)" --include=*.py . | grep -v bugfix-lab | head -3' \
        | python3 -c 'import json,sys;print((json.load(sys.stdin).get("stdout") or "").strip())')
  if [ -n "$hit" ]; then report 0 "calculator module discoverable in workspace"; echo "  found: $hit"
  else report 1 "calculator module discoverable in workspace"; fi
}

t2() {  # debug: two seeded bugs, must end with green tests
  say "T2 fix the seeded bugs in bugfix-lab"
  local run; run=$(create_run "The tests in bugfix-lab are failing. Find and fix the bug(s) in bugfix-lab/analyze.py so the whole test suite passes.")
  echo "  run_id=$run (watch at $BASE)"
  seed_file "$run" "bugfix-lab/analyze.py" 'def mean(data):
    return sum(data) / (len(data) + 1)  # BUG: off-by-one denominator


def median(data):
    xs = data  # BUG: must sort before picking the middle
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2
'
  seed_file "$run" "bugfix-lab/test_analyze.py" 'from analyze import mean, median


def test_mean():
    assert mean([2, 4, 6]) == 4


def test_median_odd():
    assert median([7, 1, 3]) == 3


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5
'
  if [ "$T_FAIL" = 1 ]; then return; fi
  if ! wait_json "$run" 'r["plan_ready"]' "$PLAN_WAIT"; then report 1 "plan ready in ${PLAN_WAIT}s"; return; fi
  api GET "/runs/$run" | plan_check 2 15 1 1 0 "write_file|shell|read_file>=2,llm>=1"; report $? "plan shape (inspect -> fix -> answer sink)"
  api POST "/runs/$run/go" >/dev/null
  finish_gate "$run"    # non-fatal: answer/artifact checks below report their own evidence
  api GET "/runs/$run" | answer_check "bug|fix|wrong|broken" "mean|median|analyz"; report $? "answer explains the fix"
  local rc
  rc=$(run_shell "$run" 'cd bugfix-lab && (python3 -m pytest -q || python -m pytest -q)' \
       | python3 -c 'import json,sys;print(json.load(sys.stdin).get("rc", -9))')
  [ "$rc" = "0" ]; report $? "pytest green when WE rerun it (rc=$rc)"
  # the marker is the BUGGY STATEMENT, not a fragment: a correct fix may quote
  # the old expression in a comment ('was sum(data) / (len(data) + 1)')
  case "$(ws_file bugfix-lab/analyze.py)" in
    *"sum(data) / (len(data) + 1)"*) report 1 "bug marker gone from analyze.py" ;;
    *) report 0 "bug marker gone from analyze.py" ;;
  esac
}

t3() {  # data processing + written synthesis on seeded CSV
  say "T3 analyze data-lab/sales.csv, write data-lab/report.md"
  local run; run=$(create_run "Analyze the sales data in data-lab/sales.csv and write a summary report to data-lab/report.md with total revenue and the top product.")
  echo "  run_id=$run (watch at $BASE)"
  seed_file "$run" "data-lab/sales.csv" 'date,region,product,units,unit_price
2024-01-05,North,Widget,12,19.50
2024-01-12,South,Gadget,7,45.00
2024-01-19,North,Widget,15,19.50
2024-02-02,East,Gizmo,20,12.25
2024-02-14,South,Widget,9,19.50
2024-02-21,West,Gadget,11,45.00
2024-03-03,North,Gizmo,30,12.25
2024-03-11,East,Widget,6,19.50
2024-03-18,South,Gadget,14,45.00
2024-03-25,West,Gizmo,25,12.25
'
  if [ "$T_FAIL" = 1 ]; then return; fi
  if ! wait_json "$run" 'r["plan_ready"]' "$PLAN_WAIT"; then report 1 "plan ready in ${PLAN_WAIT}s"; return; fi
  api GET "/runs/$run" | plan_check 3 15 2 1 0 "write_file|shell|read_file>=2,llm>=1"; report $? "plan shape (gather -> report sink)"
  api POST "/runs/$run/go" >/dev/null
  finish_gate "$run"    # non-fatal: answer/artifact checks below report their own evidence
  api GET "/runs/$run" | answer_check "report|summary"; report $? "answer points at the report"
  local rep; rep=$(ws_file data-lab/report.md | tr -d ',')
  if [ -n "$rep" ]; then report 0 "data-lab/report.md exists"; else report 1 "data-lab/report.md exists"; fi
  case "$rep" in *3177*) report 0 "report has correct total revenue (3177.75)" ;; *) report 1 "report total revenue 3177.75" ;; esac
  case "${rep,,}" in *gadget*) report 0 "report names top product (Gadget)" ;; *) report 1 "report top product Gadget" ;; esac
}

t4() {  # decomposition-focused: parallel branches must appear in the plan
  say "T4 quicksort vs mergesort (parallel decomposition)"
  local run; run=$(create_run "Compare quicksort and mergesort: implement both in Python, time them sorting 1000 random numbers, and report which is faster.")
  echo "  run_id=$run (watch at $BASE)"
  if ! wait_json "$run" 'r["plan_ready"]' "$PLAN_WAIT"; then report 1 "plan ready in ${PLAN_WAIT}s"; return; fi
  api GET "/runs/$run" | plan_check 3 25 2 1 1 "write_file|shell>=2,llm>=1"; report $? "plan has >=3 nodes, >=2 parallel branches, and a sink"
  api POST "/runs/$run/go" >/dev/null
  finish_gate "$run"    # non-fatal: answer/artifact checks below report their own evidence
  api GET "/runs/$run" | answer_check "quicksort|quick sort" "mergesort|merge sort" "fast|speed|faster"; report $? "answer compares both and gives a verdict"
}

t5() {  # answer-focused: invented facts, all parts must land in the head answer
  say "T5 invented-facts QA (grounded final answer)"
  local run; run=$(create_run "Read data-lab/facts.txt and answer three things: in what year was Project Nettlebrook signed off, who directed Project Cindermoor, and which project closed the series?")
  echo "  run_id=$run (watch at $BASE)"
  seed_file "$run" "data-lab/facts.txt" 'Project Nettlebrook signed off: 2031
Project Nettlebrook director: Halden Voss
Project Cindermoor signed off: 2033
Project Cindermoor director: Petra Lund
Project Wrenfield signed off: 2036
Project Wrenfield director: Amos Kell
'
  if [ "$T_FAIL" = 1 ]; then return; fi
  if ! wait_json "$run" 'r["plan_ready"]' "$PLAN_WAIT"; then report 1 "plan ready in ${PLAN_WAIT}s"; return; fi
  api GET "/runs/$run" | plan_check 2 15 1 1 0 "read_file|shell>=1,llm>=1"; report $? "plan reads the facts file and sinks to an answer"
  api POST "/runs/$run/go" >/dev/null
  finish_gate "$run"    # non-fatal: answer/artifact checks below report their own evidence
  api GET "/runs/$run" | answer_check "2031" "lund" "wrenfield"; report $? "answer carries all 3 facts (part coverage)"
}

# ---------- selfcheck: the checkers themselves, no stack required ----------
selfcheck() {
  say "selfcheck (canned run JSON through plan_check / answer_check)"
  local ok=1
  echo '{"nodes":{"root":{"id":"root","parent":"","deps":[],"state":"DONE","tool":null,"result":{"text":"Calc app in python, run it with python calc.py"}},
        "a":{"id":"a","parent":"root","deps":[],"tool":"write_file","state":"DONE"},
        "b":{"id":"b","parent":"root","deps":[],"tool":"shell","state":"DONE"},
        "c":{"id":"c","parent":"a","deps":["a"],"tool":"shell","state":"DONE"},
        "d":{"id":"d","parent":"root","deps":["a","b","c"],"tool":"llm","state":"DONE"}}}' \
    | plan_check 2 15 2 2 1 "write_file|shell>=2,llm>=1" || ok=0
  echo '{"nodes":{"root":{"id":"root","parent":"","deps":[],"state":"DONE","tool":null,"result":{"text":"..."}},"a":{"id":"a","parent":"root","deps":["ghost"],"tool":"shell","state":"DONE"}}}' \
    | plan_check 2 15 1 1 0 "" && ok=0      # must FAIL: missing dep
  echo '{"nodes":{"root":{"id":"root","parent":"","deps":[],"state":"DONE","tool":null,"result":{"text":"..."}},"a":{"id":"a","parent":"root","deps":[],"tool":"shell","state":"DONE"}}}' \
    | plan_check 2 15 1 1 0 "llm>=1" && ok=0     # must FAIL: no synthesis sink
  echo '{"nodes":{"root":{"id":"root","parent":"","deps":[],"state":"DONE","tool":null,"result":{"text":"Fixed the mean bug and the median bug in analyze.py"}}}}' \
    | answer_check "bug|fix" "mean|median" || ok=0
  echo '{"nodes":{"root":{"id":"root","parent":"","deps":[],"state":"DONE","tool":null,"result":{"text":"all good"}}}}' \
    | answer_check "2031" && ok=0           # must FAIL: missing fact
  if [ "$ok" = 1 ]; then echo "selfcheck OK"; else echo "selfcheck FAILED"; return 1; fi
}

# ---------- main ----------
case "${1:-all}" in
  --selfcheck) selfcheck; exit $? ;;
esac
curl -sf "$BASE/health" >/dev/null || { echo "core not reachable at $BASE"; exit 1; }
AGENTS=$(curl -s "$BASE/agents" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))')
[ "$AGENTS" -ge 1 ] || { echo "no agents registered — run: docker compose up -d --scale agent=2"; exit 1; }

OLD_POLICY=$(api GET /policy | python3 -c 'import json,sys;print(json.dumps({"auto_approve": json.load(sys.stdin)["auto_approve"]}))')
api POST /policy '{"auto_approve": ["shell","write_file","read_file","list_dir","llm","web_search","http_fetch"]}' >/dev/null
trap 'api POST /policy "$OLD_POLICY" >/dev/null' EXIT    # restore the confirm-the-mutations policy

FAILED=0
CASE="${1:-all}"
case "$CASE" in
  all) TESTS="t1 t2 t3 t4 t5" ;;
  t1|t2|t3|t4|t5) TESTS="$CASE" ;;
  *) echo "unknown test '$CASE' (want t1..t5, all, or --selfcheck)"; exit 2 ;;
esac
for t in $TESTS; do
  T_FAIL=0
  $t
  if [ "$T_FAIL" = 0 ]; then echo "--- $t PASS"; else echo "--- $t FAIL"; FAILED=1; fi
done
say "done (policy restored)"
exit "$FAILED"
