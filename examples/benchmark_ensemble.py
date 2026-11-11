#!/usr/bin/env python3
"""Measured cost/benefit of the ensemble (sample → debate → vote).

Runs a small checkable suite (arithmetic word problems — the answer is a number,
so "correct" is deterministic, not a vibe) through three coordination configs:

  baseline   samples=1   debate=0   -> one call, today's default
  sampled    samples=K   debate=0   -> K independent drafts + judge vote
  debated    samples=K   debate=R   -> K drafts, R rival-aware revisions, vote

and prints a markdown table of **accuracy / LLM calls / wall seconds** per task,
plus a check that measured calls match the cost model:

  calls = samples * (1 + debate_rounds) + (1 if samples > 1 else 0)   [+0 when n=1]

Usage:
  python examples/benchmark_ensemble.py                       # llama3.1:8b @ localhost
  OLLAMA_URL=http://gpu-box:11434 MODEL=qwen2.5:3b \
      python examples/benchmark_ensemble.py --samples 3 --rounds 1
  python examples/benchmark_ensemble.py --limit 2             # smoke run
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.ensemble import Ensemble  # noqa: E402
from harness.llm import LLM  # noqa: E402

# (question, expected number). The last three are classic rate traps — the kind
# of confident-wrong answer sampling + debate exists to catch.
SUITE: list[tuple[str, str]] = [
    ("A bakery sells muffins for $3 each. It sold 14 on Monday and twice as many "
     "on Tuesday. What was the total revenue in dollars?", "126"),
    ("Tom has 4 shelves with 17 books each. He donates 23 books. "
     "How many books remain?", "45"),
    ("A train travels 3 hours at 60 km/h, then 2 hours at 45 km/h. "
     "What total distance in km does it cover?", "270"),
    ("A rectangle is 12 cm by 7 cm. What is its perimeter in cm?", "38"),
    ("Twice a number, increased by 9, equals 25. What is the number?", "8"),
    ("If 5 workers build 5 walls in 5 days, how many days does it take "
     "1 worker to build 1 wall?", "5"),
    ("If 8 painters paint 8 houses in 8 hours, how many houses can "
     "4 painters paint in 4 hours?", "2"),
    ("A pond's lily pad doubles in area every day and covers the pond on "
     "day 30. On which day did it cover half the pond?", "29"),
]

SYS = ("You are a careful solver. Think step by step, then end your reply with "
       "a final line of exactly: Answer: <number>")
_ANS_RE = re.compile(r"Answer:\s*\$?(-?[\d.,]+)", re.I)


def extract(text: str) -> str:
    """The declared 'Answer: <n>' line, else the last number in the text."""
    m = _ANS_RE.findall(text or "")
    if m:
        return m[-1].rstrip(".,")
    nums = re.findall(r"-?\d[\d.,]*", text or "")
    return nums[-1].rstrip(".,") if nums else ""


def grade(text: str, expected: str) -> bool:
    """Numeric compare with tolerance for '6.0' vs '6' formatting wobble."""
    got = extract(text)
    try:
        return abs(float(got.replace(",", "")) - float(expected)) < 1e-6
    except ValueError:
        return got == expected


def calls_model(n: int, rounds: int) -> int:
    """Exact LLM-call cost of one ensemble run — the trade you are buying."""
    return n * (1 + rounds) + (1 if n > 1 else 0)


def run_suite(make_llm, tasks: list[tuple[str, str]], n: int, rounds: int,
              temperature: float) -> list[dict]:
    """One config over the suite. `make_llm()` hands out a fresh metered LLM
    per task so calls/seconds are attributable per task. A task whose LLM call
    dies (timeout, runner down) is recorded as a failed data point — a
    measurement run must never lose a full config to one bad call."""
    rows = []
    for q, expected in tasks:
        llm = make_llm()
        t0 = time.time()
        row: dict = {"expected": expected, "q": q[:44]}
        try:
            answer = Ensemble(llm, SYS, model=getattr(llm, "model", ""),
                              temperature=temperature).run(q, n, rounds) or ""
            row.update(ok=grade(answer, expected), got=extract(answer))
        except Exception as e:
            row.update(ok=False, got="", error=f"{type(e).__name__}")
        row.update(calls=llm.calls, seconds=round(time.time() - t0, 1))
        rows.append(row)
    return rows


def report(results: dict[str, list[dict]], n: int, rounds: int) -> str:
    """Markdown table + cost-model check for all configs."""
    lines = ["| config | accuracy | LLM calls/task | wall s/task |",
             "|---|---|---|---|"]
    for name, rows in results.items():
        acc = f"{sum(r['ok'] for r in rows)}/{len(rows)}"
        calls = sum(r["calls"] for r in rows) / len(rows)
        secs = sum(r["seconds"] for r in rows) / len(rows)
        lines.append(f"| {name} | {acc} | {calls:.1f} | {secs:.1f} |")
    cfg = {"baseline": (1, 0), "sampled": (n, 0), "debated": (n, rounds)}
    checks = []
    for name, rows in results.items():
        key = next((k for k in cfg if name == k or name.startswith(k)), None)
        if key is None:
            continue  # custom-named configs get no cost-model check
        model = calls_model(*cfg[key])
        checks.append(all(r["calls"] == model for r in rows))
    ok = "matches" if all(checks) else "DIFFERS FROM"
    lines.append(f"\nMeasured calls {ok} the cost model "
                 f"`calls = samples × (1 + debate_rounds) + vote`.")
    errs = sum(1 for rows in results.values() for r in rows if r.get("error"))
    if errs:
        lines.append(f"{errs} task(s) errored (counted as wrong) — see per-task detail.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "llama3.1:8b"))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N tasks")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-call LLM timeout (benchmarks must outlive slow calls)")
    ap.add_argument("--out", default="bench_raw.json",
                    help="raw results snapshot, rewritten after every config ('' disables)")
    args = ap.parse_args(argv)

    tasks = SUITE[args.skip:]
    if args.limit:
        tasks = tasks[: args.limit]
    make_llm = lambda: LLM(args.url, args.model, timeout=args.timeout)  # noqa: E731
    grid = {"baseline": (1, 0),
            f"sampled (samples={args.samples})": (args.samples, 0),
            f"debated (samples={args.samples}, rounds={args.rounds})": (args.samples, args.rounds)}
    results: dict[str, list[dict]] = {}
    for name, (n, rounds) in grid.items():
        results[name] = run_suite(make_llm, tasks, n, rounds, args.temperature)
        if args.out:  # checkpoint: a later crash must not take earlier configs with it
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1)
    print(f"\nModel: {args.model} @ {args.url} · temperature {args.temperature} "
          f"· {len(tasks)} tasks\n")
    print(report(results, args.samples, args.rounds))
    print("\nPer-task detail:")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
