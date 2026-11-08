# Examples

| file | what it proves |
|---|---|
| `write-paper.json` | template DAG: `web_search` leaves → `write_file` sections → `shell` assemble (the original research flavor) |
| `add-tests.json` | template DAG TDD loop: seed bug → red test → fix → green, zero LLM calls |
| `codebase-inventory.json` | DAG works offline, any codebase: parallel `shell` gathers → `write_file` outline → one `llm` summary sink |
| `acceptance-demo.sh` | end-to-end unattended run of `add-tests.json` (template + auto_approve policy) |
| `run-with-overrides.sh` | intent-only run with per-run `prompts`/`params` overrides (`PUT /runs/{id}/config`) |
| `doc_writer.py` | dynamic Python tool: drop in `/workspace/tools/` + `POST /tools/sync` → claimable |
| `mcp_hello.py` | minimal no-SDK MCP stdio server: register in `/workspace/tools/mcp.json` |

## Run them

- template JSONs: `curl -s -X POST http://localhost:8100/runs -H 'content-type: application/json' -d @examples/codebase-inventory.json`
- scripts: `./examples/acceptance-demo.sh` / `./examples/run-with-overrides.sh` (core + sandbox up; the overrides script also needs one agent)
- UI: open http://localhost:8100 → template-DAG tab → paste the JSON

The harness is domain-generic — the planner uses whatever tools/prompts you give it; these examples intentionally avoid web tools.
If the same DAG shape inventories a codebase, seeds-and-fixes a bug, and writes a paper, it will work for any domain you point it at.
