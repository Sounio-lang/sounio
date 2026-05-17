# CC-2 Prompt-To-Artefact Checklist

Objective: make the CC-2 MCP server top-tier for local agentic Sounio compiler
loops over stdio.

## Concrete Success Criteria

| Requirement | Evidence | Status |
|---|---|---|
| Four tools: `sounio_compile`, `sounio_check`, `sounio_run`, `sounio_test` | `tools/mcp/sounio_mcp/server.py`, `tools/mcp/tests/test_protocol.py` SDK `list_tools` check | PASS |
| Stdio-only MCP server | `server.py` argparse accepts only `--transport stdio`; no HTTP/SSE path is exposed | PASS |
| FastMCP decorator API used | `server.py` defines `mcp = FastMCP(...)` and registers tools/resources with decorators | PASS |
| Real-client compatibility | `tools/mcp/tests/test_protocol.py` starts the server through `mcp.client.stdio` and `ClientSession` | PASS |
| Tool calls return typed data | `server.py` returns both JSON text content and `structuredContent` | PASS |
| `sounio_check` structured diagnostics | `check.py`, `diagnostics.py`; pytest validates `diagnostic_envelope` against `tools/shared/diagnostic_schema.json` | PASS |
| Shared CC-1 schema alignment | `diagnostic_envelope.schema == sounio.diagnostic.v1`; schema validation test | PASS |
| `sounio_run` captures stdout/stderr/exit/timeout | `run.py`, `test_sounio_run_hello`, one-shot MCP smoke | PASS |
| `sounio_compile` target/optimisation handling | `compile.py`, `test_sounio_compile_rejects_unknown_target`; compile checks output existence | PASS |
| `sounio_test` per-test status | `test.py`, `test_sounio_test_can_run_filtered_file` | PASS |
| Resource pattern `sounio://stdlib/{module}` | `stdlib_docs.py`, SDK resource-template test, README examples | PASS |
| Resource pattern `sounio://errors/{error_code}` | `errors_catalog.py`, SDK resource read test, >=20 trace-backed entries | PASS |
| Error catalogue does not invent codes | Unknown codes return "not promoted yet"; invalid codes return invalid-code doc | PASS |
| Error -> Fix recipe | `tools/mcp/examples/llmloop_recipe.py`, 20 fixtures, pinned model and seed | PASS |
| Benchmark >=80% convergence on 20 fixtures | `agent_logs/CC2_convergence.md`; latest focused pytest covers `>=0.8`, observed 20/20 | PASS |
| Claude Code usage docs and subagent recipe | `tools/mcp/examples/claude_code_usage.md` | PASS |
| Cursor rules file | `.cursor/rules/sounio.mdc` with glob, patterns, anti-patterns, planned docs link | PASS |
| Root `CLAUDE.md` update | `CLAUDE.md` extended rather than replaced | PASS |
| MCP Registry submission artefact | `tools/mcp/MCP_REGISTRY_PR_DRAFT.md` | PARTIAL: operator still needs to open external PR |
| Q1 Python clean compile | `ruff check`, `mypy --strict` pass | PASS |
| Q2 iterative convergence | `agent_logs/CC2_convergence.md` records 4 cycles | PASS |
| Q3 adversarial self-critique | `tools/mcp/ADVERSARIAL_SELF_CRITIQUE.md` | PASS |
| Q4 regression | `bash scripts/run_sio_test_suite.sh hello` passed 2/2 | PARTIAL: full `tests/run-pass` not run |
| Q5 Sounio purity | `bin/souc check examples/hello.sio` and fixture benchmark | PASS |
| Q6 public API docs/examples | Tool/resource docstrings, README, usage docs | PASS |
| Q7 provenance | Loop model `deterministic-fixture-agent-v0.1`, seed `1729` | PASS |
| Q9 AI disclosure | `tools/mcp/AI_DISCLOSURE.md`; log row in `.claude/llm_offload_log.md` | PASS |
| No compiler binary modification | `git diff --quiet -- bin/souc-linux-x86_64` | PASS |
| Stay on `feature/mcp-server` | `git branch --show-current` observed `sounio-pure/r2-1-park-miller` | FAIL until rebranched |

## Required Before Commit

- Move or rebase these MCP artefacts onto the intended `feature/mcp-server`
  branch without carrying unrelated native-codegen edits.
- Run the full canonical regression gate if W4 demands strict Q4.
- Submit the MCP Registry PR from `MCP_REGISTRY_PR_DRAFT.md` or record operator
  handoff acceptance.
