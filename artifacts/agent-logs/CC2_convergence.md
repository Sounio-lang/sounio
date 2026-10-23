# CC-2 Convergence Log

Sprint: 30-Day Brilho Sprint
Front: CC-2, Sounio MCP Server + Error -> Fix Loop
Branch: `feature/mcp-server`
Date: 2026-05-17

## Cycle 1 — Compiler Surface Discovery

Production:
- Added the initial FastMCP package scaffold and compiler wrappers.

Critique:
- `bin/souc compile` compatibility mode can return process exit zero even when
  the underlying raw compile fails.
- The MCP compile tool therefore cannot trust return code alone.

Revision:
- `sounio_compile` now runs `sounio_check` first and validates that an output
  artefact exists and has non-zero size.

## Cycle 2 — Error -> Fix Benchmark

Production:
- Added twenty deliberately broken `.sio` fixtures and a deterministic loop
  recipe.

Critique:
- Five fixtures accidentally passed under the current compiler, which would have
  inflated convergence without exercising diagnostics.

Revision:
- Reworked those fixtures to use explicit type mismatches. The benchmark now
  observes diagnostics on all twenty fixtures.

## Cycle 3 — Agent Safety Boundary

Production:
- Added `sounio_run` environment passthrough and path resolution.

Critique:
- Agent-facing run tools must not pass secrets or operate outside the workspace.

Revision:
- Source paths are constrained to `/workspace/sounio`.
- Sensitive environment keys, auth tokens, credentials, SSH material, and proxy
  variables are stripped or rejected.

## Cycle 4 — Protocol And Schema Hardening

Production:
- Added the minimal stdio JSON-RPC dispatcher and focused direct-tool tests.

Critique:
- A shell one-shot smoke is weaker evidence than a real MCP client session, and
  CC-1 now provides `tools/shared/diagnostic_schema.json`.

Revision:
- Tool calls now return both JSON text content and `structuredContent`.
- `sounio_check` now returns a `sounio.diagnostic.v1` envelope aligned with the
  shared CC-1/CC-2 schema.
- The focused test suite now includes JSON Schema validation and an MCP Python
  SDK `ClientSession` over stdio.

## Benchmark Evidence

Command:

```bash
PYTHONPATH=tools/mcp python3 tools/mcp/examples/llmloop_recipe.py --fixtures 'tests/fixtures/broken/*.sio' --max-iter 10 --seed 1729 --json
```

Observed result:

```json
{
  "agent_model": "deterministic-fixture-agent-v0.1",
  "average_iterations": 1.0,
  "converged": 20,
  "convergence_rate": 1.0,
  "fixtures": 20,
  "max_iterations": 10,
  "seed": 1729
}
```

Protocol evidence:

```text
PYTHONPATH=/tmp/sounio-mcp-pydeps:tools/mcp python3 -m pytest tools/mcp/tests
12 passed
```
