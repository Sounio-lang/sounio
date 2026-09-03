# CC-2 Adversarial Self-Critique

Date: 2026-05-17
Branch: `feature/mcp-server`
Target: `tools/mcp/`

## Claims Under Test

| Claim | Probe | Outcome |
|---|---|---|
| Server boots over stdio | `python -m sounio_mcp.server --transport stdio < tests/fixtures/mcp_init.json` | PASS; returns `serverInfo.name=sounio-compiler` |
| `sounio_check` returns structured diagnostics | one-shot `tools/call` on `tests/fixtures/broken_dose.sio` | PASS; returns `valid=false` with non-empty diagnostics |
| `sounio_run` executes a valid program | one-shot `tools/call` on `examples/hello.sio` | PASS; returns stdout `Hello, Sounio\n` |
| Tool result is typed for real clients | Python MCP SDK `ClientSession.call_tool` | PASS; `structuredContent.valid=false` without reparsing text |
| Diagnostics share CC-1 schema | JSON Schema validation of `diagnostic_envelope` | PASS; validates against `tools/shared/diagnostic_schema.json` |
| Resource templates are listed | Python MCP SDK `list_resource_templates` | PASS; exposes `sounio://stdlib/{module}` and `sounio://errors/{error_code}` |
| Malformed JSON-RPC is rejected | input `not-json` | PASS; returns JSON-RPC parse error `-32700` |
| Missing tool is rejected | call `missing_tool` | PASS; returns JSON-RPC error `-32603` |
| Path traversal/outside workspace is rejected | call `sounio_check` on `/etc/passwd` | PASS; returns diagnostic `path escapes the Sounio workspace` |
| Missing/invalid resource is honest | read `sounio://errors/NOPE` | PASS; returns "Invalid error code" rather than inventing a code |
| Oversized path payload does not crash | call `sounio_check` with a 20,000-character path | PASS after revision; returns a tool diagnostic `path is too long` |

## Failure Found And Fixed

The upstream MCP SDK FastMCP stdio helper did not read piped stdin in this
headless sandbox, even for a minimal FastMCP server. The server still uses the
FastMCP decorator API for the public server surface, but the runtime stdio loop
is a small line-delimited JSON-RPC dispatcher for the standard MCP methods used
by Claude Code and the Python MCP client:

- `initialize`
- `tools/list`
- `tools/call`
- `resources/templates/list`
- `resources/read`
- `ping`

This also makes the prompt's one-shot shell acceptance commands terminate
instead of hanging after EOF.

The dispatcher also returns `structuredContent` alongside the JSON text payload,
so MCP SDK clients can consume typed dictionaries directly.

## Remaining Risks

- The custom stdio dispatcher covers the required v1 MCP surface, not optional
  MCP features such as sampling, elicitation, subscriptions, or hosted transport.
- Cross-target compilation beyond host-native ELF is passed through to the
  checked launcher and may fail honestly depending on compiler support.
- Full `tests/run-pass` was not run in this turn; the filtered canonical harness
  smoke `bash scripts/run_sio_test_suite.sh hello` passed 2/2.
- The live checkout observed during the top-tier audit was not on the intended
  `feature/mcp-server` branch, although the MCP artefacts are present in the
  working tree. Rebranch before commit/push.
