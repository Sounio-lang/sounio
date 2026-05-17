# Sounio MCP Server

`sounio-mcp-server` exposes the checked Sounio compiler to agentic tools over
local Model Context Protocol stdio transport. It is intentionally local-only:
no HTTP transport, no OAuth, no token handling, and no remote execution path.

## Install

```bash
pip install -e tools/mcp
python -m sounio_mcp.server --transport stdio
```

Claude Code:

```bash
claude --mcp-server sounio=python:-m:sounio_mcp.server
```

The server resolves `souc` through `scripts/lib/resolve_souc.sh`, which in this
checkout resolves to the checked self-hosted launcher (`bin/souc` on supported
hosts). The server rejects source paths outside `/workspace/sounio`.

Tool calls return both MCP text content and `structuredContent`. The text content
keeps the prompt's shell examples simple; `structuredContent` lets typed MCP
clients consume the result without reparsing JSON text.

## Tools

### `sounio_check`

Type-checks a `.sio` file and returns diagnostics normalised from
`souc check --json`.

Example:

```json
{
  "source_path": "tests/fixtures/broken_dose.sio"
}
```

The response includes two diagnostic views:

- `diagnostics`: agent-friendly records with string severity, one-based
  `line`/`column`, spans, related notes, and warnings.
- `diagnostic_envelope`: the CC-1/CC-2 shared `sounio.diagnostic.v1` envelope
  validated against `tools/shared/diagnostic_schema.json`.

### `sounio_compile`

Runs `sounio_check`, then compiles a valid file to a local artefact.

Example:

```json
{
  "source_path": "examples/hello.sio",
  "target": "elf",
  "optimization": "default"
}
```

Targets are accepted as `elf`, `macho`, `ptx`, `metal`, or `wasm`. The current
checked launcher is strongest for host-native ELF; other targets are passed
through to the compiler target flag and reported honestly if unsupported.

### `sounio_run`

Runs `souc run`, captures stdout/stderr/exit code, and reports `compile_error`,
`runtime_error`, `timeout`, or `ok`.

Example:

```json
{
  "source_path": "examples/hello.sio",
  "args": [],
  "timeout_sec": 30
}
```

Environment overrides are allowed only for non-sensitive keys. Keys containing
tokens, secrets, credentials, proxies, or auth material are rejected.

### `sounio_test`

Runs a Sounio test directory or file programmatically using the checked compiler.

Example:

```json
{
  "test_pattern": "tests/run-pass",
  "test_filter": "hello"
}
```

The test runner understands the existing `//@ run-pass`, `//@ compile-fail`,
`//@ check-only`, `//@ expect-stdout`, `//@ error-pattern`, and `//@ ignore`
annotations.

## Resources

- `sounio://stdlib/{module}` returns stdlib README/source context.
- `sounio://errors/{error_code}` returns compiler error explanations traced to
  current compiler source or committed compile-failure evidence.

Examples:

```text
sounio://stdlib/stats
sounio://stdlib/clinical
sounio://errors/E070
```

## Error -> Fix Loop

The loop is deliberately thin:

1. Agent writes or edits a `.sio` file.
2. Agent calls `sounio_check`.
3. Diagnostics are injected into the next prompt.
4. Agent revises the file.
5. Agent repeats until diagnostics are empty or the iteration budget is reached.

The reproducible local demonstration is:

```bash
PYTHONPATH=tools/mcp python3 tools/mcp/examples/llmloop_recipe.py \
  --fixtures 'tests/fixtures/broken/*.sio' \
  --max-iter 10 \
  --seed 1729
```

Current fixture benchmark:

- fixtures: 20
- agent model: `deterministic-fixture-agent-v0.1`
- seed: `1729`
- convergence rate: `1.0`
- average iterations: `1.0`

This is a local recipe for the LLMloop pattern, not a claim about a hosted model.
It exists so Claude Code, Cursor, ChatGPT via MCP, and CLI agents can close the
same check/fix/re-check loop using the real Sounio compiler.

## Cross-Agent Use

- CC-1 owns `tools/shared/diagnostic_schema.json`; MCP consumes the same
  `souc check --json` shape and keeps the LSP-compatible `range` fields.
- Cx-1 should use `sounio_check` as the canonical validation method for generated
  dataset examples.
- Cx-2 can use `sounio_run` as the HumanEval/MultiPL-E execution harness once
  translated `.sio` files are produced.

## Quality Notes

- No compiler binary is modified.
- No remote MCP transport is implemented.
- Error catalogue entries are trace-backed; unknown codes return an explicit
  "not promoted yet" resource instead of invented explanations.
- The checked launcher's `compile` compatibility mode may exit zero even when an
  artefact is not emitted, so `sounio_compile` verifies output existence.
- The focused Python gate covers direct tools, schema validation, one-shot
  JSON-RPC, and an actual `mcp` Python SDK client session over stdio.
