# MCP Registry PR Draft: Sounio Compiler Server

## Title

Add Sounio compiler MCP server to community servers

## Summary

This PR adds `sounio-mcp-server`, a local stdio Model Context Protocol server
for the Sounio compiler. It exposes compiler check, compile, run, and test
operations to MCP-capable coding agents, plus resources for standard-library
documentation and compiler error explanations.

## Registry Entry

```yaml
name: sounio-compiler
display_name: Sounio Compiler
description: Local stdio MCP server for checking, compiling, running, and testing Sounio .sio programs.
repository: https://github.com/Sounio-lang/sounio
homepage: https://github.com/Sounio-lang/sounio/tree/main/tools/mcp
language: python
license: Apache-2.0
transport:
  - stdio
tools:
  - sounio_check
  - sounio_compile
  - sounio_run
  - sounio_test
resources:
  - sounio://stdlib/{module}
  - sounio://errors/{error_code}
```

## Install

```bash
pip install -e tools/mcp
python -m sounio_mcp.server --transport stdio
```

Claude Code:

```bash
claude --mcp-server sounio=python:-m:sounio_mcp.server
```

## Security

- Local stdio transport only.
- No hosted transport in v1.
- No OAuth, token intake, or credential storage.
- Tool paths are constrained to the Sounio workspace.
- Sensitive environment keys are rejected for `sounio_run`.

## Verification

```bash
ruff check tools/mcp
mypy --strict tools/mcp/sounio_mcp
pytest tools/mcp/tests
```
