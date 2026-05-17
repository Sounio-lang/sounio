# MCP Registry Publication Draft: Sounio Compiler Server

## Current Submission Path

The original CC-2 prompt asked for a PR to `modelcontextprotocol/servers`.
That repository no longer accepts community-server additions to its README. Its
current PR template and contributing guide direct publishers to the separate MCP
Registry at `modelcontextprotocol/registry`.

Use `tools/mcp/registry_server.draft.json` as the draft `server.json` for
registry publication after `sounio-mcp-server` is published as a PyPI package.
The package README already contains the PyPI ownership marker:

```markdown
<!-- mcp-name: io.github.sounio-lang/sounio-mcp-server -->
```

## Summary

`sounio-mcp-server` is a local stdio Model Context Protocol server for the
Sounio compiler. It exposes compiler check, compile, run, and test operations to
MCP-capable coding agents, plus resources for standard-library documentation and
compiler error explanations.

## Draft `server.json`

```bash
python3 -m jsonschema \
  /tmp/modelcontextprotocol-registry/internal/validators/schemas/2025-12-11.json \
  -i tools/mcp/registry_server.draft.json
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
python3 -m jsonschema /path/to/registry/internal/validators/schemas/2025-12-11.json -i tools/mcp/registry_server.draft.json
```

## Operator Publish Steps

```bash
# after publishing sounio-mcp-server 0.1.0 to PyPI
mcp-publisher login github
mcp-publisher publish tools/mcp/registry_server.draft.json
```
