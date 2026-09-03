# AI Disclosure

This artefact follows the repository's GAIDeT-ICMJE 2025 disclosure convention.

## Artefact

- Name: `sounio-mcp-server`
- Location: `tools/mcp/`
- Date: 2026-05-17
- Sprint: 30-Day Brilho Sprint, CC-2 front

## AI Systems Used

- OpenAI Codex, GPT-5-class coding agent, used for implementation, tests, and
  documentation drafting inside `/workspace/sounio`.
- Context7 CLI, used to retrieve current documentation for
  `/modelcontextprotocol/python-sdk` before using the FastMCP API.

## Human Oversight

- Operator: Demetrios Agourakis.
- The operator supplied the CC-2 implementation brief and the master
  orchestration context.

## Scope of AI Contribution

- Added a Python MCP server scaffold using the MCP Python SDK FastMCP decorator
  API.
- Added compiler wrapper tools for check, compile, run, and tests.
- Added stdlib and compiler-error MCP resources.
- Added the deterministic Error -> Fix loop recipe and benchmark fixtures.
- Added Cursor and Claude Code integration documentation.

## Verification

The artefact is intended to be verified by:

```bash
pip install -e tools/mcp[dev]
ruff check tools/mcp
mypy --strict tools/mcp/sounio_mcp
pytest tools/mcp/tests
PYTHONPATH=tools/mcp python3 tools/mcp/examples/llmloop_recipe.py --fixtures 'tests/fixtures/broken/*.sio' --max-iter 10 --seed 1729
```

## Limitations

- The local benchmark uses a deterministic fixture agent for reproducibility; it
  is not a hosted-model performance claim.
- The current checked self-hosted launcher is strongest for host-native ELF.
  Cross-target requests are passed through and reported honestly.
