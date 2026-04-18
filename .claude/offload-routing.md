# LLM offload routing (Sounio)

Canonical workspace path: **`/workspace/sounio`**. Prefer repo scripts and `llm-offload` / `llm-pipeline` over ad-hoc API calls.

## Task → tool (default)

| Task type | Tool | Notes |
|-----------|------|--------|
| Outline → prose | `llm-offload -t expand -p grok` | Fast expansion |
| Spec → boilerplate | `llm-offload -t scaffold -p glm` | GLM-5 |
| Second opinion / review | `llm-offload -t review -p deepseek` | Code review |
| Rewrite / paraphrase | `llm-offload -t paraphrase -p minimax` | MiniMax (Anthropic-compatible API) |
| Provider status | `llm-offload --list-providers` | |
| Multi-model review | `llm-pipeline consensus review -i <file>` | |
| Expand + critique | `llm-pipeline expand-critique <outline.md>` | Grok → DeepSeek |
| Multi scaffold | `llm-pipeline multi-scaffold <spec.txt>` | |

## MiniMax (API-compatible)

Set when using MiniMax through Anthropic-style clients:

`ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic`

Models: M2.7 (204K), M2.5, M2.1, M2 — tools, streaming, thinking. See root `CLAUDE.md` for slash-command naming inside Claude Code.

## Google Gemini / Ultra fleet

Tables, quotas, and script examples live in **`scripts/gemini-integration-readme.md`**. Update that file when adding new Gemini entrypoints; keep this table high-level only.

## MCP Context7 (remote workspace)

Claude Code global config lives in **`~/.claude/settings.json`**. If `context7` fails to start (e.g. `npx` not found), the `command` path is wrong for this host.

**Fix:** Point `mcpServers.context7.command` to a real `npx` on this machine, for example:

- Run `command -v npx` in the workspace shell and paste that absolute path, or  
- Use a wrapper such as `/usr/bin/env` with `PATH` including Node, or `bash -lc 'exec npx -y @upstash/context7-mcp@latest'` if your shell profile loads `nvm`/`fnm`.

Do not commit `~/.claude/settings.json` into the repo; keep machine-specific paths there only.

## Memory lanes (Claude Code project memory)

For which `~/.claude/projects/-workspace-sounio/memory/*.md` files to load per task type, see **`.claude/MEMORY_LANES.md`** in this repo.
