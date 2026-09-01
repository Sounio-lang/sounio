# LLM offload routing (Sounio)

Canonical workspace path: **`/workspace/sounio`**.

Two layers:

1. **Task-aware wrapper** — `bin/llm-offload -t <task> -p <provider>` (recommended for routine offload).
2. **Raw fan-out** — `bin/llm-offload --raw <prompt-file> [providers...]` or `scripts/mcp/llm-offload.sh` directly (for multi-model consensus or custom prompts).

Run `bin/llm-offload --status` to see which API keys are loaded from `~/.sounio-keys.env`.

## Task → tool (default routing)

The wrapper prepends a task-specific system prompt from `.claude/offload-tasks/<task>.md` before sending to the chosen provider.

| Task | Default provider | Override examples | Use case |
|------|------------------|-------------------|---------|
| `expand` | `gemini` (1M ctx) | `qwen`, `cohere` | Outline → publication-quality prose |
| `scaffold` | `deepseek` | `qwen`, `grok-code` | Spec → boilerplate (.sio, SQL, ETL, LaTeX) |
| `review` | `deepseek` | `xai`, `qwen` | Devil's-advocate code/paper review |
| `paraphrase` | `minimax` (needs key) | `qwen`, `deepseek` | Cover letters, abstract polishing |
| `math-review` | `xai zai local` fan-out | `qwen`, `mistral` | Math / algebra / stats audit — **three-way, so one outage still leaves two independent opinions** |

> **Default math review:** `bin/llm-offload -t math-review` fans out to **xai (grok-4.6, fixed)**, **zai (Z.AI GLM-5.2)**, and the configured local model. A complete review requires a successful Grok 4.6 leg plus a substantive verdict from at least one distinct provider/model family. Provider skips and errors must be recorded as incomplete evidence even though the raw fan-out driver reports them in output rather than enforcing quorum through its exit status. The default response cap is 8,192 tokens; deep audits must opt in with `OFFLOAD_MAX_TOKENS`. Run `bin/llm-offload --status` to see loaded keys.
>
> **Fixed xAI identity (2026-09-01):** `xai`, `grok`, `xai-fast`, and `grok-code` all resolve to the literal model `grok-4.6`. The latter two names are compatibility aliases, not lower-version or code-model fallbacks. Aliases are canonicalized before fan-out and a request containing duplicate aliases fails closed, so they can never count as independent review legs. The public CLI accepts no environment override for this identity; `scripts/ci/llm_offload_grok_identity_selftest.sh` enforces it offline, and its explicit `--live` mode verifies the configured xAI endpoint when credentials are available.

> **Third leg added 2026-08-24.** The two-provider default degrades to ONE whenever either
> vendor is down, silently — the run still exits 0 and prints a review. On 2026-08-24 zai
> returned `1310 Weekly/Monthly Limit Exhausted` and six math reviews were logged as
> single-provider before anyone noticed the policy standard was not being met. `local` is now
> the third leg: it is the on-prem gateway (`LOCAL_LLM_URL`/`LOCAL_LLM_MODEL`), it needs no
> vendor account, and `.claude/AGENT_OFFLOAD_POLICY.md` M1 already accepts a local leg for this
> purpose. Verified the same day on a real Lean diff: it independently checked the arithmetic and
> the negative control rather than echoing the prompt.
>
> **It is a weaker leg than a second frontier vendor, not a substitute for one.** When zai
> recovers you get three; treat `local` as the floor that keeps the standard reachable, and
> prefer `--raw <prompt> xai zai` style fan-outs for anything referee-bound.

```bash
# Outline -> prose
bin/llm-offload -t expand -i docs/papers/vancomycin_pl_paper_outline.md > /tmp/draft.md

# Devil's advocate review of a Sounio file
bin/llm-offload -t review -p deepseek -i stdlib/clinical/vancomycin_pbpk.sio

# Math audit on a derivation snippet (stdin OK)
echo "Verify dCmin/dVc < 0 for ke*tau in [0.3, 1.5]" | bin/llm-offload -t math-review -p xai

# Paraphrase a cover letter
bin/llm-offload -t paraphrase -p qwen -i docs/papers/cover_letters/popl_cover_letter.md
```

## Vancomycin plan — phase-specific routing

Per `.claude/vancomycin_track.md`. Suggested router for the M0–M6 thrust:

| Phase | Best for routine work | Best for review |
|-------|-----------------------|-----------------|
| M0 design decisions | Opus 4.7 (in-session) | `bin/llm-offload -t review -p xai` |
| M1/M2 substrate | Opus 4.7 / Sonnet 4.6 | `-t math-review -p xai` (proved necessary — caught Vc-monotonicity bug) |
| M3 PK math | `-t math-review -p xai` (primary) | `-p qwen` (second opinion) |
| M4 cohort ETL | `-t scaffold -p deepseek` | `-t review -p deepseek` |
| M4 stats | `-t math-review -p xai` | `-t review -p qwen` |
| M5 prose expansion | `-t expand -p gemini` | `-t review -p deepseek` |
| M5 paper review | `bin/llm-offload --raw <draft> deepseek xai gemini` (consensus) | — |
| M6 cover letters | `-t paraphrase -p qwen` (or minimax if key set) | — |

## Multi-model consensus

For high-stakes deliverables, fan out to multiple providers and compare:

```bash
bin/llm-offload --raw /tmp/section.md deepseek xai gemini qwen
```

Default fan-out (no providers specified) hits 5 diverse models: `deepseek xai gemini qwen mistral`.

## Real providers (driven by `scripts/mcp/llm-offload.sh`)

Run `bin/llm-offload --list-providers` for the canonical list.

| Slug | Model | Key required | Strength |
|------|-------|--------------|---------|
| `deepseek` | DeepSeek Coder | `DEEPSEEK_API_KEY` | Code review, second opinion |
| `xai` / `grok` | Grok 4.6 (fixed) | `XAI_API_KEY` | Math, blunt realist, no flattery |
| `xai-fast` / `grok-code` | Grok 4.6 (compatibility aliases) | `XAI_API_KEY` | Existing command compatibility; no model fallback |
| `groq` | Llama 3.3 70B (Groq) | `GROQ_API_KEY` | Fast inference |
| `gemini` | Gemini 2.5 Pro (OpenRouter) | `OPENROUTER_API_KEY` | 1M ctx, long-context expansion |
| `qwen` | Qwen 3 235B (OpenRouter) | `OPENROUTER_API_KEY` | Math/code, Chinese perspective |
| `mistral` | Mistral Large (OpenRouter) | `OPENROUTER_API_KEY` | Formal methods, European POV |
| `llama` | Llama 4 Maverick (OpenRouter) | `OPENROUTER_API_KEY` | Diverse training |
| `cohere` | Command A (OpenRouter) | `OPENROUTER_API_KEY` | Structured analysis, lit review |
| `openrouter` | Auto-routed | `OPENROUTER_API_KEY` | Fallback |
| `minimax` | MiniMax M2.7 | `MINIMAX_API_KEY` | Long context, paraphrase |

## Status of keys (this workspace)

Run `bin/llm-offload --status` for live state. As of M6 close (2026-04-30):

- `+` DEEPSEEK_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, XAI_API_KEY
- `-` MINIMAX_API_KEY (paraphrase falls back to `qwen`)

## MCP Context7 (remote workspace)

Claude Code global config lives in **`~/.claude/settings.json`**. If `context7` fails to start (e.g. `npx` not found), the `command` path is wrong for this host.

**Fix:** Point `mcpServers.context7.command` to a real `npx` on this machine, for example:

- Run `command -v npx` in the workspace shell and paste that absolute path, or
- Use a wrapper such as `/usr/bin/env` with `PATH` including Node, or `bash -lc 'exec npx -y @upstash/context7-mcp@latest'` if your shell profile loads `nvm`/`fnm`.

Do not commit `~/.claude/settings.json` into the repo.

## Memory lanes (Claude Code project memory)

For which `~/.claude/projects/-workspace-sounio/memory/*.md` files to load per task type, see **`.claude/MEMORY_LANES.md`** in this repo.

## Adding a new task

1. Drop a `<task>.md` file in `.claude/offload-tasks/` with a `# Use case:` header line.
2. Add a `default_provider_for()` case in `bin/llm-offload`.
3. Add a row to the table above.
4. Verify with `bin/llm-offload --list-tasks`.

## Track of high-impact reviews

Append to `.claude/vancomycin_track.md` "LLM Review Notes" section after each non-trivial offload. The 2026-04-30 Grok 4.1 math-review caught a sign-error bug in `vp_cmin_point` monotonicity that would have shipped to a referee — exemplar of why this layer exists.
