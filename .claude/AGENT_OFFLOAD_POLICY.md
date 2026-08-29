# Agent LLM-Offload Policy (Sounio)

**Authority**: this document is the canonical source of truth for **all agents** working on Sounio (Claude Code, Codex, Cursor agents, generalPurpose subagents, parallel best-of-N runners).

Referenced from `CLAUDE.md`, `AGENTS.md`, `.cursorrules`. When this file disagrees with one of those, this file wins for offload-related rules.

**Status**: locked 2026-04-30 after Grok 4.1 caught a sign-error bug in `vp_cmin_point` monotonicity that had passed Lean theorem statements + 4 green tests + author self-review.

## Core principle

**A single human author (Demetrios) directs many parallel agents.** Pre-commit math/code/prose review by orthogonal LLM providers is cheap (~$0.001 / call, ~30 s) and demonstrably catches real bugs. Therefore: route routine review through `bin/llm-offload` rather than re-asking the same primary agent.

## Mandatory checkpoints

The following checkpoints are **MANDATORY** for any agent before commit / PR open / submission:

### M1 — Mathematical claims

**Trigger**: any new or modified content that contains a non-trivial mathematical derivation, including but not limited to:

- Hand-derived monotonicity / sign / convexity claims in `.sio` comments
- PK/PD formulas in `stdlib/clinical/*`, `stdlib/epistemic/*`
- GUM uncertainty-propagation derivations
- p-box / interval-extension arithmetic
- Lean theorem statements in `formal/lean4/*` (especially when `sorry` / `trivial` placeholders exist)
- Refinement-type invariants

**Action**: run

```bash
bin/llm-offload -t math-review -i <file_or_diff>
```

before commit. **As of 2026-07-07 this fans out by default to two independent providers — xai (grok-4.3) and zai (Z.AI GLM) — for every agent.** A single-vendor pass is no longer the standard for math claims; the independent second opinion is mandatory. (Z.AI requires `ZAI_API_KEY`/`ZHIPU_API_KEY`; if absent, the run degrades to xai-only and logs a SKIPPED notice — treat a Z.AI skip as an incomplete review, not a pass.) If any provider rejects a claim, EITHER fix it OR document the disagreement in `.claude/llm_offload_log.md` with explicit reasoning. Precedent for this rule: on 2026-07-07 grok-4.3 + grok-4.20-reasoning caught a sign error making the NeuroDyn "octonion" product non-normed/non-alternative that had passed all prior review.

For high-stakes math (theorem published / referee-bound), fan out:

```bash
bin/llm-offload --raw <prompt> xai qwen mistral
```

### M2 — Clinical-pathway code

**Trigger**: any new or modified file under `stdlib/clinical/`, `tests/run-pass/vancomycin*`, `tests/stdlib/clinical/`, or `formal/lean4/SounioVancomycin*`.

**Action**: run

```bash
bin/llm-offload -t review -p deepseek -i <file>
```

before commit. The reviewer prompt is hostile and will surface clinical-safety issues. If the reviewer flags BLOCKER/MAJOR, the commit is blocked until resolved or explicitly waived in `.claude/llm_offload_log.md` with rationale.

### M3 — External-facing artifacts

**Trigger**: any artifact destined for publication / submission / external eyes — papers, cover letters, dissertation chapters, conference abstracts, IRB protocols.

**Action**: before submission, fan out to ≥ 2 providers:

```bash
bin/llm-offload --raw <draft.md> deepseek xai gemini
```

Diff the responses; address every BLOCKER and MAJOR issue. Log in `.claude/llm_offload_log.md`.

### M4 — Atomic commit messages on review-driven fixes

When a `llm-offload` review catches a real bug, the commit message **must** reference:

- the provider that caught it
- the task type (`math-review`, `review`, …)
- a one-line description of the issue

Format:

```
[component] Brief description

…

LLM-offload-review:
  provider: xai (Grok 4.1 fast reasoning)
  task: math-review
  issue: <one line>
```

This makes the bug-catch traceable and preserves attribution.

## Encouraged but optional

| Task | When | Default provider |
|------|------|------------------|
| `expand` | outline → draft prose | `gemini` |
| `paraphrase` | cover letters, abstracts | `minimax` (fallback `qwen`) |
| `scaffold` | spec → boilerplate | `deepseek` |

## Files that must not be touched without offload review

- `formal/lean4/Sounio*.lean` — every new theorem statement: `math-review` mandatory.
- `stdlib/clinical/*.sio` — every change: `review` mandatory.
- `stdlib/epistemic/{knightian,composed_effects,knowledge}.sio` — every math-touching change: `math-review` mandatory.
- Cover letters, abstracts, paper prose under `docs/papers/` — every external-facing edit: fan-out review mandatory.

## Routing reference

Lives in `.claude/offload-routing.md`. Run:

```bash
bin/llm-offload --status         # which keys are loaded
bin/llm-offload --list-tasks     # available task prompts
bin/llm-offload --list-providers # all providers
```

Task-specific system prompts: `.claude/offload-tasks/<task>.md`.

## Audit log

Every non-trivial offload (caught a bug, informed a design decision, blocked a commit) goes in `.claude/llm_offload_log.md`. Append-only. Format documented in that file's header.

## Failure modes

| Mode | Action |
|------|--------|
| API key missing for default provider | Fall back per `--list-providers`. Log substitution in commit message. |
| Provider down / timeout | Retry with second-choice provider. If all fail, document in `.claude/llm_offload_log.md` with timestamp and proceed; flag for re-review on next session. |
| Reviewer hallucinates a problem | Log the disagreement in the audit log with reasoning; do not silently dismiss. |
| Reviewer caught real bug | Apply fix; log entry; reference in commit message per M4. |

## What NOT to offload

- **Trust decisions** (what model is the primary agent? which branch is canonical?). These belong to the human author.
- **Permission-bearing operations** (deleting branches, force-pushing, IRB submission, journal submission). Always human-in-the-loop.
- **PHI / patient-identifying data**. Never sent to external providers under any task.

## Enforcement

The policy is enforceable via an **opt-in pre-commit gate**:

```bash
scripts/dev/check_offload_policy.sh --install     # install as .git/hooks/pre-commit
scripts/dev/check_offload_policy.sh               # one-shot check on staged files
scripts/dev/check_offload_policy.sh --uninstall   # remove the hook
```

The gate scans staged files against three protected-path regex sets (math, clinical, external-facing) and verifies that `.claude/llm_offload_log.md` contains a today-dated row whose `Target` column substring-matches each touched file's basename. Missing evidence blocks the commit with a remediation message.

To bypass with rationale (rare), append a row to the audit log with Outcome `WAIVED` and an inline justification.

The hook is **opt-in by default** so it does not surprise existing local clones. CI may flip this to `installed-by-default` after a transition period.

Compliance is on the agent regardless of whether the hook is installed. Repeated non-compliance (committing math-bearing changes without `math-review` evidence in the audit log) is grounds to revert the commit.

## Origin story

This policy exists because, on 2026-04-30, in a single session, the author (acting as Opus 4.7) wrote the M3 vancomycin PBPK pipeline, type-checked it, ran 4 green tests, drafted Lean theorem statements, and shipped to `commit-ready` state — with a sign error in the monotonicity comment of `vp_cmin_point` that propagated into the corner-enumeration code. A 28-second `bin/llm-offload -t math-review -p xai` call invoked **after** the work was "complete" surfaced the bug symbolically with a numeric counter-example. The fix changed the pre-TDM Cmin band from `[11.30, 21.31]` to the correct `[8.49, 24.29]` and made the clinical refusal narrative stronger.

The catch took 28 seconds and ~$0.001. It would have appeared in a POPL or *Clinical Pharmacokinetics* referee comment otherwise. **This policy makes that catch the default, not the exception.**
