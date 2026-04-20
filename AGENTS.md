# AGENTS.md

## Purpose

This file is the Codex-facing project contract for the Sounio repository.

It does **not** replace `CLAUDE.md`.
Instead:

- `CLAUDE.md` remains the active source of truth for Claude Code behavior
- `AGENTS.md` is the Codex-facing execution contract
- repository facts and executable scripts override stale documentation when they disagree

When in doubt, prefer:
1. actual repo files and executable scripts
2. committed governance docs
3. historical prompt contracts
4. assumptions

## Recovery + Remote-First Bootstrap

Before implementation work:

1. Read `CLAUDE_HANDOFF.md`.
2. Confirm the current branch.
3. Treat `/workspace/sounio` as the active development surface when operating in the promoted workspace.

For non-trivial work, continue with this repo-local read order after `CLAUDE_HANDOFF.md`:

1. `README.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `docs/guide/MINIMUM_VIABLE_SOUNIO.md`
5. `docs/guide/LLM_PROGRAMMING_GUIDE.md`

Important context:

- This repository was recovered from VM `sounio-dev-01`.
- The recovery was based on tarball import, then Git re-attachment.
- The safe active branch is:
  - `integration/sounio-dev-ready-base`
- Older Claude/Codex artifacts may still reference the VM-era path:
  - `/home/demetrios/RustroverProjects/sounio`
- The current remote-first workspace path is:
  - `/workspace/sounio`

Do not treat stale VM paths as the current execution surface.
Preserve recovery state and avoid destructive "align with main" workflows unless explicitly requested.

---

## Repository identity

Sounio is a self-hosted compiler/language repository with an epistemic/scientific focus.
This repository contains:

- active self-hosted compiler implementation
- bootstrap/native compiler paths
- ontology validation work
- governance/docs/history for parallel agent workflows
- ecosystem/template subtrees that are **not** the active root config for this repo

Do not confuse root repo config with:
- `ecosystem/claude-code-sounio/`
- `ecosystem/create-sounio/templates/`

Those are distribution/template surfaces, not the active repo root behavior.

---

## Authority model

### Claude Code authority
`CLAUDE.md` at repo root is the active Claude Code source of truth.

### Codex authority
This file (`AGENTS.md`) is the active Codex source of truth for this repo.

### Shared reality
Executable repo scripts and actual file layout are the source of truth for:
- compiler resolution
- bootstrap path
- test harness routing
- ontology validation path

If a prose document disagrees with a script that is actually wired into the repo flow, trust the script and report the mismatch.

---

## Parallel-work contract

Use this division unless the user explicitly overrides it.

### Claude Code
Use for:
- read-only repository surveys
- inventory
- conflict mapping
- secondary review of diffs
- documentation lookup with Context7 when external docs are needed

### Codex
Use for:
- implementation
- file creation
- surgical refactors
- test harness wiring
- validation-path work
- small reversible commits

### Never edit in parallel
Do not let Codex and Claude Code edit the same file in the same phase.

If a file is under active edit by one agent, the other agent may:
- inspect it
- comment on it
- review it after the edit is complete

but must not write to it concurrently.

---

## Files that must not be edited in parallel

Treat these as high-risk shared control files:

- `CLAUDE.md`
- `.claude/settings.json`
- `.claude/settings.local.json`
- `scripts/lib/resolve_souc.sh`
- `scripts/run_sio_test_suite.sh`
- `scripts/ci/build_native_souc.sh`
- `bin/souc`
- `.github/workflows/ci.yml`
- `scripts/post-tool-use.sh`

If a task requires changes to any of these, isolate that task and avoid concurrent edits elsewhere that depend on them.

---

## Canonical compiler/test resolution

Do not hardcode ad hoc compiler routing if a repo resolver already exists.

### Canonical resolution path
Use:
- `scripts/lib/resolve_souc.sh`

as the canonical compiler-resolution logic unless the task explicitly targets another resolver for cleanup or compatibility reasons.

### Test harness
Use:
- `scripts/run_sio_test_suite.sh`

for suite execution.

### Codex helper entrypoints
Use the repo-local Codex wrappers under:
- `scripts/codex/setup.sh`
- `scripts/codex/action-bootstrap-smoke.sh`
- `scripts/codex/action-build-ontology-validation-souc.sh`
- `scripts/codex/action-ontology-harness-default.sh`
- `scripts/codex/action-ontology-harness-diff.sh`
- `scripts/codex/action-ontology-harness-rebuilt.sh`

These wrappers should resolve the compiler through:
- `scripts/lib/resolve_souc.sh`

rather than hardcoding `./bin/souc` directly.

### Ontology validation
When ontology work is the target, prefer the rebuilt/current-source validation path when available.

Current ontology validation scripts live under:
- `scripts/ci/build_ontology_validation_souc.sh`
- `scripts/ci/run_ontology_validation.sh`

If these scripts are untracked or not yet canonical, do not silently assume they are CI truth.
Report whether you are using:
- default/stale path
- rebuilt/current-source wrapper path

---

## Execution style

- Keep changes small, reversible, and test-backed
- Preserve bootstrap safety
- Preserve legacy code until parity is proven
- Prefer explicit failure classification over silent fallback
- Do not claim semantic milestones more broadly than the evidence supports
- Separate:
  - ontology semantics
  - validation-wrapper routing
  - bootstrap/runtime repair

---

## Language and commit constraints

- Do not add AI attribution such as `Co-Authored-By: Codex`.
- Sounio is not Rust. Prefer repo-native syntax such as `var` and `&!`, and avoid Rust-specific macros, attributes, and closure literals.
- For syntax and semantics, consult:
  - `docs/guide/LLM_PROGRAMMING_GUIDE.md`
  - `docs/compiler/KNOWN_LIMITATIONS.md`
- For direct compiler invocations outside existing harnesses, prefer the `SOUC_BIN` resolved by `scripts/lib/resolve_souc.sh`.
- Commit titles should follow the repo component prefix convention described in `CLAUDE.md`:
  - `[component] Brief description`

---

## Ontology-specific rules

For ontology work:

- treat the rebuilt current-source wrapper as the primary semantic validation path when available
- keep legacy ontology storage/query code until parity is proven
- classify failures as:
  - build/bootstrap-path
  - harness-routing
  - ontology-kernel/checker
  - baseline noise

Do not collapse these categories in reports.

---

## Context7 usage

Use Context7 only for:
- external tool/library/framework documentation
- up-to-date Codex / MCP / Claude Code docs
- version-specific external references

Do **not** use Context7 as a substitute for repository-local truth.
Repository-local truth must come from direct inspection of the repo.

---

## What not to touch unless explicitly requested

Avoid editing these areas unless the task explicitly targets them:

- `ecosystem/claude-code-sounio/`
- `ecosystem/create-sounio/templates/`
- large memory/archive artifacts
- historical governance docs unless the task is governance cleanup

---

## Commit discipline

Prefer commit-sized phases:

1. diagnosis or plumbing
2. implementation
3. read-path switch
4. tests
5. cleanup only after green gates

Do not mix:
- semantic changes
- bootstrap-path repairs
- governance rewrites
- large cleanup

in one commit unless the user explicitly asks for that tradeoff.

---

## Reporting requirements

At the end of substantial work, report:

1. files changed
2. what changed in each file
3. commands run
4. results
5. whether the work used:
   - default path
   - rebuilt wrapper
   - fallback path
6. remaining blockers
7. whether any legacy path was intentionally kept
