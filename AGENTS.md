# AGENTS.md

<!-- context7 -->
Use the `ctx7` CLI to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service -- even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use even when you think you know the answer -- your training data may not reflect recent changes. Prefer this over web search for library docs.

Do not use for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

## Steps

1. Resolve library: `npx ctx7@latest library <name> "<user's question>"`
2. Pick the best match (ID format: `/org/project`) by: exact name match, description relevance, code snippet count, source reputation (High/Medium preferred), and benchmark score (higher is better). If results don't look right, try alternate names or queries (e.g., "next.js" not "nextjs", or rephrase the question)
3. Fetch docs: `npx ctx7@latest docs <libraryId> "<user's question>"`
4. Answer using the fetched documentation

You MUST call `library` first to get a valid ID unless the user provides one directly in `/org/project` format. Use the user's full question as the query -- specific and detailed queries return better results than vague single words. Do not run more than 3 commands per question. Do not include sensitive information (API keys, passwords, credentials) in queries.

For version-specific docs, use `/org/project/version` from the `library` output (e.g., `/vercel/next.js/v14.3.0`).

If a command fails with a quota error, inform the user and suggest `npx ctx7@latest login` or setting `CONTEXT7_API_KEY` env var for higher limits. Do not silently fall back to training data.
<!-- context7 -->

## Purpose

This file is the Codex-facing project contract for the Sounio repository.

Before interpreting or challenging the founder's research direction, read
`FOUNDER_INTENT.md`. It is the shared identity and collaboration contract across
AI threads. It does not override executable evidence or this file's operational
rules.

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

1. Run `./sounio-whereami --quick`.
2. Read `ONBOARDING.md`.
3. Read `CLAUDE_HANDOFF.md`.
4. Confirm the current branch.
5. Treat `/workspace/sounio` as the active development surface when operating in the promoted workspace.
6. Use the Sounio Compiler Foundry or Slurm path for heavy validation; do not run full stress directly in `/workspace/sounio`.

Important context:

- This repository was recovered from VM `sounio-dev-01`.
- The recovery was based on tarball import, then Git re-attachment.
- The historical safe base branch is:
  - `integration/sounio-dev-ready-base`
- The current checked-out branch is the operational truth for this session.
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

Blocker ownership, severity, evidence level, handoff shape, and merge eligibility
for parallel work are governed by:

- `.claude/PARALLEL_BLOCKER_CONTRACT.md`

Lanes that alter concept, type, effect, IR-field, scientific-interpretation, or
claim meaning must also declare their Concept-IDs and semantic boundaries under:

- `docs/internal/concepts/SEMANTIC_LANE_CONTRACT.md`
- `docs/internal/concepts/registry.tsv`

Use `bash scripts/dev/sounio_semantic_status.sh` before assigning overlapping
semantic work. Scanner output is observational; dirty worktrees are potential
writers, not automatic ownership claims.

### Live lane coordination

`bin/sounio-coord` is the live, cross-worktree coordination surface. It stores
short-lived leases outside Git, keyed by the repository's shared Git directory,
so attached worktrees see the same active claims.

Before the first write in an implementation lane:

1. Run `bin/sounio-coord brief` (also shown by `./sounio-whereami --quick`).
2. Claim the exact write set:
   `bin/sounio-coord claim --agent <id> --lane <id> --intent "<goal>" --files <paths...>`.
3. Keep long-running work alive with
   `bin/sounio-coord heartbeat --agent <id> --lane <id>`.
4. Release on completion, abort, or handoff with
   `bin/sounio-coord release --agent <id> --lane <id> --reason "<result>"`.

Put `--files` last and quote glob scopes such as
`'self-hosted/compiler/**'`. The command refuses overlapping active claims.
Claims expire after four hours by default; expiry makes abandoned work visible,
but does not authorize overwriting dirty files. Git status and executable repo
truth still outrank coordination metadata. Treat this as runtime presence, not a
durable handoff record; blockers and consequential handoffs still belong in the
repository contracts named below.

Project hooks in `.codex/hooks.json` and `.claude/settings.json` automate the
common path. They register session presence, refresh a 30-minute lease during
active turns, and reserve files before structured `Write`, `Edit`, or
`apply_patch` calls. Claude releases its session lease on `SessionEnd`; Codex
currently has no session-end event, so its inactive hook lease expires. Codex
users must review and trust the project hook with `/hooks` once per hook hash.
Shell commands can write arbitrary files and cannot be scoped reliably, so a
manual exact scope remains mandatory before write-bearing Bash commands. The
startup hook prints the session's agent/lane identity; reuse it with
`bin/sounio-coord scope --agent <session-agent> --lane <session-lane> --intent "<goal>" --files <paths...>`
instead of creating a second overlapping lease.

Agents can exchange live messages across worktrees:

- Send a directed request with
  `bin/sounio-coord send --agent <id> --lane <id> --to-agent <id> --to-lane <id> --kind request --message "<text>"`.
- Read pending messages with
  `bin/sounio-coord inbox --agent <id> --lane <id>`.
- After acting on a message, acknowledge it with
  `bin/sounio-coord ack --agent <id> --lane <id> --message <message-id>`.

Prompt and post-tool hooks inject unread messages into the agent's active turn.
A rejected structured write also sends a request to the current owner
automatically. Use
`info`, `request`, `reply`, `blocker`, or `handoff` as message kinds. Messages
coordinate work in progress; durable blockers still require the blocker
contract.

If an agent leaves a blocker for another agent, it must use that contract's
Blocker-ID, severity, class, evidence, owner, worktree, branch, acceptance gate,
and next-action fields.

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
- `scripts/lib/resolve_souc.sh` for the public compiler entrypoint (`bin/souc`).
- `scripts/lib/resolve_madaros.sh` for the Stage1 modular compiler (`bin/madaros`).

as the canonical compiler-resolution logic unless the task explicitly targets another resolver for cleanup or compatibility reasons.

`bin/souc` is currently a compatibility wrapper whose default engine is Madaros.
The legacy lean_single engine remains the bootstrap seed and can be forced with
`SOUNIO_SOUC_ENGINE=lean_single`. For fleet coordination and stale-worktree
rules, read `docs/MADAROS_STATUS.md`; stale raw `artifacts/self-hosted/madaros`
binaries are not evidence against current `origin/main`.

### Test harness
Use:
- `scripts/run_sio_test_suite.sh`

for suite execution.

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
- Classify remaining blockers using `.claude/PARALLEL_BLOCKER_CONTRACT.md`
- Do not claim semantic milestones more broadly than the evidence supports
- Separate:
  - ontology semantics
  - validation-wrapper routing
  - bootstrap/runtime repair

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

When Context7 applies, use the `ctx7` CLI flow above:

1. `npx ctx7@latest library <name> "<user's question>"`
2. `npx ctx7@latest docs <libraryId> "<user's question>"`

Do not skip the library-resolution step unless the user already provided a valid `/org/project` library ID.
Do not run more than three Context7 commands for one question.

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
8. blocker contract status for any unresolved blocker: Blocker-ID, severity, class, evidence level, owner, acceptance gate, and next action
9. **LLM-offload reviews invoked** (provider, task, target, outcome) — required for any work touching math claims, clinical-pathway code, or external-facing artifacts. See `.claude/AGENT_OFFLOAD_POLICY.md`.

---

## LLM-offload policy (mandatory)

This repository runs many parallel agents with a single human author. Pre-commit review by orthogonal LLM providers via `bin/llm-offload` is **mandatory** at the checkpoints listed in `.claude/AGENT_OFFLOAD_POLICY.md`:

- **Math claims** (PK/PD, GUM, p-box, Lean statements, refinement invariants) → `bin/llm-offload -t math-review -p xai`
- **Clinical-pathway code** (`stdlib/clinical/*`, vancomycin tests, clinical Lean) → `bin/llm-offload -t review -p deepseek`
- **External-facing artifacts** (papers, cover letters, dissertation, IRB) → `bin/llm-offload --raw <draft> deepseek xai gemini`

Every non-trivial offload appends to `.claude/llm_offload_log.md`. Bug-catching offloads require an `LLM-offload-review:` trailer in the commit message. The policy document lists fallback rules when a provider key is missing or down.

**Codex agents must not skip this step**. If you find yourself about to commit a `vancomycin_*.sio`, a Lean theorem statement, or a paper draft without a logged offload review, stop and run the review first.

## Agent coordination — read the bus before you start

Ten agent slots share this pod (`claude-1..3`, `codex-1..3`, `grok-cli1..2`,
`kimi-cli1..2`) and one filesystem. Coordination used to be a document that
nobody wrote to. It is now a channel:

```bash
scripts/dev/agent-bus.sh brief          # FIRST THING. hazards, leases, recent events
scripts/dev/agent-bus.sh claim <res>    # before a build lock, a shared file, a lane
scripts/dev/agent-bus.sh post finding 'what you learned'
scripts/dev/agent-bus.sh hazard add <slug> 'what will silently ruin others' measurements'
```

It is not push — nothing interrupts another agent's loop. You hear others when
you read, so the whole protocol is: **`brief` before you start, `post` when your
state changes.** Leases expire, so a crashed agent never parks a resource.

For BeagleCockpit and anything else that has to know as things happen, the same
bus is served over MCP (`scripts/mcp/agent_bus_mcp.py`, merge `scripts/mcp/agent-bus.mcp.json` into your gitignored `.mcp.json`).
Subscribe to `bus://events` or `bus://hazards` and the server sends
`notifications/resources/updated` the moment another agent posts — that is real
push, not polling. Tools: `bus_post`, `bus_claim`, `bus_release`, `bus_hazard`,
`bus_brief`. Both doors write the same storage, so an agent on the shell CLI and
an agent on MCP are on one channel.

Post a `hazard` for anything that makes a measurement lie rather than fail:
a poisoned environment variable, a stale artifact, a checkout parked on another
branch. Those cost hours precisely because the run still exits and prints a
number. Storage is `/workspace/.agents/bus`, outside every checkout, because
agents work in different worktrees.
