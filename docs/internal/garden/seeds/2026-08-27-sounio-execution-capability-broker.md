<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-sounio-execution-capability-broker
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-sounio-execution-capability-broker
-->

# Sounio Execution Capability Broker

Status: Garden seed

## First Phrase

Do not ask whether a shell command looks safe. Replace ambient shell authority
with an execution capability whose meaning was decided by Sounio before any
process exists.

## Intuition

A pre-tool hook that searches command text for `python` or `rustc` observes only
spelling. It does not control an interpreter reached through `env`, a symlink, a
script shebang, a nested shell, or a later child process. Conversely, denying
all shell use destroys the development environment while proving little about
semantic authority.

The candidate distinction is:

```text
command text != measured execution closure != authority to execute
```

Codex and Claude can replace a pending Bash tool input before execution. Loom
can therefore issue a one-use capability that binds the original event hash,
current worktree, frozen Sounio semantics, measured command closure, allowed
purpose, and expiry. The tool runs only through the existing OCaml Guardian,
which supervises child execution and asks a Sounio policy whether each measured
language-role-purpose tuple is admissible.

## Proposed Evidence Order

```text
Garden
-> Sounio executable decision table
-> frozen Sounio execution semantics
-> OCaml parser and broker parity
-> hook input replacement
-> child-exec sabotage controls
-> commit and CI gates
```

No OCaml classifier may be attached to Bash before the Sounio executable and
its expected results are frozen by hash.

## Candidate Invariants

- Python and Rust remain non-waivable, including indirect or renamed launchers
  that the Guardian can resolve to those runtimes.
- Missing, timed-out, malformed, incomplete, or dynamically unclassifiable
  policy evidence refuses before execution.
- Shell syntax is a transport representation, never a semantic role.
- Sounio alone may create semantics or expected results.
- Lean, Koka, C++, and Haskell remain parity consumers of an exact frozen
  Sounio parent.
- OCaml may measure, supervise, journal, and execute an admitted capability,
  but cannot define what the capability means.
- External LLM output remains review-only and cannot become an expected result.
- Every admission or refusal records source, semantics, producer, role,
  toolchain, hardware, command, and result.
- Commit and CI are new execution surfaces, not privileged bypasses.
- A founder waiver is scoped, purposeful, expiring, receipt-bound, and cannot
  waive Python/Rust or semantic-authority requirements.

## Smallest Differentiating Experiment

1. Freeze a Sounio executable that admits a fully measured OCaml mechanical
   command after `SEMANTICS_FROZEN`.
2. Replay the same frame as Python and require refusal.
3. Remove only the Sounio Python-refusal rule, rebuild, and require the unchanged
   Python frame to become admissible.
4. In the native broker, attempt direct Python, `env python`, a symlink named as
   an allowed tool, a Python shebang, a nested shell, and a renamed executable.
5. Require every treatment to refuse before its sentinel side effect, while a
   matched native control executes.

The experiment is falsified if the Sounio sabotage does not change the semantic
decision, if any prohibited interpreter reaches its sentinel, or if an admitted
control cannot be tied to the exact frozen parent hash.

## Explicit Non-Claims

- This seed does not establish a secure sandbox or hostile-root boundary.
- It does not prove complete language identification for arbitrary binaries.
- It does not authorize ptrace, seccomp, or any particular supervision
  mechanism before the Sounio semantics exist.
- It does not open parity or make Loom claim-ready.
- It does not make Python, Rust, shell, OCaml, or documentation an oracle.
