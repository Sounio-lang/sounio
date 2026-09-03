<!-- docs:meta
topic_id: repo.docs.audit.madaros-archive-triage-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-archive-triage-2026-06-21
-->

# Madaros Archive Triage — 2026-06-21

## Scope

This note resolves the Bucket B `docs/audit/MADAROS_*.md` items from the
primary-checkout archive without replaying stale compiler diagnostics as current
truth.

Archive source:

- `/workspace/sounio/docs/audit/MADAROS_*.md` in the protected dirty primary
  checkout.

Current baseline used for this triage:

- `origin/main` at `044969dfe47bbb800bf868779b7e93f8b52981e0`
  (`Merge pull request #349 from Sounio-lang/codex/hello-example-io`).
- `bin/madaros` SHA256:
  `0159bb83fd47bcfab3fd424f91f02023f23ec0cd2028f3c543fa3210a8a0f13c`.
- `bin/madaros-linux-x86_64` SHA256:
  `506d24c47e6a735340b0f8ced2072fa1baf485bb8d65461857b5a8d5565b0cef`.

## Disposition Summary

| Archive item | Disposition | Reason |
|---|---|---|
| `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md` | already represented on `origin/main` | The checked-in file is newer than the archived primary copy and includes the 2026-06-20 production Slurm result. |
| `MADAROS_NATIVE_V2_CODEGEN_CENSUS_2026-06-19.md` | archive-only, superseded | The raw census predates the checked-in `MADAROS_PRINT_INT_DISPATCH_2026-06-20.md` and `MADAROS_FOR_LOOP_LOWERING_2026-06-20.md` follow-ups, and it contains acknowledged harness false readings for enum and method cases. |
| `MADAROS_ENUM_VARIANT_SIGSEGV_2026-06-20.md` | archive-only, do not promote raw | Useful forensics, but it depends on stripped-binary/core state from an older prebuilt and overlaps the active Root 2 compiler lane. |
| `MADAROS_METHOD_CALL_SIGSEGV_2026-06-20.md` | archive-only, do not promote raw | Useful correction to the old census, but current disposition is tied to the same by-value/SRET family being handled in compiler work. |
| `MADAROS_ROOT2_ENUM_FNCOUNT_2026-06-20.md` | archive-only, do not promote raw | Deep core note; needs a fresh repro or trace on the current Madaros lane before becoming authoritative repo docs. |
| `MADAROS_SRET_ROOT_SYNTHESIS_2026-06-20.md` | archive-only, do not promote raw | Historical synthesis of Root 1 and Root 2; current action belongs in the compiler lane, not a docs-only replay. |

## Current Repo Truth

The current repo already has three governed Madaros audit notes:

- `docs/audit/MADAROS_BOXNEW_SIGSEGV_2026-06-19.md`
- `docs/audit/MADAROS_PRINT_INT_DISPATCH_2026-06-20.md`
- `docs/audit/MADAROS_FOR_LOOP_LOWERING_2026-06-20.md`

Those files are the checked-in audit surface for the Box, integer-print, and
for-loop findings. The primary archive also contains later enum/method/root
notes, but they are not safe to promote as standalone truth because:

- they were written against an older prebuilt state,
- several statements are explicitly corrections to an earlier broken harness,
- they overlap compiler-adjacent Root 2/SRET work,
- the active compiler lane must own source fixes and fresh reproducer evidence.

## Actionable Compiler Handoff

The archived enum, method, Root 2, and SRET notes should be treated as evidence
for the compiler lane, not as source changes and not as independent docs truth.

If another agent wants to convert them into a compiler task, use the blocker
contract shape from `.claude/PARALLEL_BLOCKER_CONTRACT.md`:

- Blocker-ID: `MADAROS-ROOT2-SRET-ARCHIVE-TRIAGE-2026-06-21`
- Severity: high
- Class: compiler/runtime-lowering
- Evidence level: archive forensics plus current repo triage
- Owner: compiler lane
- Acceptance gate: fresh current-branch repro for enum variant and method-call
  cases, then a focused Madaros/native gate on the same branch that changes
  compiler code
- Next action: re-run the enum/method repros on the current compiler lane and
  decide whether the Root 2/SRET fix already covers them or needs a focused
  patch

## Explicit Non-Actions

This PR intentionally does not:

- copy the raw archived `MADAROS_*.md` files into `origin/main`,
- edit `self-hosted/*`,
- edit `bin/madaros` or the raw Madaros ELF,
- claim the Root 2/SRET compiler blocker is fixed.

The result is a source-controlled disposition for the archive bucket, while the
actual compiler repair remains in the compiler lane with non-overlap preserved.
