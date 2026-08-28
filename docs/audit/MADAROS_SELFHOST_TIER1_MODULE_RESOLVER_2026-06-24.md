<!-- docs:meta
topic_id: repo.docs.audit.madaros-selfhost-tier1-module-resolver-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-selfhost-tier1-module-resolver-2026-06-24
-->

# Madaros self-hosting tier-1: module resolver searches ancestor dirs (2026-06-24)

*Branch off `main`. First milestone toward Madaros self-hosting (compiling its own source):
make Madaros resolve all of the compiler's modules.*

## Root cause

`madaros --check self-hosted/compiler/main.sio` loaded only **13 of ~72 imported modules**, so
~59 modules' functions/constants resolved as undeclared → **~6,000 "use of undeclared
variable" errors**. Individual modules check clean; the bundle did not.

The resolver `module_frontend_resolve_import_relpath` built candidate paths from the importing
file's directory (`cur_dir`) plus a `stdlib/` fallback — but the compiler's modules share one
source root (`self-hosted/`), and `main.sio` lives in `self-hosted/compiler/`. So
`check::types` resolved to `self-hosted/compiler/check/types.sio` (missing) instead of the
sibling `self-hosted/check/types.sio` (exists), and was skipped.

## Fix

Added an **ancestor-directory search**: after `cur_dir/rel_path`, walk up parent directories
trying `<ancestor>/rel_path` (closest ancestor wins, depth-bounded). `check::types` from
`self-hosted/compiler/` now finds `self-hosted/check/types.sio`.

This is Madaros's own resolver (not the lean_single build path), so it only improves Madaros's
self-resolution; for ordinary programs the existing `cur_dir`/`stdlib` candidates still win
first, so there is no behaviour change for single-module or stdlib imports.

## Verified
- `madaros --check main.sio` now loads **117 modules** (was 13); the **~6,000
  undeclared-variable errors are gone** (0 errors).
- No regression: 34/60 run-pass = prebuilt main +1, 0 regressed; madaros self-builds.

## Honest scope — what this unblocks vs. what remains
- This is **tier-1** of self-hosting (resolve). With all modules loaded, the next wall is
  **tier-2 (scale)**: type-checking all 117 modules together overflows the `Checker`'s
  fixed-size symbol tables (`StructTable=128` vs 1620 structs; `FnSigTable=4096` vs ~15K fns).
  Inflating the inline caps is a **dead end** — the tables are constructed *by value*
  (`fn_sig_table_new()` returns a multi-MB table), so big inline arrays just relocate the
  by-value-aggregate failure into checker init (verified). Tier-2 needs **heap-indirect
  tables** (Box the arrays) — a separate, focused refactor.

## AI disclosure
Fix by AI agent (Claude) under human direction; root isolated by measuring loaded-vs-imported
module counts and confirming sibling-directory layout. Every claim backed by a re-runnable probe.
