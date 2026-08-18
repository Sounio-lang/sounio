<!-- docs:meta
topic_id: repo.docs.audit.portfolio-imported-native-139-untag-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.portfolio-imported-native-139-untag-2026-08-18
-->

# Portfolio imported/native 139 — full run and un-tag

**Date:** 2026-08-18  
**Lane:** `lane/grok-cli3/portfolio-139-untag-20260818`  
**Base:** `origin/main` (branched at `ef7067e48b`)  
**Question asked first:** is the problem the number (token/source CAP)? **No.**

## Measured

Instrument: source-built `artifacts/self-hosted/madaros` (99 964 767 B, 2026-08-17 17:01) via `MADAROS_RAW_BIN`, `ulimit -s 524288`, `SOUNIO_STDLIB_PATH` this worktree. Classifier SIGSEGV → 139; tiny control → 0. `rc` read from `souc run` itself.

All **169** `tests/run-pass` files with `use theorem::portfolio::*`, sequential, one process:

| class | n |
|-------|--:|
| PASS `rc=0` (fingerprints matched) | **168** |
| FAIL `rc=1` (first fingerprint mismatch) | **1** |
| FAIL139 | **0** |

Elapsed ~32 minutes. Machine table: `.scratch/portfolio_139_full.tsv`.

## What changed

- Removed `//@ known-failure` … `139` from the **168** files that returned 0.
- Left **one** tag, rewritten so it does not claim 139:

`tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_obligation_seed_v168_imported.sio`

That file **compiles**, writes an ELF, and `main` returns **1** (`receipt_fp != 455287687`). Rechecked: `recheck_rc=1`. Catalog expected-value mismatch, not a crash, not a ceiling.

## What this is not

- Not a CAP raise. Façades remain 1130 / 1127 bytes. E229 still off main.
- Not a compiler 139 fix. Zero sampled 139s.
- Not a close-out of the other ~248 imported 139 tags outside this set.

## Parallel

Leaving a 139 annotation on a file that returns 1 would be a plausible failure where the suite should see a fingerprint miss — the same class as silent `pin_count=0` (`#1830`).
