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

## CI Full Test Suite — two-engine split (2026-08-18)

GitHub Actions `Full Test Suite` compiles with **`souc-stage2`** (lean_single seed), not Madaros. After the un-tag it reported **Fail: 24**, all of them our importers, all `run exited 1`. Zero 139s.

| Slice of the 169 | n | Why |
|---|--:|---|
| Already had `//@ requires: madaros` | 145 | Harness skips unless `SOUNIO_MADAROS_AVAILABLE` |
| Missing that annotation; seed `rc=1` | **24** | Ran on lean_single; fingerprints do not match |
| Madaros `rc=1` residual (v168) | 1 | Still `known-failure` |

Local check: three of the 24 return **rc=1** under `SOUNIO_SOUC_ENGINE=lean_single` and **rc=0** under source-built Madaros.

**Fix:** add `//@ requires: madaros` to those 24 — same annotation the other 145 already carry. Do **not** re-attach a 139 lie. Do **not** raise CAP. The seed suite will skip them; a Madaros job that sets `SOUNIO_MADAROS_AVAILABLE` will run them.

## What this is not

- Not a CAP raise. Façades remain 1130 / 1127 bytes. E229 still off main.
- Not a compiler 139 fix. Zero sampled 139s.
- Not a close-out of the other ~248 imported 139 tags outside this set.

## Parallel

Leaving a 139 annotation on a file that returns 1 would be a plausible failure where the suite should see a fingerprint miss — the same class as silent `pin_count=0` (`#1830`).
