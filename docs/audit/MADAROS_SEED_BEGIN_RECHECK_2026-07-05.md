<!-- docs:meta
topic_id: repo.docs.audit.madaros-seed-begin-recheck-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-seed-begin-recheck-2026-07-05
-->

# Madaros `seed_begin` recheck — 2026-07-05 (post-PR #622)

Status: **BLOCKER PERSISTS** on the default path. Oracle lane green.

## Context

Recheck of the multi-module native lowering segfault first recorded in
[`MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md`](MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md),
triggered by the EISA precision-track plan (Phase 0 gate) after PR #622
(S3 HLIR receipts + S4 preflight) reported all remote checks green.

## Environment

- Working branch: `gpu/epistemic-tensor-core-next` at `88f4282fe` (2026-07-04)
- PR #622 commits (`4fe2ba090`, `73d9ecadc`, tip `3d50c951e`) live on
  `origin/work/madaros-v2-sota-codex` and are **not ancestors of this HEAD**.
- Inspection of `git show --stat` for both #622 commits confirms they touch
  `self-hosted/compiler/main.sio`, gates and docs — **not**
  `self-hosted/ir/lower.sio`. No fix for `seed_begin` should be expected from
  #622 even after merge.
- `bin/souc` default engine: Madaros (`artifacts/self-hosted/madaros`,
  v0.80.0, built 2026-07-04 17:06).
- Local uncommitted modifications present on `self-hosted/ir/lower.sio` and
  `self-hosted/compiler/module_frontend.sio` (pre-existing on this worktree).

## Measured matrix

Command family (per test):

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run tests/stdlib/theorem/test_smt_<name>.sio            # default
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run <same>               # oracle
```

Exit codes below are the raw shell exit of the invocation (the wrapper
surfaced the real 139 this time; the earlier false-green exit-0 shape did not
reproduce in this run, but output-based verification remains mandatory).

| Test (`tests/stdlib/theorem/`) | default (Madaros) | oracle (lean_single) |
|---|---|---|
| `test_smt_adaptive_epistemic.sio` | exit 139 — `lower_array: seed_begin` segfault | exit 0 — ALL PASS |
| `test_smt_beta_polarity_ts.sio` | exit 139 — same | exit 0 — ALL PASS |
| `test_smt_epistemic_eval.sio` | exit 139 — same | exit 0 — ALL PASS |
| `test_smt_regime_getters.sio` | exit 139 — same | exit 0 — ALL PASS |
| `test_smt_solver_basic.sio` | exit 139 — same | exit 0 — ALL PASS |
| `test_smt_thompson_stats.sio` | exit 139 — same | exit 0 — ALL PASS |

Failure shape (identical across all 6):

```
lower_array: seed_begin
bin/madaros: line 342: <pid> Segmentation fault (ulimit -v 16777216; exec timeout 300 "$RAW_MADAROS" "$src" -o "$out")
```

## Additional finding (2026-07-05, EISA Phase 1)

The blocker is **not specific to `theorem::smt`**. A freshly written
single-import module (`use math::dd64::*`, new file
`stdlib/math/dd64.sio`, pure functions + one struct, no arrays of
structs) reproduces the same segfault on the default lane:

```
lower_array: seed_done
lower_array: dep_begin 1
Segmentation fault (bin/madaros line 342)
```

All four `tests/stdlib/math/test_dd64_*.sio` witnesses: segfault on
default, `ALL PASS` on `SOUNIO_SOUC_ENGINE=lean_single`. This narrows the
class: any cross-module stdlib import that goes through
`lower_array` seed/dep materialisation appears affected, i.e. the
default path is currently unusable for **all multi-module stdlib
programs**, not merely the solver surface. Note the failure point here is
`dep_begin 1` (after `seed_done`), a slightly different stage than the
`seed_begin` shape of the smt matrix — record both shapes in the forensic
dispatch.

## Diff vs 2026-06-22 audit

No change in failure class, trigger (imports of `theorem::smt`), or message.
The 2026-06-24 seed fixes (`SEED_FIX_GENERAL_LVALUE_DEREF_STORE`,
`SEED_FIX_VALUE_NESTED_FIELD_STORE`) did not cover this path.

## Decision (EISA precision track, Phase 0 gate)

Per the approved plan matrix:

- The dd64 track (Phase 1) proceeds with witnesses executed on **both lanes**;
  each witness records its `validated_lane`. If dd64 tests (single stdlib
  import, no `theorem::smt`) pass on the default lane, they claim
  `validated_lane: default+lean_single`; otherwise `validated_lane:
  lean_single` explicitly. No silent green.
- Claims of "default path" support for the precision track are **blocked**
  until this blocker closes.

## Blocker contract fields

- Blocker-ID: `BLK-MADAROS-SEED-BEGIN`
- Severity: high (blocks scientific imports on default path)
- Class: compiler / native-lowering (`self-hosted/ir/lower.sio`, array seed
  materialisation across module boundary)
- Evidence: this matrix (6×2), reproduction commands above, prior audit
  2026-06-22
- Owner: unassigned (needs dedicated forensic dispatch; out of scope for the
  EISA precision track)
- Acceptance gate: 6/6 `ALL PASS` with real exit 0 on
  `./bin/souc run tests/stdlib/theorem/test_smt_*.sio` at default engine
- Next action: forensic dispatch per `docs/audit/` protocol; candidate first
  probe is minimising the import surface of `theorem::smt` until the seed
  materialisation fault isolates
