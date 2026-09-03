<!-- docs:meta
topic_id: repo.docs.audit.e011-stale-prebuilt-phantom-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: glm-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e011-stale-prebuilt-phantom-2026-08-18
-->

# E011 target evaporation — the stale prebuilt as a lying instrument (2026-08-18)

**Lane:** glm-cli1 / `pbpk28-clinical-callsites-20260818`
**Dispatch:** the "E011 family" in the dissertation suite's twelve toolchain
failures, in `rapamycin_kaxi_fuse_prior` and `pbpk28_rapamycin_clinical`.
**Outcome:** the E011 family no longer existed at current main. What the
dispatch table described was the output of a compiler binary predating the
fix that closed it. This is the fourth time on 2026-08-18 that a yesterday's
table described a repository that no longer existed — and the first where the
table was honest about the past and the **instrument** lied about the present.

## Instrument validation (done before believing any error table)

- Tracked prebuilt `bin/madaros-linux-x86_64` last refreshed at `3d1f143e7a`,
  **2026-08-17 07:09:56Z**.
- grok-cli5's #1820 ("fix(check): Madaros Seq surface") merged
  **2026-08-18 00:44Z**, changing `self-hosted/check/{check,compat,types}.sio`
  — checker source — with **no binary refresh** in the PR.
- Controls on the #1820 witness (`tests/run-pass/madaros_seq_minimal_witness.sio`):
  - prebuilt: rc=1 — 5×E011, 3×E013, 3×E137
  - post-#1820 build (`/workspace/.wt/grok-cli5/artifacts/self-hosted/madaros-seq`,
  the lane's own fresh artifact): rc=0, zero diagnostics

Any error table produced with `./bin/souc` (default Madaros engine) between
07:09 yesterday and the next prebuilt refresh therefore describes a compiler
that no longer matches the source. #1820's own audit recorded the same
control ("Prebuilt `bin/souc` kaxi (control) — E137 `seq_new` still").

## Re-measurement of the dispatched targets (main `dde4b0b0d4`)

| file | prebuilt (stale) | post-#1820 Madaros | suite engine (lean_single shim) |
|---|---|---|---|
| `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` | rc=1: 5×E011, 3×E137, 1×E013 | **rc=0** | **run rc=0, `PASS`, `sd_post == sd_expected`** |
| `stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio` | rc=1: 48×E011, 34×E004, 34×E137, 10×E013, 8×E009, 5×E175, E001, E010 | rc=1: E011 **0**, E013 **0**, E137 **1**, E004 32, E009 3, E175 5, E001 1 | rc=1 at 5 call sites → fixed in #1868 → **run rc=0, emits `_CLINICAL_PENDING_OBSERVED`** |

Two consequences:

1. **`rapamycin_kaxi_fuse_prior` owes no work.** It is green under the
   engine the suite actually runs and under a current-source Madaros. Its
   E011s exist only under the stale prebuilt. It leaves the failure list
   without anyone touching the file.
2. The clinical file's real residue was never E011: under the suite engine
   it was five call-site defects (fixed, #1868); under current-source
   Madaros it is the owned residue below.

## Where the "E011 family" table came from

`43bc560bf8` (2026-08-03, "unblock pbpk28_rapamycin_clinical's parse — and
publish what it hid") published the census behind the parse wall:
E011 48, E137 34, E004 34, E013 10, E009 8, E175 5, E010 1, E001 1.
#1820 closed the Seq-surface share of that census at source
(E011 48→0, E013 10→0, E137 34→1 under a current-source build). The counts
that did **not** move are the real outstanding residue, and each has an owner:

| residue | owner |
|---|---|
| E175 ×5 — `error_bar_entry` / `error_bar_chart` private in `stdlib/plot/epistemic.sio` | E175 lane (active sweep) |
| E004 ×32 — f64/i64 operators inside `stdlib/chemistry/ontology.sio` (lean_single coerces; Madaros does not) | Madaros Seq/compat follow-up (grok-cli5) |
| E009 ×3, E001 ×1 — remaining call-typing in the clinical import chain | rides with the above |

## Fleet rules this event adds

- **The prebuilt has a build date. Check it against the newest
  `self-hosted/` commit before believing any error table measured with
  `./bin/souc`:**
  `git log -1 --format=%ad -- bin/madaros-linux-x86_64` vs
  `git log -1 --format=%ad -- self-hosted/`. Prebuilt older = phantoms
  possible; re-measure with a current-source build or the suite's engine
  before dispatching work on the findings.
- **A red measured with the default engine is not a statement about the
  suite.** `dissertation_pbpk_suite_gate.sh` pins
  `SOUC_BIN=scripts/ci/souc-seq-leansingle.sh` (Seq-capable lean_single)
  by default; CI may override `SOUC_BIN`. State which engine produced a
  failure count.
- **Prebuilt refresh is owed** (`build(madaros): refresh prebuilt`, last
  done `3d1f143e7a`): until it lands, every default-engine measurement
  repeats this phantom. Heavy build — CI or Slurm, not the pod.

## Receipts

- Witness controls: prebuilt rc=1 vs `madaros-seq` rc=0 (this session,
  first-hand; both logs quoted in PR #1868's description).
- kaxi member acceptance: `timeout 90 scripts/ci/souc-seq-leansingle.sh run
  tests/run-pass/rapamycin_kaxi_fuse_prior.sio` → rc=0,
  `rapamycin_kaxi_fuse_prior cl_post=10.982609 sd_post=1.641761
  sd_expected=1.641761` + `PASS`.
- Clinical member acceptance (post-fix): run rc=0, final line
  `PBPK28_RAPAMYCIN_CHEBI_9168_GO_6805_CLINICAL_PENDING_OBSERVED`.
- Related: `docs/audit/MADAROS_SEQ_SURFACE_2026-08-17.md` (#1820's own
  audit and its prebuilt-control note); PR #1868 (the five call sites +
  iterated walls); PR #1855 (same day, same class: a bytewise gate that
  could never fail being mistaken for a numerical failure).
