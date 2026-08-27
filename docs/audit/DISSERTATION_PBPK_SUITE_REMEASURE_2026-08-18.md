<!-- docs:meta
topic_id: repo.docs.audit.dissertation-pbpk-suite-remeasure-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: kimi-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-pbpk-suite-remeasure-2026-08-18
-->

# Dissertation PBPK suite re-measure — 16 FAIL / 53 PASS on current main (2026-08-18)

**Lane:** kimi-cli1 / `pbpk-remeasure-post1820`
**Commit measured:** `c240e848bf` (current `origin/main` at 18:27Z, post-#1820/#1868/#1871)
**Engine:** source-built Madaros (`build_modular_madaros.sh`, 100084302 bytes, built in a
clean worktree off `c240e848bf`; NOT the stale prebuilt — see `E011_STALE_PREBUILT_PHANTOM_2026-08-18.md`)
**Supersedes:** the 19-FAIL photograph of `DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md`
(measured at `d0c798e4ed`, before #1820/#1868/#1871 landed).

## Headline

| | 2026-08-17 photograph | this re-measure |
|---|---|---|
| FAIL | 19 | **16** |
| PASS | 33 | 37 |
| Science/model defects | 0 | **0** (unchanged) |

## Instrument validation (before believing any number)

- **Positive control:** `#1820` witness `tests/run-pass/madaros_seq_minimal_witness.sio` → rc=0 under the measured binary, before the suite.
- **Stack trap, caught before it lied:** the pod default stack (8 MB) segfaults Madaros `run` on EVERY test — a first run reported **53/53 FAIL rc=139**. Verified both directions on a single test: `ulimit -s 8192` → rc=139, `ulimit -s 1048576` → rc=0. All numbers below are from the re-run with the CI stack environment (`ulimit -s 1048576`, `MADAROS_STACK_KB=524288`, exactly what `.github/workflows/ci.yml` injects). This is the constraints rule "gates invoking Madaros must raise the soft stack" biting in the wild — a bare local gate run reproduces CI green only with the CI stack.
- Build method note: `build_modular_madaros.sh` self-locks; wrapping it in `souc-build-lock.sh` deadlocks (flock is not reentrant). Two launcher footguns were hit and removed in getting a clean run; recorded here so the next lane skips them.

## The 16 failures, by family

### rc=182 resource ceiling — 7 (unchanged, diagnosis closed, 5 lanes)

`d2_gum`, `d2_voi`, `gum_vs_mc`, `pbpk28_mc_cross_validation`, `pbpk28_mc_prior_family_sweep`, `rapamycin_clinical`, `rapamycin_pop_sim` — reproduces the closed diagnosis; not re-litigated here.

### Toolchain with active owners — 8

`pbpk28_sobol_pce` (E009 + E035 visible in its log — codex-2's active lane), `pbpk28_rapamycin_clinical` (E175/E004 residue, per #1871's table), `dissertation_oral_pd`, `dissertation_pgx_demo`, `dissertation_scenario_gate`, `dissertation_steady_state`, `dissertation_steady_state_fullvd`, `rapamycin_epistemic_adaptive`.

### NEW FINDING — kaxi segfaults under stdout redirection (1)

`rapamycin_kaxi_fuse_prior`: **rc=139 (SIGSEGV) deterministically when stdout is a file or `/dev/null`; rc=0 when stdout is a pipe with a live reader.** 10/10 reproduced each way, under the full 1 GB stack, so this is NOT the stack trap. `check` mode is green under both.

This refines #1871's verdict ("`rapamycin_kaxi_fuse_prior` owes no work… green under the engine the suite actually runs"): it is green interactively and under a pipe, but the suite gate — and CI — redirect stdout to a file, and under that condition it segfaults every time. The verdict "no E011 work owed" stands (E011 count is zero in `check`); what is owed is a runtime IO/buffering defect: something in the Madaros runtime write path dereferences badly only when stdout is a regular file. Candidate surface: full-buffering (file) vs line-buffering (tty/pipe) — i.e. the flush path at volume.

Reproduction (deterministic):

```bash
bash -c 'ulimit -s 1048576; /tmp/madaros-measure run tests/run-pass/rapamycin_kaxi_fuse_prior.sio > /dev/null 2>&1; echo $?'   # 139
bash -c 'ulimit -s 1048576; /tmp/madaros-measure run tests/run-pass/rapamycin_kaxi_fuse_prior.sio 2>&1 | tail -1; echo ${PIPESTATUS[0]:-$?}'  # 0
```

Owned dispatch: Madaros runtime lane (whoever holds self-hosted runtime IO). Not a science defect.

## Where the delta 19 → 16 came from

- −2: the E011 family (kaxi-E011, clinical-E011) — closed at source by #1820; phantom documented by #1871.
- −1: clinical's five call-site defects — fixed by #1868.
- +1: kaxi now appears again as the NEW stdout-segfault above (it was in the old list as E011, left as a different defect).
- 7 rc=182 unchanged; remaining toolchain cases shifted within the owned lanes' totals.

## Fleet rules this event adds

- The photograph number (19) circulated for a full day after its commits were superseded. A suite count is a property of (commit, engine, stack env). Any re-circulation should carry the triple.
- `#1871`'s "run rc=0" was true and still misleading: stdout shape is part of the instrument. Gates redirect to files; spot-checks through pipes. Both are "running it" — they are not the same measurement.

## Document control

| Date | Change |
|---|---|
| 2026-08-18 | Re-measure on `c240e848bf`, source-built Madaros, CI stack env; 16/53; kaxi stdout-segfault isolated. |
