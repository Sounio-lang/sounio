<!-- docs:meta
topic_id: repo.docs.audit.ci-engine-per-job-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: glm-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ci-engine-per-job-census-2026-08-19
-->

# Which engine does CI actually test? Per-job engine census (2026-08-19)

**Lane:** glm-cli1 / `ci-engine-census-20260819`
**Origin:** PR #1964 established that the CI Full Test Suite runs `souc-stage2`, which is **lean_single**, not Madaros — so its 21 XPASS were seed results, and a source-built Madaros leaves the FO arity branch fully open (ADD3/ADD4 = 0.000000, IMP_ADD3 = 0; only IMP_ADD2 = 5 holds, as #1939 claimed). This census measures the engine behind **every** CI job, because that fact decides what "the suite passes" means.
**Head measured:** `11dd0d0f40` (origin/main). Workflow: `.github/workflows/ci.yml` (821 lines). Log evidence: run `32203259100` (completed `success`, all 12 jobs, 2026-08-19).

## How engines resolve (mechanics, from the checked-in wrapper)

`bin/souc` resolution order: `MADAROS_RAW_BIN` → `SOUNIO_MADAROS_BIN` → `artifacts/self-hosted/madaros` → `bin/madaros-linux-x86_64`; if none exists it **falls back to lean_single with a stderr notice**. In a CI checkout `artifacts/self-hosted/madaros` never exists and `bin/madaros-linux-x86_64` is git-tracked — so an unpinned `bin/souc` in CI resolves to the **tracked prebuilt Madaros**, whose last refresh (`3d1f143e7a`, 2026-08-17) predates later compiler-source changes (the same staleness documented in `docs/audit/E011_STALE_PREBUILT_PHANTOM_2026-08-18.md`). `souc-stage2` is a different object entirely: `scripts/ci/selfhost_host_gate.sh:186` produces it by compiling `self-hosted/compiler/lean_single.sio` with the stage-1 host binary — the **lean_single seed lineage**.

## The fallback crux — direct evidence

Grep for the wrapper's fallback notice ("falling back to the" / "Madaros raw ELF not found") across **all 12 job logs** of green run `32203259100`: **0 hits in every job.** No CI job silently falls back. The engine story in CI is not silent degradation — it is deliberate, per-job pinning of three different populations.

## Per-job table

| job | binary resolved | engine | how determined |
|---|---|---|---|
| Impact | none (path classifier) | neither | job definition |
| Contracts | no compiler for most steps; the two dissertation steps (and honesty gates that execute Sounio) pin `SOUNIO_GATE_SOUC`/`SOUC_BIN` = `bin/souc` → resolves to **tracked prebuilt** `bin/madaros-linux-x86_64`; several steps deliberately run lean_single references | **stale prebuilt Madaros** (+ deliberate lean refs) | ci.yml env blocks + the authors' own comment ("Neither script builds Madaros from source; both resolve the checked-in souc wrapper"); wrapper resolution order; zero fallback notices |
| Native Self-Host (Linux x86_64) | in-job `artifacts/souc-stage2` | **lean_single** | `selfhost_host_gate.sh:186` (stage2 = compiled `lean_single.sio`); job YAML |
| Source-Bootstrap Self-Host (Linux) | in-job `souc-stage2`; cross-builds the arm64 artifact by compiling `lean_single.sio` with it | **lean_single** | job YAML steps |
| Madaros Current-Source f64 Lowering | in-job `/tmp/.../madaros` built from the PR's own source; `MADAROS_BIN` pinned | **Madaros, current source** | job YAML: build + pins |
| Native Self-Host (macOS arm64) | downloaded artifact = `lean_single.sio` cross-compiled by `souc-stage2` | **lean_single** | source-bootstrap job's cross-build step |
| **Full Test Suite** | downloaded `/tmp/souc-stage2` (`SOUNIO_TEST_SOUC_BIN`) | **lean_single** | ci.yml; run log: 3× `souc-stage2` references, **0** `Madaros v0` banners |
| Madaros Witness Gate | in-job `/tmp/madaros-ci.elf` built from the PR's own source; `MADAROS_RAW_BIN` pinned on every step | **Madaros, current source** | job YAML: `build_modular_madaros.sh` + pins |
| Sounio Lint | none (syntax) | neither | job definition |
| Lean Proofs | Lean toolchain (`lake`) | neither | job definition |
| Website | node | neither | job definition |
| CI Decision | aggregate gate | neither | job definition |

## Count

Of the 12 jobs: **2 exercise Madaros built from current source** (Madaros Witness Gate, Madaros Current-Source f64 Lowering), **4 exercise lean_single** (Full Test Suite, both Linux self-host jobs, macOS arm64), **1 exercises a stale tracked prebuilt Madaros** in its compiler-touching steps (Contracts; a third population — neither current source nor seed), **5 exercise no Sounio engine** (Impact, Sounio Lint, Lean Proofs, Website, CI Decision).

**The ratio is the finding:** the job named "Full Test Suite" — the one whose green most reads as "the compiler passes the tests" — tests the seed. Madaros coverage in CI is two dedicated jobs; nothing called a "suite" runs against current-source Madaros. An unknown share of this repository's green is therefore statements about lean_single, including the 21 XPASS of #1964 and the FO arity branch measured open on source-built Madaros.

## Controls (mandatory, both directions)

- **Certain Madaros (current source): Madaros Witness Gate.** The job itself runs `bash scripts/ci/build_modular_madaros.sh /tmp/madaros-ci.elf` — building Madaros from the PR's own checked-out source — and pins `MADAROS_RAW_BIN: /tmp/madaros-ci.elf` on every subsequent step (ci.yml, `madaros-witness-gate`). There is no path by which those steps reach lean_single or the prebuilt.
- **Certain lean_single: Full Test Suite.** `SOUNIO_TEST_SOUC_BIN: /tmp/souc-stage2` in ci.yml; the artifact's provenance is `selfhost_host_gate.sh:186` (`lean_single.sio` compiled by the stage-1 host binary); the run's own log references `souc-stage2` three times and contains zero Madaros banners.

## `_reject_science_without_madaros` — what it guards and what it does not

The wrapper rejects `--science-boundary` / `--claim-contract` / pkg-verify surfaces when the engine is a raw-ELF override or the lean_single fallback — it guards the **science-boundary emission surface**, requiring the Madaros launcher for it. It does **not** distinguish current-source Madaros from the stale tracked prebuilt: a science-flagged call in CI would pass the guard while running a compiler that predates current source. The test suite invokes no science flags at all, so it bypasses the guard trivially — by never touching the guarded surface. The guard constrains engine *family*, not engine *currency*; it was never an engine-identity check for the suite.

## Receipts

- Run `32203259100` (completed `success`): per-job fallback-notice grep = 0 across all 12 jobs; Full Test Suite log: 3× `souc-stage2`, 0× `Madaros v0`; Contracts log shows deliberate lean_single references (`[epistemic-fab] PASS: F1 lean_single reference…`, `[f64-bitcast-boundary] … engine=lean_single`, the canonical fixed-point check of `bin/souc-lean-single-x86_64`, ontology-smoke bootstraps) and the dissertation steps' own PASS lines.
- ci.yml: `full-test-suite` (lines 637-664), `madaros-witness-gate` (665-703, build + pins), dissertation steps with the authors' comment at lines 100-113, `native-selfhost-linux-x86_64` (391-426, artifact upload of `souc-stage2`), `selfhost_host_gate.sh:186,191` (stage2/stage3 = `lean_single.sio`).
- Cross-references: PR #1964 (21 XPASS are lean_single; source-built Madaros ADD3/ADD4 = 0), `docs/audit/E011_STALE_PREBUILT_PHANTOM_2026-08-18.md` (the same tracked prebuilt emitting phantom error families vs current source).

## Semantic declaration

Measurement only: no workflow, script, or source file was changed. Every cell in the table above is grounded in the cited ci.yml lines, script lines, or run-log greps from a completed successful run.

**Claims-Forbidden:** this document does **not** establish which engine *should* run each job — only which engine each job *does* run, and by what mechanism. Whether the Full Test Suite should run current-source Madaros, the seed, or both is a policy decision for the maintainers; nothing here argues it either way.
