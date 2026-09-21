<!-- docs:meta
topic_id: repo.docs.audit.cross-engine-runpass-census-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.cross-engine-runpass-census-2026-08-17
-->

# Cross-engine run-pass census (2026-08-17)

**Author:** grok-cli1  
**Branch (local):** `lane/grok-cli1/engine-divergence-census-20260817`  
**Base:** `origin/main` @ `c5754c0c84`  
**Instrument:** `scripts/research/cross_engine_runpass_census.sh`  
**Heavy path:** scrubbed `srun` / `scripts/dev/slurm_srun_minimal.sh` on `cpu-ops`  
**Stage:** `/orangefs/training/sounio/engine-census-20260817/`  
**Engines:** default Madaros (`bin/souc`) vs `SOUNIO_SOUC_ENGINE=lean_single`  
**Binary class:** **prebuilt fleet surface** (not a source-rebuilt Madaros from this tip).

## Why this exists

On 2026-08-17 alone, three new dual-engine divergences landed as PRs/anecdotes:

| ID | Madaros | lean_single | Note |
|---|---|---|---|
| #1798 | accepted forward `inverse_of` | **E158** | **CLOSED** — Madaros aligned |
| #1792 | `var=0.000000` | ~1e-5 | **OPEN** — thesis fabrication |
| #1801 | **E219** on non-allowlisted extern | no E219 surface | Madaros-only refuse |

Plus earlier: **tilde** semantics differ; **E218** V0-A (Madaros refuses f128/f256 arithmetic, lean_single accepts).

CI oracles often pin lean_single. That means the fleet was validating the **non-default** motor while the default disagrees with it in many places. This census turns that from anecdote into a corpus count on `tests/run-pass`.

## Instrument validation (required)

Positive controls **must fire** or the sweep aborts:

1. **AGREE_ACCEPT** — `tests/run-pass/_diag_sobol.sio` (both check OK).
2. **MADAROS_ONLY_REJECT** — `tests/compile-fail/f256_v0b_arithmetic_rejected.sio` (Madaros `error[E218]`, lean accepts).

A zero-disagreement result after failed controls would measure nothing.

## CHECK-phase results (full run-pass, 1759 files)

| Bucket | N | Share of 1759 |
|---|---:|---:|
| **Total** | 1759 | 100% |
| **Agree** (AGREE_ACCEPT + AGREE_REJECT) | 841 | 47.8% |
| **Diverge (explicit)** | 918 | **52.2%** |
| accept-vs-reject | 607 | 34.5% |
| — Madaros-only reject | 230 | 13.1% |
| — lean-only reject | 377 | 21.4% |
| different diagnostic (raw `DIAG_DIFF`) | 15 | 0.9% |
| different diagnostic (strict: both have E-codes) | 6 | 0.3% |
| same diagnostic, different span (raw `SPAN_DIFF`) | 296 | 16.8% |
| timeout | 0 | 0% |

### Class histogram (raw)

```
AGREE_ACCEPT         835
LEAN_ONLY_REJECT     377
SPAN_DIFF            296
MADAROS_ONLY_REJECT  230
DIAG_DIFF             15
AGREE_REJECT           6
```

### Caveat on `SPAN_DIFF`

The first-pass span extractor accepted bare `N:N` pairs from non-diagnostic text (including noise such as `50272:24`). A **strict** recount requiring `.sio:line:col` collapses nearly all raw `SPAN_DIFF` into `AGREE_REJECT` / weak-span both-reject. **Do not cite 296 as true span disagreement.** Prefer:

- accept-vs-reject **607**
- strict dual-coded DIAG_DIFF **6**
- raw DIAG_DIFF **15** (includes one-sided E-code)

### Top Madaros-only error codes (among 230)

| Count | Code | Sketch |
|---:|---|---|
| 91 | NONE | reject without `error[E…]` form |
| 42 | E137 | unresolved name / import surface |
| 13 | E009 | type / generic |
| 13 | E175 | import / multi-module family |
| 10 | E004 | parse / syntax family |
| 10 | E035 | type |
| 7 | E012 | async |
| 7 | E036 | epistemic / measure family |

lean-only rejects: **374/377** with `lean_err=NONE` — lean_single often fails without the Madaros-style `error[E###]` wire form (still a real accept-vs-reject).

### Anecdote coverage inside run-pass

| Anecdote | In run-pass check sweep? |
|---|---|
| E218 f128/f256 | **Yes** — 2 files carry E218 on Madaros (`f128_v0b_literal_smoke`, `f256_v0b_literal_forms` appear as DIAG_DIFF E218 vs E200) |
| E219 | **No** in run-pass (positive control on compile-fail / ffi_posix) |
| E158 / #1798 | **No** residual in run-pass after close on main |
| #1792 var=0 | **Runtime phase** (both must accept at check first) |

## Runtime phase (835 AGREE_ACCEPT files, dual `souc run`)

Naive normalized-stdout hash called **every** file `RUNTIME_DIFF` (835/835). That is an
**instrument failure**, not a scientific claim: Madaros `run` still emits compile banners
(`Compilation successful!`, `/tmp/madaros-run…`) that lean_single does not, and 299 cases are
lean empty-stdout vs Madaros non-empty after the weak normalizer. Reclassified on the TSV notes:

| Runtime reclass | N | Meaning |
|---|---:|---|
| **rc_diff** | **83** | exit codes disagree (hard runtime divergence) |
| rc_same | 752 | same exit code |
| — lean stdout empty, Madaros not | 299 | mostly instrument / print-path noise |
| — both non-empty, hash differs | 453 | mix of banner residue + real numeric drift |

### Science-surface samples (both rc=0, content inspected)

| File | Observation |
|---|---|
| `door1_dense1024_epistemic.sio` | Madaros: uncertainty `0.114585 -> 0.000000`; lean: `-> 2.756059e-17` — **same family as #1792 zero-vs-tiny** |
| `dissertation_pbpk28_degenerate_parity_ref.sio` | trailing PARITY|ct/cavg scientific values differ (0 vs ~3e-7) |
| `dissertation_pbpk_qss_analytical_ref.sio` | similar trailing float drift |
| `beta10_product_cancel_variance.sio` | both print `var_…=0.000000` and PASS (agree on zeros here) |
| `darwin_pbpk28_smoke.sio` | mass/step markers match |

So runtime divergence is **real** (83 rc + multiple numeric content diffs), but the raw 835
hash-diff count is **not** the number to cite. Prefer **rc_diff=83** and case-level science
samples until the normalizer strips Madaros run banners completely.

## How to re-run

```bash
# Login pod: stage is already on OrangeFS after first transfer, or re-tar + srun stdin.
export SLURM_CONF=/tmp/slurm-direct.conf
srun --partition=cpu-ops --nodes=1 --ntasks=1 --cpus-per-task=32 --time=02:00:00 \
  --chdir=/tmp \
  --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,HOME=/tmp \
  /bin/bash -lc '
    cd /orangefs/training/sounio/engine-census-20260817/tree
    export SOUNIO_STDLIB_PATH=$PWD/stdlib
    export CROSS_ENGINE_OUT_DIR=/orangefs/training/sounio/engine-census-20260817/out-check
    export CROSS_ENGINE_JOBS=32 CROSS_ENGINE_TIMEOUT=20
    export CROSS_ENGINE_FILE_LIST=/orangefs/training/sounio/engine-census-20260817/tree/file_list.txt
    ulimit -S -s 524288
    bash scripts/research/cross_engine_runpass_census.sh
  '
```

Helper: `scripts/dev/slurm_srun_minimal.sh` (supported path today; **sbatch not repaired**).

## Related tools

- `scripts/research/cross_engine_diagnostic_agreement.sh` — whole-repo accept/reject only (restored into tree; was not on `main`).
- `scripts/research/cross_engine_runpass_census.sh` — this census (check + optional run, classified).

## Honest claims

1. Dual-engine divergence on run-pass is a **majority event** under this instrument (918/1759 diverge at check).
2. This is **not** a defect rate; it is a disagreement census.
3. CI pinned to lean_single is **not** measuring default Madaros agreement; it measures lean_single.
4. SPAN_DIFF raw counts are **instrument-limited**; use accept-vs-reject + strict DIAG_DIFF for argument.
5. Prebuilt Madaros may lag source; rebuild before claiming source-current parity.

## What this does not close

- ABI fix for #1792 fabrication.
- Making CI’s claim oracle dual-engine or Madaros-default.
- Full source-current Madaros rebuild gate before the census.
