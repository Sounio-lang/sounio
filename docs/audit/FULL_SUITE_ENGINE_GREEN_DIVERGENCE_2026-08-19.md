<!-- docs:meta
topic_id: repo.docs.audit.full-suite-engine-green-divergence-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: codex-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.full-suite-engine-green-divergence-2026-08-19
-->

# Full Test Suite engine-green divergence

## Answer

On `origin/main@11dd0d0f4074c7125fd06a272848f45cfe7b9cdb`, **437 of the
1,527 tests counted as PASS by the CI engine fail under source-built Madaros**.
That is **28.6% of the lean_single green set**.

This is the answer to "how many greens are about the wrong engine" under the
strict definition used by the suite itself: `pass` and `vxpas` are green;
`fail`, `xfail`, `vxfail`, and `xpas` are not. The 1,326 tests skipped by both
engines are outside the comparison.

| lean_single | Madaros | tests |
|---|---|---:|
| green | green | 1,090 |
| **green** | **not green** | **437** |
| not green | green | 27 |
| not green | not green | 132 |
| skipped by both | skipped by both | 1,326 |

The inverse is larger when the question is execution outcome rather than suite
colour: **55** tests fail underneath an `xfail`/`vxfail` on lean_single but
execute successfully on Madaros (`pass`, `vxpas`, or `xpas`). The strict table
shows 27 because an `xpas` is deliberately not green in this harness. There are
zero literal `fail` results on lean_single in this run; its non-green results
are 89 `xfail`, 49 `vxfail`, and 21 `xpas`.

## Semantic declaration

This is a measurement-only audit. It changes no Concept-ID, type, effect,
ontology, compiler behaviour, scientific interpretation, test annotation,
workflow, or acceptance criterion.

## Instrument

The measured command is the command in `.github/workflows/ci.yml`:

```sh
SOUNIO_TEST_SOUC_BIN=<engine-elf> \
SOUNIO_TEST_JOBS=4 \
bash scripts/run_sio_test_suite.sh --format junit --jobs 4
```

Both arms used the same archived source tree, `CI=1`, stack reservation,
environment, harness, 3,012-file selection, four workers, annotations, known
failure manifest, and vacuous-annotation baseline. Per-test JSON was preserved
under a directory with one writer. No test, tag, baseline, or workflow was
changed.

The harness selected 3,012 files. It emitted executable verdict JSON for 1,686
files and skipped 1,326 before execution on both arms. Both arms emitted the
same 1,686 `relfile` keys; set equality was exact.

### Engine receipts

| field | lean_single / CI engine | canonical Madaros |
|---|---|---|
| construction | `souc-host` compiled current `lean_single.sio` to `souc-stage2` | `scripts/ci/build_modular_madaros.sh` from the same source tree |
| ELF SHA-256 | `87ebb6ecb4dc03b173cb5ae83e0633a0acff2e8624740917113d554ae83f3105` | `6bafb5bf4e4cd55b1535071f08210eeb820ca74948e60d42213131b8872c2b65` |
| ELF bytes | 2,552,029 | 100,561,123 |
| suite job / host | 10370 / `cpuops-t560-proxmox` | 10370 / `cpuops-t560-proxmox` |
| suite duration | 458 s | 421 s |
| suite rc | 0 | 1 |

The source-built Madaros build was performed on Slurm and
`build_modular_madaros.sh` was not wrapped in `souc-build-lock.sh`; it acquired
its own lock. The isolated reproducibility build was Slurm job 10373 on
`gpuorangefs-5860-proxmox`: rc=0, 277 s, 100,561,123 bytes, and SHA-256
`6bafb5bf4e4cd55b1535071f08210eeb820ca74948e60d42213131b8872c2b65`.
It is byte-identical to the Madaros used by job 10370.

The current lean self-host chain has an independent fixed-point warning:
`souc-stage2` and `souc-stage3` were both valid ELF outputs, but executable
payload comparison returned rc=1. This does not change which binary the Full
Test Suite consumes (`souc-stage2`), but it means this snapshot would not pass
the upstream native-selfhost job and must not be described as an all-green
current CI run.

## Aggregate outcomes

| harness status | lean_single | Madaros |
|---|---:|---:|
| PASS (including any `vxpas`) | 1,527 | 1,117 |
| literal FAIL | 0 | 437 |
| XFAIL | 89 | 73 |
| XPASS | 21 | 37 |
| VXFAIL | 49 | 22 |
| SKIP | 1,326 | 1,326 |

The raw execution view treats `xpas` as an underlying pass and
`xfail`/`vxfail` as an underlying failure. Under that view the four buckets are
1,099 pass/pass, 449 lean-pass/Madaros-fail, 55 lean-fail/Madaros-pass, and 83
fail/fail. These are useful for annotation repair, but **449 is not the answer
to the founder's green-CI question** because it includes lean `xpas` results
that intentionally make the suite non-green.

## Controls

**Agreement control:** `tests/run-pass/_diag_sobol.sio` is `pass` on both
engines (lean 0 s; Madaros 1 s).

**Divergence control:** `tests/run-pass/gum_fo_across_call.sio` is `xpas` on
lean_single and `xfail` on Madaros (`run exited 1`). The annotation requires
`GUM_FO_ACROSS_CALL_OK`; lean_single produces it while Madaros loses the
variance across the call. This control therefore proves that the two arms did
not accidentally resolve to the same engine. The separate ADD3 fixture is
annotated `requires: madaros`, so the CI-matched corpus skips it on both arms
and it is not used to inflate the difference set.

## First-set structure and sample

The 437 lean-green/Madaros-fail tests comprise:

| surface | tests |
|---|---:|
| `tests/run-pass` | 191 |
| `tests/compile-fail` | 103 |
| `tests/ui` | 11 |
| `tests/stdlib` | 132 |

The Madaros harness failure summaries are led by 269 `run exited 1`, 38
`expected compile failure but passed`, 19 `check exited 1`, 17 ontology
diagnostic mismatches, 16 illegal-instruction exits (132), 15 segmentation
faults (139), 11 refinement-diagnostic mismatches, and three handle-ceiling
exits (182). A Madaros failure is therefore not one mechanism.

A deterministic 25-test stratified sample took the first five lexicographic
members from each of five observed failure shapes: accepted negative,
diagnostic mismatch, check rejection, runtime exit 1, and runtime crash. Direct
Madaros reruns were captured separately from the suite.

| classification | sampled | evidence boundary |
|---|---:|---|
| harness or lean-specific diagnostic contract | 10 | Five negative tests changed verdict between the harness and a direct rerun; five more were rejected by Madaros with a named but different diagnostic (for example E035 instead of expected E067, E036 instead of the expected Temporal sentence, and E176 instead of `private struct literal`). |
| unsupported or divergent frontend surface | 10 | Run-pass samples stopped at parse, resolution, type, or effect checking: graphics module parse failure, E137, E015, E004, E008, and E035. Calling all ten compiler defects would require deciding the intended Madaros semantics for each source. |
| compiler/runtime defect or instability | 5 | Five closure tests were 139 in the suite. Immediate direct reruns no longer crashed but stopped after `lower_array: seed_begin` with an unnumbered `typecheck: failed`; the instability is itself a defect, but the sample does not identify one root cause. |

Thus the 25-test sample classifies 25/437 by a stated, reproducible rule. It
does **not** extrapolate those proportions to the remaining 412 tests.

## Rejected attempts

Two attempt paths are excluded from the result:

1. The compute node lacked the `file` utility, so `selfhost_host_gate.sh`
   stopped at its format attestation after producing `souc-host`. Stage2 and
   stage3 were then built with the same commands and validated by ELF magic.
2. Interrupting the first `kubectl exec` did not cancel its Slurm allocation;
   it later collided with a second run's output directory. Jobs 10361 and
   10368 and all of their suite JSON were discarded. Job 10370 used fresh
   `out-run2` paths and was the sole writer.

An initial 25-test log-capture list was also rejected because a TTY introduced
CRLF and a misquoted cleanup removed the letter `r` from paths. Those rc values
were not used. The clean list was retransferred without TTY and rerun.

## Claims forbidden

- This measurement does not establish which engine **should** gate the
  repository.
- A Madaros failure is not automatically a compiler defect; classification
  distinguishes defects, unsupported features, harness differences, and
  lean-specific test assumptions.
- The 437 count does not include skipped tests, lean XPASS tests, or Madaros
  failures outside the lean-green set.
- This snapshot is not an all-green CI claim because the source-built
  lean_single fixed-point comparison failed before the Full Test Suite stage.
- The sample classification is not a classification of all 437 cases.

## Retained evidence

OrangeFS root:
`/orangefs/training/sounio/engine-green-divergence-11dd0d0f4074-20260819`

The retained root contains source staging receipt, build logs, ELF hashes and
sizes, job 10370 suite logs, JUnit files, all per-test JSON for both engines,
the clean sample list, sample logs, and rejected-attempt logs.
