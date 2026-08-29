<!-- docs:meta
topic_id: repo.docs.audit.known-failure-xpas-signal-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.known-failure-xpas-signal-2026-08-18
-->

# Known-failure that passes is a signal

**Date:** 2026-08-18  
**Lane:** `lane/grok-cli3/known-failure-xpas-20260818`  
**Question:** after #1890 dropped 240 stale 139 tags, how does the next 240 get detected without another census?

## Two engines, two questions

The same table as #1890. Cite it when this comes back:

> The Madaros decides whether a 139 (or any Madaros-named) tag is true. lean_single decides whether the file needs `requires: madaros`. Zeros on lean_single do not license dropping a Madaros tag.

## What the harness already did

Suite-visible `//@ known-failure` files are **run**, not skipped. The harness already classifies a pass as `xpas`.

Before this change, `xpas` was counted as **Pass**, printed only under `--verbose`, and the suite printed `All tests passed!` and exited 0. `vxpas` (vacuous-annotation baseline) had already been taught the opposite lesson — always announce — because a stale entry that silently absorbs a pass is how a baseline rots. Known-failure `xpas` never got that treatment.

`tests/known_failures/` is **not** scanned (20 files). Those cannot silently XPAS in the suite; they also cannot be caught by this mechanism.

## Cost (criterion written first; N = suite-visible set)

Instrument: this worktree `bin/souc`, `souc run` via the harness, never `souc <file> -o`. Madaros = `artifacts/self-hosted/madaros` (2026-08-17 17:01), not an E230-patched source build. Controls: `hello` rc=0 on both engines.

On `origin/main` at this measurement, N = **47**.

| Engine | wall | XPAS | XFAIL | SKIP | harness rc |
|---|--:|--:|--:|--:|---|
| lean_single (Full Test Suite) | **5 s** | 8 | 32 | 7 | 0 (`All tests passed!`) |
| default Madaros | **61 s** | 22 | 24 | 1 | 0 (`All tests passed!`) |

That is gate-sized, not quarterly archaeology. The 239 #1890 files were already inside the suite cost; the suite just swallowed the signal.

## Cross-tab (lean × Madaros)

| lean | Madaros | n | meaning |
|---|---|--:|---|
| XPAS | XPAS | 3 | tag stale on both — drop |
| XPAS | XFAIL | 5 | seed pass, Madaros still fails — `requires: madaros` |
| XFAIL | XPAS | 19 | mostly lean_single-named f128/f256 reservations; Madaros already rejects. Tag is about the seed |
| XFAIL | XFAIL | 13 | still a real failure on both |
| SKIP | XFAIL | 6 | already `requires: madaros` |
| SKIP | SKIP | 1 | `sret_forwarding_tuple_aggregate.sio` has no run-pass annotation |

## Mechanism

1. **Listen to the run we already pay for.** Harness always prints XPAS, counts it separately from Pass, refuses `All tests passed!` when XPAS > 0, and records it as a JUnit failure. `SOUNIO_XPAS_FATAL=1` makes the job exit 1.
2. **Recheck Madaros-named tags on every f64 job**, not only when the test file is in the diff. `scripts/ci/known_failure_madaros_recheck.sh` (wired from `madaros_changed_tests_gate.sh`) runs every suite-visible `requires: madaros` + `known-failure` file with `SOUNIO_XPAS_FATAL=1`.
3. Classify the seed XPASses so Full Suite stays honest:
   - dropped (both pass): `imported_f64_return`, `rng_fused_global_read_single_eval`, `test_smt_solver_basic`, `mismatch_let`
   - `requires: madaros`: `print_f64_large_magnitude`, `lorenz_i256_product_smoke`
   - **not touched** (other lane owns the file or the LoRA snapshot): `gum_fo_across_call.sio`, `turbofish_concrete_type_mismatch.sio`

Default `SOUNIO_XPAS_FATAL` stays off on Full Test Suite until those two owners classify their files. The announcement is on; the fail switch is on for the Madaros recheck.

## Not done

- `tests/known_failures/` still not in the suite.
- f128/f256 compile-fail tags left in place (they name a lean_single gap; Madaros XPAS is expected).
- CAP / token table / handle table / global 30 s / E230 not touched.
- This lane does not merge.
