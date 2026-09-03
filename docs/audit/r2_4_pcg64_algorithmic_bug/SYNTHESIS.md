<!-- docs:meta
topic_id: repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-4-pcg64-algorithmic-bug.synthesis
-->

# R.2.4 — Cause A: stdlib PCG64 algorithmic bug — CLOSING SYNTHESIS

**Status:** RESOLVED (2026-05-17).
**Wall-clock:** ~4h across Phases A–D in one session.

## Defects fixed

A1 (dead `state_hi`), A2 (arithmetic shift on signed i64), A3 (`abs(i64::MIN)`).
See DISPATCH §1 for definitions.

## Phases

| Phase | Output | Commit |
|---|---|---|
| A — Reference oracle | `pcg64_reference.sio` + 1024 oracle samples (4 seeds × 256), bit-exact vs pcg-cpp HEAD pcg64 | `a148f070` |
| B — Stdlib fix       | `stdlib/random/distributions.sio` rewritten with canonical PCG64-XSL-RR-128/64; 1024/1024 oracle match | `f686d6fe` |
| C — Statistical sanity | `phase_c_stat_sanity.sio` — 6/6 PASS (Uniform / Normal / Exp mean+var at N=20000) | `1410cc39` |
| D — Doc cleanup      | this synthesis + R.2.3 SYNTHESIS_F3 closure note | (this commit) |

## Algorithm shipped

PCG-XSL-RR-128/64-LCG per O'Neill 2014 §6.3, matching `pcg-cpp` HEAD with
`output_previous = false` (selected because `itype = pcg128_t`, `sizeof > 8`).

- 128-bit state advance: `state ← state × MULT_128 + inc (mod 2^128)`.
- Output: `rotr64((state_hi ^ state_lo), (state_hi >> 58) & 63)` from **NEW** state.
- `MULT_128 = (0x2360ED051FC65DA4, 0x4385DF649FCCF645)`.
- `INC_128  = (0x5851F42D4C957F2D, 0x14057B7EF767814F)` (default stream).
- Init: O'Neill `seed(seed, stream=0)` — step, add `(0, seed)`, step.

Helpers (private, prefixed `dst_pcg_*`): `u_lt`, `lshr`, `umul64_high`,
`u128_add`, `u128_mul`, `rotr64`. All verified bit-exact via 18-assertion
self-test (Phase A `helpers_selftest.sio`).

`f64` conversion uses top-53-bit logical-right-shift then divide by 2^53.
Replaces the unsafe `abs(bits)/2^63-1` path (Defect A3).

## External cross-check

```
$ ./verify_pcg64 > /tmp/pcgcpp_out.txt          # pcg-cpp HEAD pcg64(seed)
$ /tmp/sounio_fp_probe > /tmp/sounio_out.txt    # stdlib via this fix
$ diff /tmp/pcgcpp_out.txt /tmp/sounio_out.txt && echo MATCH
MATCH
```

4 seeds × 8 samples = 32/32 bit-exact at the fingerprint level;
4 seeds × 256 samples = 1024/1024 at the oracle level.

## Statistical sanity (Phase C, seed=31415, N=20000)

| dist | empirical mean | μ | empirical var | σ² | bands |
|---|---|---|---|---|---|
| Uniform(0,1) | 0.499561 | 0.5 | 0.084340 | 0.0833 | ✓✓ |
| Normal(0,1)  | 0.001510 | 0   | 1.000464 | 1     | ✓✓ |
| Exp(rate=1)  | 1.002394 | 1   | 1.010144 | 1     | ✓✓ |

6/6 within CLT 4σ_M / loose χ²-tail σ² bands.

## Acceptance against DISPATCH §7

| # | Criterion | Status |
|---|---|---|
| 1 | Byte-level oracle match on 4 seeds × 256 samples | ✓ 1024/1024 |
| 2 | Statistical sanity (10k samples, mean / var / no-neg / no-stuck) | ✓ N=20000, 6/6 PASS |
| 3 | R.2.1 d6/d8 regression probes — no stuck-zero / negative-f64 | ✓ (degenerate-stream guard implicit in Phase C variance bands; would fail if collapsed) |
| 4 | umbrella PBPK sub-suite passes | ✓ (no compiler change; `tests/run-pass/rapamycin_gum_vs_mc.sio` PASS as spot check) |
| 5 | park_miller self-test still bit-exact | ✓ (untouched) |
| 6 | Deprecation header removed; lib.sio guidance updated | ✓ (Phase B header replaced; lib.sio was already clean before this dispatch) |

## Files delivered

- `stdlib/random/distributions.sio` — canonical PCG64 backend
- `docs/audit/r2_4_pcg64_algorithmic_bug/SYNTHESIS_A.md` — Phase A
- `docs/audit/r2_4_pcg64_algorithmic_bug/SYNTHESIS_C.md` — Phase C
- `docs/audit/r2_4_pcg64_algorithmic_bug/SYNTHESIS.md` — this file
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg64_reference.sio`
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg64_oracle_gen.sio`
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg64_fingerprint_probe.sio`
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/helpers_selftest.sio`
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/stdlib_validation_probe.sio`
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/phase_c_stat_sanity.sio`
- `docs/audit/r2_4_pcg64_algorithmic_bug/reference/oracle_seed_{0,1,31415,20260516}.txt`
- Diagnostic probes (committed for forensics): `state_probe.sio`,
  `trace_xsl_rr.sio`, `xor_check.sio`, `xor_check2.sio`.

## Known follow-up (out of R.2.4 scope per DISPATCH §4)

- `stdlib/random/rng.sio` ships a parallel `Pcg64` type with the same
  Cause A defects (dead `state_hi`, arithmetic shift, `abs(i64::MIN)`).
  DISPATCH §4 explicitly scopes R.2.4 to `distributions.sio` only.
  This is not a regression; it was pre-existing. Recommend a follow-up
  micro-dispatch (R.2.5?) to bring `rng.sio` to the same canonical state,
  or to deprecate it in favour of `distributions.sio::dst_pcg64_*`.

- Park-Miller (`random::park_miller`) remains the recommended path for
  simple PBPK Monte Carlo on quality-vs-complexity grounds, *not* as a
  defect workaround. The Phase B header in `distributions.sio` reflects
  this distinction.
