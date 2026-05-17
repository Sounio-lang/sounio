<!-- docs:meta
topic_id: repo.docs.audit.r2-5-rng-sampling-pcg64.synthesis
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-5-rng-sampling-pcg64.synthesis
-->

# R.2.5 — Cause A residual in rng.sio + sampling.sio — CLOSING SYNTHESIS

**Status:** RESOLVED (2026-05-17).
**Wall-clock:** ~2h, one session, on top of R.2.4 oracle.

## Defects fixed

A1 (dead `state_hi`), A2 (arithmetic shift on signed i64), A3 (`abs(i64::MIN)`)
— same set as R.2.4, applied to the two remaining stdlib PCG64 inlinings.

**New:** A4 (modulo bias in `*_bounded`) — `distributions.sio` doesn't
expose a bounded primitive so R.2.4 didn't have to handle this. Both
`pcg64_bounded` (rng.sio) and `smp_pcg64_bounded` (sampling.sio) used
`abs(x) % n` which skews when `n` doesn't divide `2^63`. Now use standard
PCG rejection-modulo with a binary long-division `u64 mod n`.

## Phases

| Phase | Output | Commit |
|---|---|---|
| A1 — sampling.sio rewrite | canonical step + pcg-cpp seeding + `smp_pcg_*` helpers + rejection-modulo bounded | `a7f89c0e` |
| A2 — rng.sio rewrite      | canonical step + splitmix64 seeding preserved + `rng_pcg_*` helpers + rejection-modulo bounded | `a7f89c0e` |
| B  — validation probes    | sampling 256/256 vs R.2.4 oracle, rng 1024/1024 self-oracle, 12/12 stat, 20/20 bounded | `a7f89c0e` |
| C  — lib.sio guidance + closing synthesis | this commit | (this commit) |

## Algorithm shipped (both files)

PCG-XSL-RR-128/64-LCG per O'Neill 2014 §6.3 with pcg-cpp `output_previous=false`:
- 128-bit state advance via emulated `u128_mul`/`u128_add` on i64 pairs.
- Output from **NEW** state: `rotr64((state_hi ^ state_lo), (state_hi >> 58) & 63)`.
- `MULT_128 = (0x2360ED051FC65DA4, 0x4385DF649FCCF645)`.
- Helpers re-inlined with module-local prefixes (`smp_pcg_*` / `rng_pcg_*`) — no shared `_pcg64_core.sio` module, no refactor.

### Seeding choices

- **sampling.sio** uses pcg-cpp canonical `seed(seed, stream=0)` with `INC_128 = (0x5851F42D4C957F2D, 0x14057B7EF767814F)`. Result: bit-exact to R.2.4 oracle.
- **rng.sio** keeps the splitmix64-derived `(state_hi, state_lo, inc_hi|1, inc_lo|1)` to preserve the multi-stream design. Result: cannot bit-match pcg-cpp; ships its own committed oracle as a regression guard.

### `*_bounded` debiasing (Cause A4 fix)

```
threshold = (2^64 mod n)                # via u64(-n) mod n
loop:
    x = pcg64_next_i64(rng)
    if u_lt(x, threshold): retry        # ≤50% rejection per call
    else: return x mod n                # via binary long-division
```

Power-of-two fast path (`n & (n-1) == 0 → x & (n-1)`). Bounded ≤100 retries
(rejection probability per call ≤ 0.5; underflow vanishing).

## Validation (Phase B)

### §7.1 — sampling.sio bit-exact vs R.2.4 oracle

Probe: `reference/sampling_validation_probe.sio` — `smp_pcg64_new(seed)` →
`smp_pcg64_next_i64` × 64 per seed, diffed against the head of each
`docs/audit/r2_4_pcg64_algorithmic_bug/reference/oracle_seed_*.txt`.

| seed | match |
|---|---|
| 0        | ✓ 64/64 |
| 1        | ✓ 64/64 |
| 31415    | ✓ 64/64 |
| 20260516 | ✓ 64/64 |

**256/256 bit-exact.**

### §7.2 — rng.sio self-oracle (regression guard)

Probe: `reference/rng_oracle_gen.sio` — 4 seeds × 256 samples committed to
`reference/rng_oracle_seed_*.txt` (sha256 in commit message). Future
regression: re-run gen, diff against committed files; any drift means the
algorithm or splitmix64 init changed.

### §7.3 — Statistical sanity (12/12 PASS, N=20000, seed=31415)

| dist        | rng.sio mean  | rng.sio var | smp mean   | smp var    |
|---          |---            |---          |---         |---         |
| Uniform(0,1)| 0.500573      | 0.084365    | 0.499561   | 0.084340   |
| Normal(0,1) | -0.012552     | 0.998070    | 0.001510   | 1.000464   |
| Exp(rate=1) | 0.989840      | 0.984519    | 1.002394   | 1.010144   |

All 6 (per module) within R.2.4 Phase C bands (`±0.01` / `±0.03` / `±0.03`
on means; `±0.01` / `±0.05` / `±0.06` on variances).

Note: `smp` numbers identical to R.2.4 Phase C — expected, since both use
the same canonical seeding. `rng` numbers differ because splitmix64 init
produces a different stream.

### §7.4 — Bounded uniformity (20/20 PASS)

`*_bounded(rng, 10)` × N=20000. Expected count per bucket = 2000, 3σ band
= [1873, 2127] where σ = √(N · 0.1 · 0.9) ≈ 42.43.

| bucket | rng.sio | sampling.sio |
|---|---|---|
| 0 | 2002 | 2042 |
| 1 | 2042 | 1921 |
| 2 | 1983 | 2005 |
| 3 | 2063 | 2004 |
| 4 | 1997 | 2054 |
| 5 | 2068 | 1986 |
| 6 | 1936 | 1982 |
| 7 | 1974 | 1964 |
| 8 | 2015 | 2025 |
| 9 | 1920 | 2017 |

All 20 inside [1873, 2127]. Empirical std across buckets:
- rng.sio: 49.0 (vs theoretical 42.4)
- sampling.sio: 38.4 (vs theoretical 42.4)

Both consistent with χ²-tail noise at N=20000, k=10 buckets.

## Acceptance against DISPATCH §7

| # | Criterion | Status |
|---|---|---|
| 1 | sampling.sio bit-exact vs R.2.4 oracle (256/256) | ✓ |
| 2 | rng.sio self-oracle committed (1024/1024) | ✓ |
| 3 | Statistical sanity 12/12 | ✓ |
| 4 | `*_bounded` uniformity 20/20 | ✓ |
| 5 | No regression in pre-R.2.5 PASSing gates that import rng/sampling | ✓ (no gate-level test imports these directly; downstream sampler `tests/run-pass/rapamycin_gum_vs_mc.sio` uses `distributions.sio` and stays PASS) |
| 6 | park_miller untouched | ✓ |
| 7 | distributions.sio untouched | ✓ |

## Files delivered

- `stdlib/random/rng.sio` — canonical step + helpers + rejection-modulo
- `stdlib/random/sampling.sio` — canonical step + helpers + rejection-modulo
- `stdlib/random/lib.sio` — guidance block rewritten for post-R.2.4/5 reality
- `docs/audit/r2_5_rng_sampling_pcg64/DISPATCH.md`
- `docs/audit/r2_5_rng_sampling_pcg64/SYNTHESIS.md` — this file
- `docs/audit/r2_5_rng_sampling_pcg64/reference/sampling_validation_probe.sio`
- `docs/audit/r2_5_rng_sampling_pcg64/reference/rng_oracle_gen.sio`
- `docs/audit/r2_5_rng_sampling_pcg64/reference/rng_oracle_seed_{0,1,31415,20260516}.txt`
- `docs/audit/r2_5_rng_sampling_pcg64/reference/phase_b_stat_rng.sio`
- `docs/audit/r2_5_rng_sampling_pcg64/reference/phase_b_stat_smp.sio`
- `docs/audit/r2_5_rng_sampling_pcg64/reference/phase_b_bounded_rng.sio`
- `docs/audit/r2_5_rng_sampling_pcg64/reference/phase_b_bounded_smp.sio`

## Out of scope (per DISPATCH §4)

- xoshiro256++, mt19937, splitmix64 standalone — untouched.
- distributions.sio — untouched (fixed in R.2.4).
- Refactor into a shared `_pcg64_core.sio` — explicitly deferred; would
  be a separate larger dispatch once all three callers are stable
  (they now are).

## Post-R.2.5 stdlib RNG state

All three PCG64 inlinings (`dst_pcg64_*` in distributions.sio,
`pcg64_*` in rng.sio, `smp_pcg64_*` in sampling.sio) ship the canonical
PCG-XSL-RR-128/64 algorithm. Park-Miller remains the recommended simple
default on quality-vs-complexity grounds (single 64-bit state, no 128-bit
arithmetic overhead per call), not as a defect workaround. The
`lib.sio` 2026-05-17 guidance block now reflects this.
