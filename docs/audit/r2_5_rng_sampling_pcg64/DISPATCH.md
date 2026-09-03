<!-- docs:meta
topic_id: repo.docs.audit.r2-5-rng-sampling-pcg64.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-5-rng-sampling-pcg64.dispatch
-->

# DISPATCH R.2.5 — Cause A residual: rng.sio + sampling.sio PCG64

**Opened.** 2026-05-17
**Predecessor.** R.2.4 (Cause A in `stdlib/random/distributions.sio` — RESOLVED commits `a148f070`/`f686d6fe`/`1410cc39`/`e6801d254`).
**Class.** Pure stdlib fix. No compiler involvement.
**Priority.** P3 — does NOT block PBPK28 (park_miller canonical; R.2.4 fixed the dst_pcg64 path). Does block any future code that imports `rng_pcg_*` from `random::rng` or `smp_pcg64_*` from `random::sampling`.
**Branch.** Operator chooses; recommend continuing on `sounio-pure/r2-1-park-miller` so R.2.4's reference oracle is in scope without re-deriving it.
**Time budget.** 1–3h in a single session.

---

## §0 — Sounio-Pure constraint

May read/write `stdlib/random/rng.sio`, `stdlib/random/sampling.sio`, `stdlib/random/lib.sio` (guidance), and probes under `docs/audit/r2_5_rng_sampling_pcg64/reference/`. No Python / R / external tools — Sounio-only. The R.2.4 reference oracle at `docs/audit/r2_4_pcg64_algorithmic_bug/reference/oracle_seed_*.txt` is the canonical bit-exact target; **no new pcg-cpp cross-check is needed** (R.2.4 already certified it 32/32).

If the fix needs to touch the compiler, **HALT and report** — R.2.5 explicitly excludes compiler changes. R.2.3 closed the compiler-side issue.

---

## §1 — Bug definition

Two stdlib modules ship parallel, independent PCG64 implementations with the exact same Cause A defects that R.2.4 fixed in `distributions.sio`:

### §1.1 — `stdlib/random/rng.sio` (`pcg64_*` + `RngPcgWrapper`)

```sounio
fn pcg64_next_i64(rng: Pcg64) -> (Pcg64, i64) {
    let old_state = rng.state_lo
    let new_lo = rng.state_lo * 6364136223846793005 as i64 + rng.inc_lo
    let new_hi = rng.state_hi + rng.inc_hi              // ← Defect A1 (dead)
    let xorshifted = (old_state >> 18) ^ old_state      // ← Defect A2 (arith shift)
    let rot = (old_state >> 59) as i64                  // ← Defect A2
    let result = xorshifted >> (rot & 63)               // ← Defect A2 compounded
    ...
}

fn pcg64_next_f64(rng: Pcg64) -> (Pcg64, f64) {
    ...
    let positive = if bits < 0 { 0 - bits } else { bits }   // ← Defect A3 (i64::MIN)
    let fraction = (positive as f64) / 9223372036854775807.0
    ...
}

fn pcg64_bounded(rng: Pcg64, n: i64) -> (Pcg64, i64) {
    ...
    let positive = if x < 0 { 0 - x } else { x }       // ← Defect A3 again
    let bounded = positive % n                          // ← Defect A4 (modulo bias)
    ...
}
```

Public surface affected: `pcg64_new`, `pcg64_next_i64`, `pcg64_next_f64`, `pcg64_next_f64_nonzero`, `pcg64_next_bool`, `pcg64_bounded`, plus the `RngPcgWrapper` shims (`rng_pcg_new`, `rng_pcg_next_f64`, `rng_pcg_next_i64`, `rng_pcg_next_bool`, `rng_pcg_bounded`).

### §1.2 — `stdlib/random/sampling.sio` (`smp_pcg64_*`)

Same algorithm pattern as `rng.sio` and pre-R.2.4 `distributions.sio`:

```sounio
fn smp_pcg64_new(seed: i64) -> SmpPcg64 {
    SmpPcg64 {
        state_hi: seed ^ 0x853c49e6748fea9b,            // ← never advances
        state_lo: seed * 6364136223846793005,
        inc_hi: 1,
        inc_lo: 1,
    }
}
// smp_pcg64_next_i64 / next_f64 / bounded: identical defect set to §1.1
```

Public surface affected: `smp_pcg64_new`, `smp_pcg64_next_i64`, `smp_pcg64_next_f64`, `smp_pcg64_next_f64_nonzero`, `smp_pcg64_bounded`. Downstream samplers in `sampling.sio` (`sample_normal`, `sample_index`, `sample_weighted_index`, `smp_shuffle_f64`, etc.) inherit the degeneracy.

### §1.3 — Defect taxonomy (same as R.2.4)

- **A1 — `state_hi` dead weight.** `state_hi += inc_hi` but never enters output mixing.
- **A2 — Arithmetic right shift on signed i64.** Sounio `>>` sign-extends.
- **A3 — `abs(i64::MIN)` returns `i64::MIN`.** `if x < 0 { 0 - x }` is identity at the most-negative value.
- **A4 — Modulo bias in `*_bounded`.** `positive % n` skews when `n` doesn't divide `2^63`. (New diagnosis vs R.2.4 — `distributions.sio` doesn't expose a `*_bounded`, so this only matters in `rng.sio` / `sampling.sio`.)

---

## §2 — What R.2.4 already established

| Resource | Status |
|---|---|
| Canonical algorithm (PCG-XSL-RR-128/64-LCG, output_previous=false) | Documented in `r2_4/SYNTHESIS.md` "Algorithm shipped". |
| Helpers (`lshr`, `umul64_high`, `u128_add`, `u128_mul`, `rotr64`, `u_lt`) | Live in `stdlib/random/distributions.sio` as `dst_pcg_*`. |
| Reference oracle (4 seeds × 256 samples) | `docs/audit/r2_4_pcg64_algorithmic_bug/reference/oracle_seed_*.txt`. |
| External cross-check (pcg-cpp HEAD) | 32/32 fingerprint + 1024/1024 oracle (already done; no re-cross-check needed for R.2.5). |
| Statistical sanity bands (Uniform/Normal/Exp, N=20000) | `phase_c_stat_sanity.sio`, 6/6 PASS. |

R.2.5 therefore **inherits** the algorithm + oracle from R.2.4. No new derivation.

---

## §3 — Attack plan

### Phase A — Algorithm parity (45 min)

Rewrite `pcg64_*` in `rng.sio` and `smp_pcg64_*` in `sampling.sio` against the canonical algorithm. Decision points:

1. **Helper sharing.** Three options:
   - (a) **Re-inline** the six helpers (`*_lshr`, `*_umul64_high`, `*_u_lt`, `*_u128_add`, `*_u128_mul`, `*_rotr64`) under module-local prefixes (`rng_pcg_*`, `smp_pcg_*`). Zero cross-module coupling, ~50 LOC per module. **Recommended.**
   - (b) Make `distributions.sio` helpers `pub` and `use` them. One source of truth; introduces an import edge in the stdlib graph.
   - (c) Hoist helpers into a new `stdlib/random/_pcg64_core.sio` shared module. Clean factoring but bigger blast radius.

2. **Seeding.** Two options:
   - (a) **Keep splitmix64-derived `(state_hi, state_lo, inc_hi, inc_lo)` in `rng.sio`** so callers retain the multi-stream story `rng.sio` was designed around — but apply the new step algorithm. Quality: bit-correct algorithm; streams differ from pcg-cpp pcg64 because of the splitmix64 init (intentional). **Recommended for `rng.sio`.**
   - (b) Switch to canonical pcg-cpp `seed(seed, stream=0)` init. Bit-exact to pcg-cpp / R.2.4 oracle, but loses the splitmix64 multi-stream design. **Recommended for `sampling.sio`** since `smp_pcg64_new` is single-seed convenience already.

3. **`*_bounded` debias.** Use the standard PCG rejection trick: `threshold = ((2^64 - n) % n) cast to i64`; sample u64, reject if `u_lt(sample, threshold)`, then `sample % n`. ~10 LOC. Bounded loop iteration count (≤2 with overwhelming probability for n ≪ 2^63).

### Phase B — Validation (30 min)

1. **`sampling.sio` oracle match.** `smp_pcg64_new(seed)` with seeding option (b) must produce bit-exact output against R.2.4 oracle for all 4 seeds × first N (recommend N=64 for size). New probe at `docs/audit/r2_5_rng_sampling_pcg64/reference/sampling_validation_probe.sio`. PASS = 4×64 = 256 bit-exact.

2. **`rng.sio` self-consistency.** Since `rng.sio` keeps splitmix64 seeding (option 2a), bit-exact-to-pcg-cpp is impossible. Instead, write its own oracle: 4 seeds × 256 samples emitted from the new `pcg64_*`, committed as `rng_oracle_seed_*.txt`. PASS = this run matches its own committed oracle (regression guard).

3. **Statistical sanity.** Re-run the R.2.4 Phase C probe pattern against both `rng_pcg_next_f64` (via `RngPcgWrapper`) and `smp_pcg64_next_f64`. 12/12 bands (6 per module). Tolerances identical to R.2.4 Phase C.

4. **`*_bounded` uniformity.** N=20000 draws from `pcg64_bounded(rng, 10)` and `smp_pcg64_bounded(rng, 10)`: each bucket count ∈ [N/10 - 3σ, N/10 + 3σ] where σ = √(N·p·(1-p)) ≈ 42.4. Bands: [1873, 2127] per bucket.

### Phase C — Doc cleanup (15 min)

1. `stdlib/random/lib.sio`: confirm guidance still aligned (no edit needed if already clean post-R.2.4).
2. `stdlib/random/README.md`: spot-check for any "use park_miller because PCG is broken" language; remove if present.
3. New `SYNTHESIS.md` in `docs/audit/r2_5_rng_sampling_pcg64/` against §7 acceptance.

### Phase D — Commit (10 min)

Single commit per phase (A, B, C) on the dispatch branch. HALT for operator review between Phase B and C if any band fails.

---

## §4 — Out of scope

- **xoshiro256++.** Different algorithm family; not part of this dispatch.
- **mt19937 / splitmix64 standalone.** SplitMix64 is correctly implemented (already used as a seed mixer); no change.
- **`stdlib/random/distributions.sio`.** Fixed in R.2.4. Touching it here is a sign of scope creep — HALT.
- **`tests/run-pass/`-level regression discovery.** Cause A in `rng.sio`/`sampling.sio` has been live long before R.2.1; any downstream test that "passes" with broken streams was either degenerate-by-construction or used Park-Miller. Diagnosing each is a separate exercise.
- **Compiler changes.** R.2.3 closed that channel.

---

## §5 — Halt conditions

- **Reference oracle mismatch under seeding option 2b (sampling.sio).** Means the helper re-inlining drifted from R.2.4's certified copy. Bring the diff back to the operator.
- **Regression in any committed `tests/stdlib/random/*` or `tests/run-pass/*` gate that imports `rng.sio` or `sampling.sio` and was PASSing pre-R.2.5.** Revert and surface; the old degenerate streams may have been load-bearing in some test that conflated "stuck value" with "correct".
- **Statistical bands fail.** Algorithm or seeding drifted; surface the diff against R.2.4 `phase_c` output.
- **Any temptation to refactor the broader stdlib RNG architecture** (e.g. "let me also unify Pcg64 + DstPcg64 + SmpPcg64 into one type"). HALT and ask. R.2.5 is parity, not redesign.

---

## §6 — Deliverables on close

1. `stdlib/random/rng.sio` — canonical PCG64 step + helpers under `rng_pcg_*` prefix; splitmix64 seeding preserved.
2. `stdlib/random/sampling.sio` — canonical PCG64 step + helpers under `smp_pcg_*` prefix; pcg-cpp canonical seeding.
3. `docs/audit/r2_5_rng_sampling_pcg64/reference/sampling_validation_probe.sio` — 4×64 bit-exact vs R.2.4 oracle.
4. `docs/audit/r2_5_rng_sampling_pcg64/reference/rng_oracle_seed_{0,1,31415,20260516}.txt` — `rng.sio` self-oracle (256 samples × 4 seeds).
5. `docs/audit/r2_5_rng_sampling_pcg64/reference/phase_b_stat_sanity.sio` — 12-band statistical probe (6 per module).
6. `docs/audit/r2_5_rng_sampling_pcg64/reference/phase_b_bounded_uniformity.sio` — chi-square buckets for `*_bounded`.
7. `docs/audit/r2_5_rng_sampling_pcg64/SYNTHESIS.md` — closing writeup.

---

## §7 — Acceptance

R.2.5 is **VALIDATED** if and only if:

1. ✓ `sampling.sio` `smp_pcg64_*` (with canonical pcg-cpp seeding) bit-exact against R.2.4 oracle for 4 seeds × 64 samples = 256/256.
2. ✓ `rng.sio` `pcg64_*` (with splitmix64 seeding) bit-exact against its newly-committed self-oracle for 4 seeds × 256 samples = 1024/1024 (regression guard).
3. ✓ Statistical sanity 12/12 PASS (6 per module: Uniform mean+var, Normal mean+var, Exp mean+var at N=20000).
4. ✓ `*_bounded` uniformity 20/20 buckets within 3σ band (10 per module, n=10).
5. ✓ No regression in any pre-R.2.5 PASS gate that imports `rng.sio` or `sampling.sio`.
6. ✓ Park-Miller untouched.
7. ✓ `stdlib/random/distributions.sio` untouched.

If 1 or 2 fails: FAIL. Algorithm/seeding drifted from R.2.4. Back to Phase A.
If 3 or 4 fails but 1/2 pass: PARTIAL. Investigate band derivation.
If 5 fails: HALT. The old broken stream was load-bearing in a test; report before deciding.

---

## §8 — Notes

- DISPATCH §4 of R.2.4 explicitly deferred `rng.sio` and `sampling.sio` ("Other RNGs … Leave alone"). R.2.5 picks up that scope. There is no new bug discovery here; R.2.4 SYNTHESIS §"Known follow-up" already named these.
- Three independent PCG64 inlinings exist in stdlib because each `.sio` module was authored before stdlib cross-imports were ergonomic. R.2.5 keeps that structural choice (helper option 1a) rather than refactoring; a unified `_pcg64_core.sio` is a separate, larger dispatch best run only after all three callers are stable.
- The `RngPcgWrapper` API is preserved — `rng.sio`'s `rng_pcg_new` / `rng_pcg_next_*` / `rng_pcg_bounded` keep their signatures; downstream callers don't move.

**END OF DISPATCH.**
