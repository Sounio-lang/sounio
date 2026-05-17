<!-- docs:meta
topic_id: repo.docs.audit.r2-4-pcg64-algorithmic-bug.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-4-pcg64-algorithmic-bug.dispatch
-->

# DISPATCH R.2.4 — Cause A: stdlib PCG64 algorithmic bug

**Opened.** 2026-05-17
**Predecessor.** R.2.3 (compiler `(Struct, scalar)` tuple-return bug — LANDED in commit `95b68a04`, see `docs/audit/r2_3_compiler_tuple_return_bug/SYNTHESIS_F3.md`)
**Class.** Pure stdlib fix in `stdlib/random/distributions.sio`. No compiler involvement.
**Priority.** P3 — does NOT block PBPK28 (park_miller already shipped as canonical RNG); does block any future code that wants PCG64 quality without algorithm review.
**Branch.** Operator chooses; recommend a fresh branch off `sounio-pure/r2-1-park-miller` so the compiler fix from R.2.3 is in scope for validation.
**Time budget.** 2–6h in a single session.

---

## §0 — Sounio-Pure constraint

This dispatch may read/write `stdlib/random/distributions.sio`, write probes under `tests/`, and compile/run with the in-repo `./bin/souc`. No Python / R / external tools for diagnostics or validation — Sounio-only. Statistical reference values may be quoted from published PCG literature (O'Neill 2014) or generated with the in-repo `bin/kretikos` if needed for byte-level golden comparison, but the validators themselves must be `.sio` programs.

If the fix needs to touch the compiler (it shouldn't — R.2.3 closed the compiler-side issue), **HALT and report** — R.2.4 explicitly excludes compiler changes.

---

## §1 — Bug definition: Cause A

`stdlib/random/distributions.sio:62-102` defines `DstPcg64` and its three samplers (`dst_pcg64_new`, `dst_pcg64_next_i64`, `dst_pcg64_next_f64`). The current implementation has three distinct algorithmic defects, documented in R.2.1 Phase D synthesis and reproduced below verbatim:

```sounio
fn dst_pcg64_next_i64(rng: DstPcg64) -> (DstPcg64, i64) {
    let old_state = rng.state_lo
    let new_lo = rng.state_lo * 6364136223846793005 + rng.inc_lo
    let new_hi = rng.state_hi + rng.inc_hi              // ← never re-read

    let xorshifted = (old_state >> 18) ^ old_state      // ← `>>` is ARITHMETIC on Sounio i64
    let rot = (old_state >> 59) & 63                    // ← same problem
    let result = xorshifted >> rot                       // ← compounded

    return (DstPcg64 { state_hi: new_hi, state_lo: new_lo,
                       inc_hi: rng.inc_hi, inc_lo: rng.inc_lo }, result)
}

fn dst_pcg64_next_f64(rng: DstPcg64) -> (DstPcg64, f64) {
    let result = dst_pcg64_next_i64(rng)
    let bits = result.1
    let positive = if bits < 0 { 0 - bits } else { bits }  // ← FAILS on i64::MIN
    let fraction = (positive as f64) / 9223372036854775807.0
    return (result.0, fraction)
}
```

**Defect A1 — `state_hi` is dead weight.**
The recurrence advances `state_lo` only (`state_lo * mult + inc_lo`). `state_hi` is incremented by `inc_hi`=1 every step but **never participates in `xorshifted`, `rot`, or the output `result`**. Effectively this is a single-i64-state Lehmer-like generator, not a real 128-bit PCG. Period is therefore at most ~2^62 (single-state LCG with this modulus structure), not the 2^128 advertised by the PCG64 name.

**Defect A2 — Arithmetic right-shift on signed i64.**
Sounio's `>>` operator sign-extends when the operand is negative (verified empirically in R.2.1 d4 probe: `(0 - 0x7000000000000000) >> 18 = -30786325577728`). PCG64's design assumes **logical / unsigned** right shift on its state register. Once `state_lo` becomes negative, both `(old_state >> 18)` and `(old_state >> 59)` propagate sign bits, fundamentally distorting the output distribution. The XOR-and-rotate output pattern breaks.

**Defect A3 — `abs(i64::MIN)` returns negative.**
`if bits < 0 { 0 - bits } else { bits }` computes the absolute value, but **`0 - (-2^63) = -2^63`** under two's-complement i64 (the most negative value is its own negation). When `dst_pcg64_next_i64` happens to return `i64::MIN`, the subsequent `(positive as f64) / 9.22e18` produces a **negative** f64 — i.e. `dst_pcg64_next_f64` can return values outside the documented [0, 1] range.

---

## §2 — What R.2.1 already ruled in/out

From `r2_1_pcg_fix/diagnostic/SYNTHESIS_D.md`:

| Hypothesis | Probe | Verdict |
|---|---|---|
| H1: i64 multiplication panics / fails to wrap | `d3_h1_overflow.sio` | NEGATED — multiply wraps deterministically |
| H2: `>>` on negative i64 is arithmetic | `d4_h2_shifts.sio` | CONFIRMED — sign-extending |
| H3: multiplier literal truncated | quick println | NEGATED — `6364136223846793005` round-trips exactly |
| H4: state propagation through nested call corrupted | `d6_h4_struct.sio`, `d8_state_leak.sio` | CONFIRMED — but root cause was **Cause B (compiler)**, closed in R.2.3 |

So under the R.2.3-fixed compiler, H4 should now be silent. R.2.4's scope is to fix A1, A2, A3.

---

## §3 — Attack plan

### Phase A — Reference oracle (30 min)

Goal: establish a known-good byte-level reference stream for fixed seeds.

1. Pick the canonical PCG64-XSL-RR variant from O'Neill 2014 (or the simpler PCG-XSH-RR for 64-bit state if the algorithm-cost trade-off matters). The canonical c-style reference fits in <40 lines.
2. Write `reference/pcg64_reference.sio` — a Sounio-native, correctness-first implementation. Use one of:
   - **Logical right shift via mask:** `lshr(x, n) = (x >> n) & ((1 << (64 - n)) - 1)` for `0 < n < 64`. Wrap as a helper.
   - **Explicit 32-bit halves:** model state as two i64s holding the low 32 bits of an unsigned 64-bit each, sidestepping arithmetic-shift entirely.
3. Generate 256 samples from seeds {0, 1, 31415, 20260516}. Capture as `reference/oracle_seed_{N}.txt` (one i64 per line).
4. Cross-check the first few values against a trusted external implementation (e.g. PCG's own C reference, run separately by the operator outside the workspace). Once validated, the oracle becomes the in-repo reference.

### Phase B — Direct fix in distributions.sio (1–3h)

Three fix options ranked by surgery:

**(B1) Mask-based logical-shift retrofit.** Minimal change. Add `dst_pcg64_lshr(x, n) -> i64` helper that masks off sign-extended bits. Rewrite the three `>>` sites in `dst_pcg64_next_i64`. Fix `dst_pcg64_next_f64` to handle `i64::MIN`. **Doesn't fix A1** — `state_hi` remains dead weight, so the generator stays single-state. Period and quality remain low. Acceptable if "PCG-like" is enough.

**(B2) Promote to true 128-bit state.** Make `state_hi`/`state_lo` model a real 128-bit unsigned integer; carry the multiplication across both halves. Use the standard PCG carry pattern: `(a_hi * b_lo + a_lo * b_hi + carry_from_lo_lo) → new_hi`. Then output mixing uses bits from BOTH halves. Fixes A1, A2 (with logical-shift helper), A3. Significantly larger diff (~50 lines), but actually matches the PCG64 name. Verified against reference oracle from Phase A.

**(B3) Single-state PCG variant.** Drop `state_hi`. Convert to PCG-XSH-RR with a 64-bit state and 32-bit output, then concatenate two outputs into a 64-bit result. Smaller code; period only ~2^62; quality lower than B2 but better than current B1-style. Document quality drop explicitly.

Recommended: **B2** for stdlib (matches the named contract), with explicit unit tests against the reference oracle.

### Phase C — Validation (30–60 min)

1. **Byte-level oracle match.** New `tests/stdlib/random/test_dst_pcg64_oracle.sio` reads `reference/oracle_seed_*.txt` and asserts bit-equal output for first 256 samples on each seed. PASS = full match for all four seeds.

2. **Statistical sanity (small N).** New `tests/stdlib/random/test_dst_pcg64_stats.sio`:
   - 10,000 calls to `dst_pcg64_next_f64`; assert mean ∈ [0.49, 0.51], variance ∈ [0.082, 0.085] (uniform [0,1] has μ=0.5, σ²=1/12≈0.0833).
   - Assert no negative outputs (catches A3 regression).
   - Assert no stuck values: max consecutive identical outputs ≤ 1.
   - **NOT** TestU01-level; that's a multi-day fixture beyond R.2.4's budget. Document this gap.

3. **Phase D regression replay.** Re-run R.2.1's `d6_h4_struct.sio` and `d8_state_leak.sio` on the fixed PCG64 — Cause B's symptom should now be absent (validated already in R.2.3) AND Cause A's stuck-zero / negative-f64 symptoms should now be absent. Compare to park_miller as a sanity ladder.

4. **No regression in PBPK paths.** Re-run `dissertation_pbpk_suite` umbrella sub-gate. Since current PBPK code uses park_miller and not `dst_pcg64`, the fix should be additive — verify the dispatch hasn't accidentally regressed.

### Phase D — Restore stdlib RNG canonicity (15 min)

Once Phase C passes:
1. Remove the `!! DEPRECATED PCG64 BACKEND !!` header from `stdlib/random/distributions.sio` (lines 3-31).
2. Update `stdlib/random/lib.sio` guidance: PCG64 returns to canonical role; park_miller remains as a lightweight option, not a workaround.
3. Update `docs/audit/r2_2_b/SYNTHESIS.md` status field to "RESOLVED in R.2.4".
4. Update `docs/audit/r2_3_compiler_tuple_return_bug/SYNTHESIS_F3.md` "Remaining work" — strike Cause A line.

### Phase E — Commit (10 min)

Single commit on the dispatch branch:
- `stdlib/random/distributions.sio` — the fix
- `tests/stdlib/random/test_dst_pcg64_oracle.sio` — byte-level golden
- `tests/stdlib/random/test_dst_pcg64_stats.sio` — statistical sanity
- `docs/audit/r2_4_pcg64_algorithmic_bug/` — dispatch + SYNTHESIS + reference oracle
- `stdlib/random/lib.sio` — guidance update
- HALT for operator review before push.

---

## §4 — Out of scope

- **TestU01 / PractRand.** Multi-day fixture. Defer to a separate quality-assurance dispatch if needed.
- **Other RNGs (xoshiro, mt19937, etc.).** Leave alone.
- **park_miller.sio.** Stays as-is — single-state, lightweight option.
- **Compiler changes.** R.2.3 closed that channel. If a fix here requires touching the compiler, HALT immediately.
- **stdlib distribution samplers downstream of PCG64** (Normal / LogNormal / Gamma / Beta / Poisson). They auto-benefit from a fixed PCG64; no direct edits to their bodies.

---

## §5 — Halt conditions

- **A reference-oracle mismatch persists after B2 implementation.** Surface is broader than thought; bring the oracle stream + the failing Sounio output back to the operator.
- **The fix introduces regressions in any existing umbrella sub-gate.** Revert the stdlib change (no compiler rebuild needed; `bin/souc` is unaffected).
- **Statistical sanity fails despite oracle match.** Means the oracle itself is wrong — re-derive from scratch and report.
- **Any temptation to use Python / R / external tooling for diagnostics.** Stop and ask.

---

## §6 — Deliverables on close

1. `stdlib/random/distributions.sio` — diff. Smallest correct change consistent with B2 (or B1/B3 if operator chose smaller scope).
2. `tests/stdlib/random/test_dst_pcg64_oracle.sio` — golden test.
3. `tests/stdlib/random/test_dst_pcg64_stats.sio` — statistical sanity.
4. `docs/audit/r2_4_pcg64_algorithmic_bug/SYNTHESIS.md` — closing writeup.
5. `docs/audit/r2_4_pcg64_algorithmic_bug/reference/oracle_seed_*.txt` — byte-level reference streams.
6. Removed deprecation header in `distributions.sio`.
7. Updated `stdlib/random/lib.sio` guidance.

---

## §7 — Acceptance

R.2.4 is **VALIDATED** if and only if:

1. ✓ Byte-level oracle match on all four canonical seeds (256 samples each).
2. ✓ Statistical sanity (10k samples, mean / variance / no-negatives / no-stuck).
3. ✓ R.2.1 d6/d8 regression probes show no stuck-zero / negative-f64 behavior.
4. ✓ `umbrella_gate` PBPK sub-suite passes (no PBPK regression).
5. ✓ park_miller self-test still bit-exact (defensive).
6. ✓ Deprecation header removed; lib.sio guidance updated.

If 1-3 pass but 4 fails: PARTIAL. Investigate.
If 1 fails: FAIL. The fix algorithm is wrong; back to Phase A.

---

## §8 — Notes

- Park-Miller stays canonical for thesis-bound outputs **until** R.2.4 validates AND a separate confidence pass confirms PCG64 is genuinely better for the PBPK use case. This is a deliberate caution — even with the algorithm fixed, switching the dissertation RNG mid-cycle requires re-validating downstream Monte Carlo bounds.
- `bin/souc-linux-x86_64` should be untouched in this dispatch. If a rebuild is needed (it shouldn't be), document it carefully.
- The 4-field `DstPcg64` struct return is now safe because R.2.3 fixed the SRET-slot collision. No need to redesign the struct.

**END OF DISPATCH.**
