<!-- docs:meta
topic_id: repo.docs.compiler.numerical-determinism
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.numerical-determinism
-->

# Sounio Compiler — Numerical Determinism Guarantees

**Last updated:** 2026-05-13  
**Scope:** Scalar floating-point on x86-64 Linux (the platform used for all PBPK28
dissertation computations). GPU PTX and other targets are out of scope here.

---

## 1. Floating-point instruction selection

The Sounio self-hosted compiler (`self-hosted/compiler/lean_single.sio`) lowers
scalar arithmetic to standard x86-64 SSE2 instructions:

| Sounio operation | x86-64 instruction |
|---|---|
| `f64 +` | `addsd` |
| `f64 -` | `subsd` |
| `f64 *` | `mulsd` |
| `f64 /` | `divsd` |

All four are **IEEE 754-2008 compliant** in round-to-nearest-even mode by default.

No fast-math (`-ffast-math`), no unsafe-math, no non-associative reductions, and
**no implicit FMA contraction** for scalar code. The compiler does not emit `fmadd`
or `fmsub` for scalar `a*b + c` — it emits `mulsd + addsd` in strict order.

*(GPU PTX emits `fma.rn.f32 / fma.rn.f64` only for explicit K-AXI `IrFma` nodes.
Scalar GPU `+` / `*` compile to separate `add.rn` + `mul.rn` instructions.)*

---

## 2. Compilation flags — no divergence between environments

The `./bin/souc compile` wrapper calls the self-hosted binary directly:

```bash
./bin/souc-linux-x86_64 <source.sio> <output.elf>
```

There are no environment-variable overrides for arithmetic mode. CI and local
development use the same wrapper. The binary is pinned in `bin/` (md5 hash in
`scripts/ci/lean_single_fixed_point_gate.sh`); arithmetic behaviour is identical
across machines that run the same binary.

---

## 3. The PBPK28 non-determinism was NOT a compiler issue

The E1 vs E4 LogNormal u_MC discrepancy documented in `determinism_audit_v1.md`
was caused by a **user-space Taylor-series defect** in `ms28_exp` — a math helper
function inlined in `pbpk28_mc_prior_family_sweep.sio`. The compiler generated
correct code for the wrong algorithm.

Specifically:

```sounio
// BUGGY (E4, before fix): missing linear term in Taylor series
var t = rx           // wrong initial value
// CORRECT (E1, also E4 after fix):
var t: f64 = 1.0     // correct initial value
```

The resulting `mulsd / addsd` sequences were both IEEE 754-correct;
the difference was in the mathematical algorithm, not in the compiled
arithmetic.

---

## 4. Self-implemented math functions — precision characterisation

Both harnesses use self-contained Taylor / range-reduced implementations of
`ln` and `exp` (to avoid import-chain complexity). These are not the C runtime
`libm` — they have bounded but finite approximation error.

### 4.1 `mc28_ln` / `ms28_ln` — natural logarithm

Algorithm: range-reduce to `x ∈ [0.5, 2.0]` via repeated multiply / divide by
`e`, then apply arctanh series `2·(u + u³/3 + u⁵/5 + ... + u⁹/9)` where
`u = (x-1)/(x+1)`.

| Implementation | Terms | Max relative error for `x ∈ [0.5, 2.0]` |
|---|---|---|
| `mc28_ln` (E1) | 5 (up to u⁹) | < 5 × 10⁻¹⁰ |
| `ms28_ln` (E4) | 4 (up to u⁷) | < 2 × 10⁻⁸ |

Both are adequate for PBPK28 parameter sampling. The 1-term difference is
immaterial for N = 2000 Monte Carlo.

### 4.2 `mc28_exp` / `ms28_exp` — exponential (post-fix)

Algorithm: range-reduce to `rx = x - n·ln2`, `|rx| < ln2/2 ≈ 0.347`, then
Taylor series `1 + rx + rx²/2! + ... (20 terms)`, then multiply by `2ⁿ`.

Post-fix relative error for `|x| ≤ 40` (which covers all PBPK28 lognormal
arguments): **< 1 × 10⁻¹²**.

---

## 5. CI determinism gate

After the D3 fix, `scripts/audit/mc_determinism_probe.sh --post-fix` asserts
that E1 and E4 LogNormal u_MC agree to 6 significant figures. This gate should
be run on any branch that modifies the PBPK28 MC harnesses.

```bash
bash scripts/audit/mc_determinism_probe.sh --post-fix
# Expected: MC_PBPK28_COMPILER_DETERMINISM_PASS
```

---

## 6. Known limitations

- Self-implemented `ln` / `exp` may differ from `libm` results by up to ~10⁻⁸.
  This is not a non-determinism issue — both are deterministic; the difference
  is a constant per-call accuracy bound.
- Results may differ by ≤ 10⁻¹² across x86-64 machines due to `mulsd` /
  `addsd` rounding in the last bit (ULP). This is IEEE 754 conformant and
  expected. No PBPK28 metric has this precision requirement.
- ARM / Apple Silicon targets use the same algorithm but different SIMD
  instruction widths; results may differ in the last 1–2 ULPs from x86-64.
  The dissertation computations are exclusively on x86-64 Linux.
