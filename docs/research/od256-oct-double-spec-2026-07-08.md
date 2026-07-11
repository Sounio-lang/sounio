<!-- docs:meta
topic_id: repo.docs.research.od256-oct-double-spec-2026-07-08
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.od256-oct-double-spec-2026-07-08
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# od256 — oct-double (~424-bit) software octuple precision — spec (draft)

Date: 2026-07-08
Status: draft spec + validated reference. Not yet implemented in Madaros codegen.

## Motivation

Precision in Sounio is a **software** construct built from error-free transforms
(EFT), not a silicon feature. `dd64` (double-double, ~106-bit) and `qd128`
(quad-double, ~212-bit) already prove this. `od256` is the next rung: an
**8-component non-overlapping double expansion**, ~**424-bit** (~127 decimal
digits), entirely in software, on CPU and (the goal) GPU.

The precision-track doc (`eisa-precision-track-2026-07-05.md`) scoped out *IEEE
binary256 hardware* — correctly, no such silicon exists. `od256` is the answer to
the same desire without waiting for hardware: the same technique that gives wide
arithmetic on an 8-bit machine gives octuple precision on `f64` lanes. "GPU
octuple" is exactly `od256` lowered through K-AXI → PTX.

## Representation

```
pub struct Od256 { x0, x1, x2, x3, x4, x5, x6, x7 : f64 }   // 64 bytes
```

Value `x = x0 + x1 + … + x7`, with `x0` most significant and the canonical
non-overlap target `|x_{i+1}| <= 0.5 ulp(x_i)`. Same invariant discipline as
`qd128`, extended from 4 to 8 limbs.

## Algorithm lineage

`qd128` follows Hida–Li–Bailey (ARITH-15, 2001) with **hand-unrolled** 4-limb
accumulation boxes (Six-Three-Sum, Nine-Two-Sum, …). Hand-unrolling 8 limbs is
intractable and error-prone, so `od256` uses the **generic-n multi-double**
approach of **CAMPARY** (Joldes, Muller, Popescu, Tucker — "Arbitrary Precision
Computing with the CAMPARY library", 2016; GPU-oriented by design):

- **EFT primitives reused verbatim from `math::dd64`**: `two_sum` (Knuth),
  `quick_two_sum` (Dekker), `two_prod` (Dekker split, no FMA — matches the
  hardware-agnostic path). No new kernels; the dd64 kernels are already witnessed
  bit-exactly (`tests/stdlib/math/test_dd64_eft_exact.sio`).
- **Exact accumulation via Shewchuk grow-expansion**: fold each raw term into a
  non-overlapping increasing-magnitude expansion. Exact, order-independent.
- **Renormalize = keep the K=8 most-significant limbs**; the discarded tail *is*
  the ~2^-424 rounding error. (CAMPARY's VecSum/VecSumErrBranch is the
  fast production variant; grow-expansion is the correctness-reference variant —
  they agree on value.)

## Operations (v0 scope)

| fn | with | notes |
|---|---|---|
| `od_zero`, `od_from_f64`, `od_from_dd`, `od_from_qd` | — / Mut | widening is exact |
| `od_to_f64`, `od_to_dd`, `od_to_qd` | — | narrowing = leading limbs |
| `od_renorm(terms)` | Mut | generic accumulate→8 limbs (the core) |
| `od_add`, `od_sub` | Mut, Panic | merge 16 limbs → renorm |
| `od_mul` | Mut, Panic | 64 two_prod partial products → renorm |
| `od_div` | Mut, Div, Panic | Newton on reciprocal (od precision) |
| `od_sqrt` | Mut, Div, Panic | Newton on inverse-sqrt |
| `od_neg`, `od_abs`, `od_cmp`, `od_eq`, `od_lt` | — | limb-wise |

Out of v0 scope: transcendentals, correctly-rounded results (multi-double is
faithful, not correctly-rounded), non-finite (NaN/Inf) semantics — caller
responsibility, same discipline as `qd128`.

## ABI note (dodges the Madaros many-f64-arg bug)

`Od256` is **64 bytes (> 16)** → SysV/AAPCS64 classify it **MEMORY**, so it is
passed by reference/stack, NOT as 8 scalar `f64` register args. This **avoids**
the known default-Madaros ABI bug on wide scalar-`f64` argument lists
(`BLK-20260707-madaros-f64-arg-abi-oct-mul`): always pass `Od256` by struct,
never explode it into 8 `f64` parameters. Until default codegen is proven for
this path, prove `od256` runtime via `SOUNIO_SOUC_ENGINE=lean_single`.

## Validated precision (reference vs mpmath)

Reference implementation `scripts/ci/od256_mpmath_gate.py` (Python, bit-identical
EFT to dd64) vs mpmath at 700-bit ground truth, 4000 random operands each ~424-bit:

| op | worst-case effective bits | ≈ decimal digits |
|---|---|---|
| add | 414 | 125 |
| sub | 419 | 126 |
| mul | 431 | 130 |
| π round-trip | 442 | 133 |

Gate thresholds: add/sub ≥ 400 bits, mul ≥ 390 bits. **PASS.** This validates the
*design* before any Madaros code is written; the `.sio` implementation must match
these numbers (run the same gate against traced `od256` output).

Honest caveats (inherited from the qd literature): Priest renormalization
sufficiency is proven only for ≤51 overlap bits; we rely on the same assumption
as `qd128`. Finite-domain only.

## GPU octuple path

CAMPARY is GPU-first. `od256` on GPU = lower the limb-wise EFT ops through
**K-AXI → PTX** (see the `gpu-kaxi-ptx-cubin` skill). This needs the K-AXI→PTX
emitter extended to emit the `two_sum`/`two_prod` limb sequences (analogous to the
still-scaffold `f32_assoc_gum` octonion lane). That extension is the concrete
"GPU octuple" deliverable.

## EISA connection

`od256` is a candidate **v3 `err`-lane carrier** for the Epistemic ISA (beyond
`dd64` in v0/v1 and `qd128` in v2) — for kernels whose roundoff must be tracked
below 2^-212. Orthogonal to the `u` (GUM) lane, which stays `f64`.

## Milestones

1. **M0 (DONE):** validated Python reference + mpmath gate (this doc).
2. **M1 (DONE, 2026-07-09):** `stdlib/math/od256.sio` compiles under `lean_single`
   and passes the exact-limb unit gate vs the Python reference for
   add/sub/mul/neg/abs — `tests/stdlib/math/test_od256_gate.sio` (`ALL PASS`,
   rc=0). `od_mul` already lowers and matches.
3. **M2:** replace the `od_div`/`od_sqrt` f64-seed stubs with Newton iterations;
   full mpmath gate green on traced output (incl. div/sqrt).
4. **M3 (SKETCH DONE, 2026-07-09):** K-AXI → PTX limb lowering (GPU octuple) —
   design + hand-written EFT PTX in `od256-gpu-lowering-2026-07-09.md` +
   `tests/golden/kaxi_ptx/od256/eft_primitives.ptx`. Remaining: `kaxi_to_ptx`
   emitter pattern, ptxas-accept, golden capture, GPU numeric gate.
5. **M4:** wire as EISA v3 err-lane (optional).

## Files

- Spec: this doc.
- Reference + CI gate: `scripts/ci/od256_mpmath_gate.py`.
- Skeleton: `stdlib/math/od256.sio`.
