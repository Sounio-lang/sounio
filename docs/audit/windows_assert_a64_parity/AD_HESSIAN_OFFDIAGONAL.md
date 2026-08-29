<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.ad-hessian-offdiagonal
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.ad-hessian-offdiagonal
-->

# A64 PARITY — second-order AD: off-diagonal + multiply Hessian (commit 2 of 2)

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED. Completes the a64 second-order AD
started in `AD_HESSIAN_DIAGONAL.md`.
**Class.** Codegen on `self-hosted/compiler/lean_single.sio` —
`compile_primary_a64` (`hessian_of` off-diagonal reads + multiply-operator
Hessian) + new helper `emit_mul_hessian_comp_a64`.
**Branch.** `feat/windows-assert-exit`.

---

## §1 — The gap (after commit 1)

Commit 1 gave diagonal Hessians (`h_jj`) through transcendentals and a
`hessian_of` frontend that read only the diagonal. The off-diagonal cross-terms
`h_jk` (j≠k) were absent, so multivariate compositions like `hessian_of(x*atan(y),0,1)`
returned 0 on a64 (x86 `0.5`).

## §2 — The fix

1. **Multiply-operator Hessian.** New helper `emit_mul_hessian_comp_a64` emits one
   component `H_jk(a·b) = H_jk(a)·b + s_j(a)·s_k(b) + s_k(a)·s_j(b) + a·H_jk(b)`.
   The `b4am` multiply block now captures LHS/RHS Hessian shadows (10 components
   each) and calls the helper for the full 4-channel symmetric block (h00,h01,h02,
   h03,h11,h12,h13,h22,h23,h33). The `s_j(a)·s_k(b)+s_k(a)·s_j(b)` outer-product
   term is what produces a cross-Hessian from two first-order gradients.
2. **Frontend off-diagonal reads.** `hessian_of`'s dispatch now resolves the 6
   off-diagonal slots (h01,h02,h03,h12,h13,h23) in addition to the diagonal,
   including the `VAR_HSHADOW_jk` lookup for bound variables.

**Not added (deliberate):** off-diagonal Hessian through a *single transcendental*
of a multi-channel argument (e.g. `sin(x+y)`). x86 **GTT-refuses** that form (the
gradient-topology check rejects `hessian_of(sin(x+y),0,1)` at compile time), so it
is unreachable in supported code; a64 has no GTT layer and returns 0 there — a
minor, documented divergence, not a missing capability.

## §3 — Verification (real Apple M3)

- **Self-host fixed point.** PASS: stage1==stage2==stage3,
  `md5=700e7b7926c3fb47b214d9da779802f6`; binary rebuilt.
- **Target test 6/6.** `epistemic_hessian_transcendentals` output byte-matches x86
  on M3: `-0.5, 0.0, 0.5, 0.0, 1.0, 0.0` — including the two product cross-terms
  `hessian_of(x*atan(y),0,1)`=0.5 and `hessian_of(asin(x)*y,0,1)`=1.0.
- **Extra multiply components** (M3 == x86): `x*y` h01=1.0, h00=0.0; `exp(x)*y`
  h01=1.648721.
- **x86 non-regression.** `epistemic_hessian_transcendentals`,
  `pbpk_rapamycin_second_order`, `sensitivity_multi_channel`,
  `sensitivity_chain_product` all exit 0.

## §4 — Conclusion

Second-order AD on the ARM64 backend is now at parity with x86 for all supported
forms: diagonal Hessians through transcendentals (commit 1) and off-diagonal
cross-Hessians through the multiply operator (commit 2), verified numerically on
real hardware. The `epistemic_hessian_transcendentals` regression test passes
6/6, byte-identical to x86.
