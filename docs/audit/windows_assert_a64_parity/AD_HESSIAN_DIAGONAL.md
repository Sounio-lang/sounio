<!-- docs:meta
topic_id: repo.docs.audit.windows-assert-a64-parity.ad-hessian-diagonal
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.windows-assert-a64-parity.ad-hessian-diagonal
-->

# A64 PARITY — second-order AD: diagonal Hessian + `hessian_of` (commit 1 of 2)

**Opened / closed.** 2026-05-21.
**Status.** RESOLVED — CODE CHANGE LANDED. Diagonal (j==k) Hessian only;
off-diagonal cross-terms + multiply-operator Hessian are commit 2.
**Class.** Codegen on `self-hosted/compiler/lean_single.sio` — `compile_primary_a64`
(new `hessian_of` frontend + second-order math-builtin propagation).
**Branch.** `feat/windows-assert-exit`.
**Predecessor.** `AD_SHADOWS_FIRST_ORDER.md` (first-order), `PRINT_F64_NEGATIVE.md`.

---

## §1 — The gap

`hessian_of(...)` was x86-only, and `compile_primary_a64` did no second-order
Hessian propagation, so `hessian_of(f(x), j, k)` was unavailable on `aarch64-*`.

## §2 — The fix (diagonal scope)

1. **Frontend.** Added a `hessian_of(expr, j, k)` dispatch to `compile_primary_a64`,
   mirroring the sibling `sensitivity_of`: parse the expr + literal j,k, normalise
   j≤k, and read the diagonal Hessian shadow slot (`h00/h11/h22/h33`, also via
   `VAR_HSHADOW_jj` for a bound variable). Off-diagonal (j≠k) resolves to slot −1
   and emits `0.0` — second-order cross-terms are commit 2.
2. **Propagation.** In the math-builtin block, after the first-order chain rule,
   compute `f''(arg)` into a slot, then for each active diagonal channel
   `H_jj(f(g)) = f''(g)·s_j² + f'(g)·H_jj(g)`. f'' per function (x=arg, r=result,
   f'=first deriv): sqrt −f'/(2·arg); exp r; ln −1/arg²; sin −sin(arg); cos −r;
   tan 2·r·f'; atan −2·arg·f'²; tanh −2·r·f'; asin/acos arg·f'³. Negations use
   `0−x` (fsub). All 36 H channels are cleared first; only the 4 diagonals carry
   meaningful values (matching the first-order 4-channel scope).

## §3 — Verification (real Apple M3)

- **Self-host fixed point.** PASS: stage1==stage2==stage3,
  `md5=2ca0e36299e598df335f9700943c12e3`; binary rebuilt.
- **All 10 transcendentals**, `hessian_of(f(x.value),0,0)`, byte-match x86 on M3:
  sqrt −0.707107, exp 1.648721, ln −4.000000, sin −0.479426, cos −0.877583,
  tan 1.418689, atan(1) −0.500000, tanh −0.726862, asin 0.769800, acos −0.769800.
- **x86 non-regression.** `epistemic_hessian_transcendentals`,
  `sensitivity_transcendental`, `sensitivity_trig`, `pbpk_rapamycin_second_order`
  all exit 0.
- **Scope boundary confirmed.** `hessian_of(x*atan(y),0,1)` (off-diagonal product):
  x86 `0.5`, a64 `0.0` — exactly the cross-term commit 2 will add.

## §4 — Conclusion

Diagonal second-order Hessians (univariate second derivatives) now work on the
ARM64 backend for all 10 transcendentals, at parity with x86, verified on real
hardware. **Commit 2** adds off-diagonal Hessian components (h_jk, j≠k) to the
transcendental block and the multiply-operator Hessian, for multivariate
compositions like `x*atan(y)`.
