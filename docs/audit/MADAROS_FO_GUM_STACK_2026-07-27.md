# Madaros FO GUM stack — measured map (2026-07-27)

**Engine:** Madaros (`artifacts/self-hosted/madaros`, rebuild via `scripts/ci/build_modular_madaros.sh`).  
**Discipline:** measure before claim; peel `.value` for FO; do not invent Knowledge⊗Knowledge FO.

## What landed (stack)

| Layer | Capability | Evidence |
|-------|------------|----------|
| Multi-channel FO | ≤32 independent Knowledge seeds; `Var = Σ s_k² σ_k²` | `madaros_gum_fo_*` multichannel / deep_poly |
| Combine rules | +, −, ×, ÷ with diagonal Hessian | lower.sio `fo_combine_sens_*` |
| Interproc | Pure helpers ≤16 params via bytecode | `eight_param`, `sixteen_param`, `interproc`, `let_bytecode` |
| Interproc unary math | Bytecode ops 20+kind (exp/log/sin/…) + chain-rule Hess; FWD skip for math names | `interproc_exp`, `pk_exposure` |
| Nested multi-helper FO | Expand FO-xfer callees into bytecode (param→arg trees; kind 1–6) | `nested_helpers`, `pk_exposure` |
| Knowledge ⊗ Knowledge | GUM value/variance construct + multi-channel FO combine | `knowledge_ops`, `pk_exposure` |
| Multi-pass FO register | Reverse-order pure helpers expand nested FO (4-pass preregister) | `reverse_order` |
| Correlate + H off-diag | `correlate(a,b,ρ)`; Var/E₂ include H_ij Cov_ij (16-ch) | `correlate` |
| Control | `if` SELECT blend (const + runtime) | `div_if`, `if_helper` |
| 2nd-order mean | `E₂[f] ≈ f(μ) + ½ Σ H_kk σ_k²` | `second_order_mean`, `fo_emit_second_order_bias` |
| Hessian diag | `hessian_diag_of(expr) → Σ H_kk` | `pow_const` |
| Unary math FO | exp, log, sin, cos, sqrt, tan, atan, tanh, asin, acos, pow(const) | `transcendental`, `tan_family`, `asin_acos`, `pow_const` |
| Natives | exp/log (xmm0 fix), tan, pow=exp(y log x), asin/acos | `math_native` |
| Composition | `exp(sin)`, `cos(√)`, `log(exp)` | `deep_composition` |
| Field chain | `.value` + let shadow | `field_chain` |

## Trust gate

```bash
# uses existing Madaros ELF
MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash scripts/ci/madaros_gum_fo_trust_gate.sh

# optional rebuild first
SOUNIO_FO_REBUILD=1 bash scripts/ci/madaros_gum_fo_trust_gate.sh
```

Summary JSON lands under `$SOUNIO_FO_TRUST_DIR/summary.json` (or a temp dir printed by the script).

**Measured 2026-07-27 (post correlate FO):** 22/22 PASS under rebuilt Madaros.

## Science driver

`examples/epistemic_fo_second_order/fo_pk_exposure_driver.sio`

Multi-factor Css model with pure helpers (including `clearance_helper` with `exp`), FO variance, second-order mean, and an **honest negative**: `variance_of((kF * kDose).value)` is ~0 while peel `variance_of(f * dose)` is not.

## Science driver measured (2026-07-27)

`examples/epistemic_fo_second_order/fo_pk_exposure_driver.sio` — Css = (F·Dose/τ)/(CL0·exp(η)):

| Quantity | Value |
|----------|------:|
| Css point | 6.666666 |
| Var(rate) | 4.784722 (matches analytic) |
| Var(CL) via `clearance_helper` | 0.340000 (matches analytic **and** inline) |
| Var(Css) via nested `css_helper` | 0.795833 (matches call-site **and** fully inlined) |
| E₂[Css] | 6.724000 (bias +0.057333) |
| Σ H_kk (Css) | 7.292592 |
| `variance_of((kF*kDose).value)` | 689.0 (matches peel `f*dose`) |

## Known holes (do not paper over)

1. **Helper definition order** — nested expand requires callees registered before callers (source order). Reverse-order mutual recursion is not FO-expanded.
3. **Off-diagonal Hessian** is not stored; independent seeds only need diagonal for the Taylor mean (cross terms need Cov≠0).
4. **Correlated seeds** (shared η) are not modelled — treat as separate measures today.
5. **pow native** is `exp(y·log x)`; FO mean also rewrites const-exponent `pow` to `exp(c·log)` in lower.
6. **Method FO** inherits receiver sens only (view-style), not full mangled-method transfer.
7. **No stdlib public wrapper** yet — builtins are compiler-injected (`variance_of`, `second_order_mean`, `hessian_diag_of`).

### Closed this session

- **Interproc FO through unary math inside pure helpers.** Bytecode op `20+uk`; FWD skip for math names. Gate: `interproc_exp`.
- **Correlate + off-diagonal Hessian.** `correlate(a,b,ρ)` sets Cov=ρ·σa·σb between primary FO channels; product/add combine track H_ij (16×16); Var/E₂ pick up cross terms. Gate: `correlate` (ρ=1: E₂[(x+y)²]=9.09, Var=0.09).
- **Multi-pass FO pure-fn registration.** `lowerer_fo_preregister_pure_fns_multipass_mut` (4 passes) before body lower so reverse-order nested helpers expand. Gate: `reverse_order`.
- **Knowledge ⊗ Knowledge FO/GUM.** `lower_knowledge_binary_expr_ref` builds Knowledge with GUM variance and multi-channel FO; `.value` peel preserves FO binds. Gate: `knowledge_ops`.
- **Nested multi-helper FO bodies.** `fo_bc_expand_xfer_call` / `fo_bc_inline_xfer_bytecode` expand kinds 1–6 at compile time (LOAD_PARAM → call-arg subtrees; locals remapped). Gate: `nested_helpers` (css_h and exposure_h depth-2).

## Gate inventory

All files matching `tests/run-pass/madaros_gum_fo_*.sio` are members of the trust gate (22 files including `correlate`). Adding a new FO gate = drop a `madaros_gum_fo_*.sio` with a `MADAROS_GUM_FO_*_PASS` token.

## Next bold moves (ordered)

1. Multi-pass FO registration (tolerate reverse definition order / mutual pure helpers).
3. Off-diagonal H_jk + optional correlation table.
4. stdlib `epistemic::fo` surface wrapping the builtins for dissertation code.
5. Wire `madaros_gum_fo_trust_gate.sh` into composite CI when branch policy allows.

---

*Audit date: 2026-07-27. Numbers in individual gates are re-derivable by re-running each `.sio` under Madaros.*
