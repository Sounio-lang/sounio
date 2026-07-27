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

**Measured 2026-07-27 (post unary bytecode):** 18/18 PASS under rebuilt Madaros.

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
| Var(Css) call-site compose | 0.795833 (matches fully inlined) |
| E₂[Css] | 6.724000 (bias +0.057333) |
| Σ H_kk (Css) | 7.292592 |
| `variance_of((kF*kDose).value)` | ~0 (**not FO** — remaining hole) |

## Known holes (do not paper over)

1. **Knowledge arithmetic is not FO.** `a * b` for `Knowledge` values does not seed multi-channel sens; measured peel FO works, product FO does not.
2. **Nested multi-helper FO bodies** (a pure helper whose body only calls other FO helpers, e.g. `css_helper = rate_h / cl_h`) do not yet compile to FO bytecode transfer. **Workaround:** compose helpers at the call site (`infusion_rate(...) / clearance_helper(...)`). Unary math + arithmetic *inside* a single pure helper **does** FO-transfer (`clearance_helper = cl0 * exp(eta)`).
3. **Off-diagonal Hessian** is not stored; independent seeds only need diagonal for the Taylor mean (cross terms need Cov≠0).
4. **Correlated seeds** (shared η) are not modelled — treat as separate measures today.
5. **pow native** is `exp(y·log x)`; FO mean also rewrites const-exponent `pow` to `exp(c·log)` in lower.
6. **Method FO** inherits receiver sens only (view-style), not full mangled-method transfer.
7. **No stdlib public wrapper** yet — builtins are compiler-injected (`variance_of`, `second_order_mean`, `hessian_diag_of`).

### Closed this session

- **Interproc FO through unary math inside pure helpers** (was hole #2). Bytecode compile emits op `20+uk`; eval applies chain-rule sens/hess; simple-transfer path skips FWD to bare `exp`/`log` names so bytecode runs. Gate: `madaros_gum_fo_interproc_exp.sio`.

## Gate inventory

All files matching `tests/run-pass/madaros_gum_fo_*.sio` are members of the trust gate (18 files including `interproc_exp`). Adding a new FO gate = drop a `madaros_gum_fo_*.sio` with a `MADAROS_GUM_FO_*_PASS` token.

## Next bold moves (ordered)

1. FO bytecode for **calls to FO-transferred helpers** inside pure helper bodies (close nested multi-helper hole).
2. FO through `Knowledge` mul/div (seed channels on Knowledge ops, not only `.value`).
3. Off-diagonal H_jk + optional correlation table.
4. stdlib `epistemic::fo` surface wrapping the builtins for dissertation code.
5. Wire `madaros_gum_fo_trust_gate.sh` into composite CI when branch policy allows.

---

*Audit date: 2026-07-27. Numbers in individual gates are re-derivable by re-running each `.sio` under Madaros.*
