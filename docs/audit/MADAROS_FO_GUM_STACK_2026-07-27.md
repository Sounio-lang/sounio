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
| Quotient off-diag H | FO div combine tracks H_ij (with Cov) | `div_correlate` |
| stdlib `epistemic::fo` | Pure FO-transfer helpers (css/clearance/mul/div) | `stdlib_surface` |
| Multi-mod import FO | Prepass registers pure FO_XFER from *all* loaded modules; Program-by-value load avoids A14 nested field-address residual (`&programs[i].items` → None) | `import` |
| Shared FO channel | Same σ² reg / same peel reuses one channel (η+η → 4σ²; CL·V shared η) | `shared_channel` |
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

**Measured 2026-07-27 (post div-offdiag + stdlib fo):** 24/24 PASS under rebuilt Madaros.  
**Measured 2026-07-28 (multi-mod import FO):** 25/25 PASS — `madaros_gum_fo_import` closes imported pure-helper FO (v_imp = v_peel for mul/div/css).  
**Measured 2026-07-28 (shared channel freeze):** 26/26 PASS — `madaros_gum_fo_shared_channel` freezes σ²-reg channel identity (already live; no lowerer change).

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

1. **Helper definition order** — multipass prereg handles reverse *definition* order for pure nested expand; true mutual recursion is still not FO-expanded.
3. **Off-diagonal Hessian storage** — product/add/div *combine* track H_ij when Cov≠0; full symbolic H matrix is not materialised as a first-class value.
4. **~~Shared peel / same σ² channel~~ CLOSED** — `fo_seed_from_variance` reuses channel by variance_reg; peels, lets, Knowledge aliases, and interproc params share correctly. Gate: `shared_channel`. Residual: *independent* `measure`s that *ought* to share a latent still need explicit `correlate(a,b,ρ)` or a single shared peel.
5. **pow native** is `exp(y·log x)`; FO mean also rewrites const-exponent `pow` to `exp(c·log)` in lower.
6. **Method FO** inherits receiver sens only (view-style), not full mangled-method transfer.
7. **FO builtins remain compiler-injected** (`variance_of`, …) — stdlib `epistemic::fo` wraps *pure helpers*, not the builtins themselves (AST-walk requirement).

### Closed this session

- **Interproc FO through unary math inside pure helpers.** Bytecode op `20+uk`; FWD skip for math names. Gate: `interproc_exp`.
- **Quotient off-diag + stdlib fo surface.** `fo_combine_sens_div` tracks H_ij; `stdlib/epistemic/fo.sio` pure helpers; CI wires `madaros_gum_fo_trust_gate.sh`.
- **Correlate + off-diagonal Hessian.** `correlate(a,b,ρ)` sets Cov=ρ·σa·σb between primary FO channels; product/add combine track H_ij (16×16); Var/E₂ pick up cross terms. Gate: `correlate` (ρ=1: E₂[(x+y)²]=9.09, Var=0.09).
- **Multi-pass FO pure-fn registration.** `lowerer_fo_preregister_pure_fns_multipass_mut` (4 passes) before body lower so reverse-order nested helpers expand. Gate: `reverse_order`.
- **Knowledge ⊗ Knowledge FO/GUM.** `lower_knowledge_binary_expr_ref` builds Knowledge with GUM variance and multi-channel FO; `.value` peel preserves FO binds. Gate: `knowledge_ops`.
- **Nested multi-helper FO bodies.** `fo_bc_expand_xfer_call` / `fo_bc_inline_xfer_bytecode` expand kinds 1–6 at compile time (LOAD_PARAM → call-arg subtrees; locals remapped). Gate: `nested_helpers` (css_h and exposure_h depth-2).
- **Multi-mod imported pure-helper FO.** Module-frontend FO prepass walks all loaded `Program`s before seed body lower. Nested field-address `&programs[i].items` was A14 residual (Option::None → FO_XFER empty); fix: `var prog = programs[i]; fo_preregister(&prog.items)`. Gate: `import` (v_imp_mul=v_peel_mul=0.25; css match).
- **Shared FO channel identity (freeze).** Measured live without new lowerer work: same σ² reg reuses channel; η+η, CL·V shared η, Knowledge alias, interproc double-use all match analytic; independent measures stay 2-channel. Gate: `shared_channel`.

## Gate inventory

All files matching `tests/run-pass/madaros_gum_fo_*.sio` are members of the trust gate (26 files including `import`, `shared_channel`). Adding a new FO gate = drop a `madaros_gum_fo_*.sio` with a `MADAROS_GUM_FO_*_PASS` token.

## Next bold moves (ordered)

1. **Method FO** — full mangled-method transfer (not just receiver sens).
2. **Science driver on `epistemic::fo` imports** — multi-mod is green; PK driver should call stdlib `fo_css` and match local-helper numbers.
3. **Mutual pure-helper FO expand** — true recursive pure helpers (beyond multipass DAG).
4. Keep `correlate` for *distinct* measures that share a latent in the science model.

---

*Audit date: 2026-07-27. Numbers in individual gates are re-derivable by re-running each `.sio` under Madaros.*
