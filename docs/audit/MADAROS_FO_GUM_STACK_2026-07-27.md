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
| PK science via import | Oral Css FO through `use epistemic::fo::{fo_css,…}` multi-mod | `pk_import` |
| Method FO | Mangled pure-method transfer; recv = param 0; f64 FO args | `method` |
| Struct field FO | Struct-lit peels → `base_field` keys; alias copy; op17 `self.field` | `struct_field` |
| Nested method FO | `self.rate/clearance` expand inside `self.css` under FO_BC_IMPL_TYPE | `nested_method` |
| Mutual FO expand | Cycle/miss → primary-arg FO; solid kind-6 skip on multipass | `mutual` |
| Field FO via call | Pure struct-return projections + identity FWD field peel | `field_call` |
| Nested field FO | `o.inner.cl0`, let-bind copy, `make(...).inner.cl0` paths | `nested_field` |
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
**Measured 2026-07-28 (PK import science):** 27/27 PASS — `madaros_gum_fo_pk_import` + `examples/.../fo_pk_exposure_import_driver.sio`: imported `fo_css` matches local Css FO (Var=0.795833, E₂=6.724, H=7.292592).  
**Measured 2026-07-28 (method FO):** 28/28 PASS — `madaros_gum_fo_method`: `ops.clearance`/`ops.css` match free-fn FO (Var CL=0.34, Css=0.795833, E₂=6.724).  
**Measured 2026-07-28 (struct field FO):** 29/29 PASS — `madaros_gum_fo_struct_field`: `pk.cl0` matches peel; alias; `pk.clearance(η)` via `self.cl0` = 0.34; exposure product FO live.  
**Measured 2026-07-28 (nested method FO):** 30/30 PASS — `madaros_gum_fo_nested_method`: `pk.css` via nested `self.rate/clearance` matches free/site (Var=0.795833, E₂=6.724).  
**Measured 2026-07-28 (mutual FO):** 31/31 PASS — `madaros_gum_fo_mutual`: free/method mutual pairs at depth-1 match peel (Var=0.0121); DAG top still Var=4.  
**Measured 2026-07-28 (field call FO):** 32/32 PASS — `madaros_gum_fo_field_call`: `make_pk(...).cl0`, `id_pk(pk).cl0`, nested identity match peels.  
**Measured 2026-07-29 (nested field FO):** 33/33 PASS — `madaros_gum_fo_nested_field`: `o.inner.cl0`, let-bind, `make_outer(...).inner.cl0` match peels.

## Science drivers

| Path | File | FO surface |
|------|------|------------|
| Same-module pure helpers | `examples/epistemic_fo_second_order/fo_pk_exposure_driver.sio` | local `css_helper` |
| **Multi-mod import (preferred science)** | `examples/epistemic_fo_second_order/fo_pk_exposure_import_driver.sio` | `use epistemic::fo::{fo_css,…}` |

Both model Css = (F·Dose/τ)/(CL0·exp(η)). Import driver is the multi-mod receipt after FO prepass + Program-by-value fix.

## Science drivers measured (2026-07-27 / 2026-07-28)

Css = (F·Dose/τ)/(CL0·exp(η)) — **same numbers on local and imported helpers**:

| Quantity | Value |
|----------|------:|
| Css point | 6.666666 |
| Var(rate) | 4.784722 (matches analytic) |
| Var(CL) via `fo_clearance` / local | 0.340000 (matches analytic **and** inline) |
| Var(Css) via `fo_css` / nested local | 0.795833 (matches call-site **and** fully inlined) |
| E₂[Css] | 6.724000 (bias +0.057333) |
| Σ H_kk (Css) | 7.292592 |
| `variance_of((kF*kDose).value)` / `fo_mul2` | 689.0 (matches peel `f*dose`) |
| Shared-η `fo_clearance(5,η)·fo_clearance(50,η)` | 2500.0 |

## Known holes (do not paper over)

1. **Helper definition order** — multipass prereg handles reverse *definition* order for pure nested expand; true mutual recursion is still not FO-expanded.
3. **Off-diagonal Hessian storage** — product/add/div *combine* track H_ij when Cov≠0; full symbolic H matrix is not materialised as a first-class value.
4. **~~Shared peel / same σ² channel~~ CLOSED** — `fo_seed_from_variance` reuses channel by variance_reg; peels, lets, Knowledge aliases, and interproc params share correctly. Gate: `shared_channel`. Residual: *independent* `measure`s that *ought* to share a latent still need explicit `correlate(a,b,ρ)` or a single shared peel.
5. **pow native** is `exp(y·log x)`; FO mean also rewrites const-exponent `pow` to `exp(c·log)` in lower.
6. **~~Method FO~~ CLOSED for pure f64-arg methods** — mangled `Type_method` FO_XFER with recv as param 0. Gate: `method`.
6b. **~~Struct field FO~~ CLOSED** — struct-lit inits bind FO to `base_field` keys; field access + alias copy; bytecode op 17 `LOAD_PARAM_FIELD` for `self.field` in method bodies. Gate: `struct_field`.
6b2. **~~Field FO via pure call~~ CLOSED** — `mangle(fn,field)` identity projections for struct-lit returns; identity FWD peels `.field` through args. Gate: `field_call`.
6b3. **~~Nested field FO~~ CLOSED** — recursive struct-lit bind (`o_inner_cl0`); path keys; recursive projections; let-bind of nested struct copies FO. Gate: `nested_field`. Residual: non-pure constructors; >2 field nest on call results.
6c. **~~Nested method FO~~ CLOSED** — FO bytecode expands `ExprMethodCall` via mangled `Type_method` under `FO_BC_IMPL_TYPE` (+ multipass register). Gate: `nested_method`.
6d. **~~Mutual FO expand~~ CLOSED for fixed small depth** — expand stack cycle + missing-callee → primary-arg FO; multipass refreshes only opaque transfers (solid kind-6 kept). Gate: `mutual` (depth-1 even/odd free+method). Residual: deep data-dependent recursion not fully unrolled; primary-arg is an approximation on cycles.
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
- **PK science via multi-mod import.** `fo_pk_exposure_import_driver` + gate `pk_import`: imported `fo_css`/`fo_clearance`/`fo_mul2` match local FO and audit table (Var Css, E₂, H, kprod, shared-η product).
- **Method FO (f64-arg pure methods).** `fo_apply_call_transfer_recv` + xfer arg shift (recv=param0); eval continues past FO-empty self. Gate: `method` (mul2/clearance/css match free-fn).
- **Struct field FO.** `fo_bind_struct_literal_fields` / `fo_load_field_sens` / alias copy; FO bytecode op 17 for `self.field`. Gate: `struct_field` (peel match + method clearance/exposure).
- **Nested method FO.** `FO_BC_IMPL_TYPE` during method register; `fo_bc_expand_xfer_call_recv` for `self.other(...)` with recv=param0. Gate: `nested_method`.
- **Mutual FO expand.** Cycle/miss primary-arg fallback; multipass skips solid non-opaque kind-6 (avoids re-register corruption). Gate: `mutual`.
- **Field FO via pure call.** `fo_register_struct_field_projections` + `lower_expr_field_variance_ref` for Call/MethodCall bases. Gate: `field_call`.
- **Nested field FO.** Recursive bind/projections; `fo_expr_struct_path_key`; call path `make().inner.cl0`. Gate: `nested_field`.

## Gate inventory

All files matching `tests/run-pass/madaros_gum_fo_*.sio` are members of the trust gate (33 files including `nested_field`). Adding a new FO gate = drop a `madaros_gum_fo_*.sio` with a `MADAROS_GUM_FO_*_PASS` token.

## Next bold moves (ordered)

1. Dissertation PK structs: peels on fields + nested methods + shared channel + call/nested projections.
2. Deep recursive FO unroll (beyond depth-1 primary-arg approximation).
3. Keep `correlate` for *distinct* measures that share a latent.
4. Deeper call-result nests (`make().a.b.c`) if science needs them.

---

*Audit date: 2026-07-27. Numbers in individual gates are re-derivable by re-running each `.sio` under Madaros.*
