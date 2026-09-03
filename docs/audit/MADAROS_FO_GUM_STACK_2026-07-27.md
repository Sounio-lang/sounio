<!-- docs:meta
topic_id: repo.docs.audit.madaros-fo-gum-stack-2026-07-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-fo-gum-stack-2026-07-27
-->

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
| Deep call field FO | Arbitrary-depth `make().c.b.a.x` path_suffix peel | `deep_field` |
| PK struct e2e | Peels + nested methods + shared channel + call projections | `pk_struct` |
| Method FO Call-result recv | `make_pk(...).css(...)` + let-alias of Call-return via self.field | `method_call_recv` |
| Free-fn struct-arg field FO | `clearance_of(p,η)` / `css_of(make_pk(...),…)` via `p.cl0` | `freefn_field` |
| Non-identity pure ctor FO | `make_dbl(x).cl0` / `make_scaled(x,s).cl0` projected bytecode | `nonpure_ctor` |
| Correlate shared latent | distinct `measure`s joined by `correlate(a,b,ρ)` match shared peel | `correlate_latent` |
| Deep mutual FO | Const-depth specializations `f_d0..f_d8` for `n<=0` / `n<0.5` chains | `mutual_deep` |
| Let-bound pure ctor FO | `let y=x*2; Pk{cl0:y}` projections compile lets + field | `let_ctor` |
| Impure ctor pure-field FO | Mut constructors still FO field slices of pure inits | `impure_ctor` |
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
**Measured 2026-07-30 (deep call field FO):** 35/35 PASS — `madaros_gum_fo_deep_field`: `make_d(...).c.b.a.x` (3 mids), identity nest, method-return nests match peels (Var=0.09).  
**Measured 2026-07-30 (PK struct e2e):** 35/35 PASS — `madaros_gum_fo_pk_struct`: peels + nested methods + shared channel + call/nested projections on a dissertation-shaped `Pk` (Css Var=0.795833, E₂=6.724, CL=0.34).  
**Measured 2026-07-30 (method FO Call-result recv):** 36/36 PASS — `madaros_gum_fo_method_call_recv`: `make_pk(...).clearance/css`, let-alias of Call-return, `id_pk(make_pk(...)).clearance` match free/peel (CL=0.34, Css=0.795833, E₂=6.724).  
**Measured 2026-07-30 (free-fn field + nonpure ctor + correlate latent):** 39/39 PASS — `freefn_field` (`clearance_of`/`css_of` on lit/call/alias), `nonpure_ctor` (`make_dbl`/`make_scaled` projected FO), `correlate_latent` (two independent η measures + ρ=1 match shared peel; PK-shaped product).  
**Measured 2026-07-30 (mutual deep + let ctor + impure ctor):** 42/42 PASS — `mutual_deep` (f/g n=0..5 parity peels), `let_ctor` (let-bound field projections + alias), `impure_ctor` (Mut ctor pure-field FO + let).

## Science drivers

| Path | File | FO surface |
|------|------|------------|
| Same-module pure helpers | `examples/epistemic_fo_second_order/fo_pk_exposure_driver.sio` | local `css_helper` |
| **Multi-mod import (preferred free helpers)** | `examples/epistemic_fo_second_order/fo_pk_exposure_import_driver.sio` | `use epistemic::fo::{fo_css,…}` |
| **PK struct e2e (trust gate)** | `tests/run-pass/madaros_gum_fo_pk_struct.sio` | `Pk` peels, `css` nested methods, call projections |
| **Pk-method dissertation driver** | `examples/epistemic_fo_second_order/fo_pk_struct_method_driver.sio` | methods + call-result + alias + free parity + shared η + correlate ρ=1 |
| Gate | `scripts/ci/fo_pk_struct_method_driver_gate.sh` | greps science table; token `FO_PK_STRUCT_METHOD_DRIVER_PASS` |
| **ρ-sweep + τ FO companion** | `examples/epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio` | exposure ρ∈{0,0.5,1} + call-result; Css with σ_τ=0.5 |
| Gate | `scripts/ci/fo_pk_struct_rho_tau_driver_gate.sh` | 1575 / 2200 / 2825; Css_τ=0.872993 > fixed 0.795833 |
| **Multi-dose τ-series** | `examples/epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio` | Css(τ∈{8,12,24}); Var∝1/τ²; kel=CL/V (η cancels) |
| Gate | `scripts/ci/fo_pk_struct_multidose_driver_gate.sh` | 10/6.67/3.33; Var 1.79/0.796/0.199; kel Var=5.2e-5 |
| **Import ↔ method parity** | `examples/epistemic_fo_second_order/fo_pk_import_method_driver.sio` | `epistemic::fo` multi-mod = Pk methods = call-result = site |
| Gate | `scripts/ci/fo_pk_import_method_driver_gate.sh` | all Css FO 0.795833; E₂ 6.724; CL 0.34; rate 4.784722 |

Oral Css = (F·Dose/τ)/(CL0·exp(η)). Import driver is the multi-mod free-helper receipt.  
**Pk-method driver** (2026-07-31): Var Css=0.795833, E₂=6.724, CL=0.34, exposure shared=2825 / indep=1575 / ρ=1=2825.  
**ρ-τ companion** (2026-07-31): exposure Var = 1575 + 1250·ρ (ρ=0/0.5/1 → 1575/2200/2825); Css with σ_τ=0.5 → Var=0.872993 (E₂=6.735574).  
**Multi-dose** (2026-07-31): Css(8/12/24)=10/6.667/3.333; Var=1.790625/0.795833/0.198958 (∝1/τ²); kel=0.1 with Var=0.000052.  
**Import↔method** (2026-07-31): multi-mod `fo_css`/`fo_clearance`/`fo_infusion_rate` bit-agree with Pk methods, call-result, and call-site composition on Css/CL/rate FO and E₂.  
**AUC + half-life R5** (2026-08-01): `fo_pk_struct_auc_thalf_driver.sio` — AUC Var=114.6, \(E_2=80.688\); \(t_{1/2}\) Var=0.249835 (= peel); kel Var=5.2e-5; method = call-result = free = site. Gate: `fo_pk_struct_auc_thalf_driver_gate.sh`.  
**R5b import parity:** `stdlib/epistemic/fo.sio` adds `fo_auc`/`fo_kel`/`fo_thalf`/`fo_volume`; multi-mod bit-agrees with methods (`fo_pk_import_auc_thalf_driver_gate.sh`).  
**Rac + residual fraction R6** (2026-08-01): `fo_pk_struct_rac_driver.sio` — \(f_{\mathrm{rem}}=0.301195\), Var=0.000679; Rac=1.431014, Var=0.002848, \(E_2=1.434130\); method = peel. Gate: `fo_pk_struct_rac_driver_gate.sh`.  
**R6b import parity:** `fo_rac`/`fo_frac_rem` multi-mod freezes match method/peel (`fo_pk_import_rac_driver_gate.sh`). Rac FO is heavy — drivers budget ≤5–6 FO sites (multi-site residual silent-exit beyond that).  
**Cmax/Cmin/PTF R7** (2026-08-01): `fo_pk_struct_cmax_driver.sio` — Cmax=11.448115, Var=2.050059, \(E_2=11.539124\); Cmin point=3.448115; PTF=1.2, Var=0.007488. Gate: `fo_pk_struct_cmax_driver_gate.sh`.  
**R7b:** `fo_cmax`/`fo_cmin`/`fo_ptf` multi-mod; Cmin Var=0.306096 on import (`fo_pk_import_cmax_driver_gate.sh`). ≤3 FO sites when stacking Cmax-class FO.  
**Fraction of SS R8** (2026-08-01): `fo_pk_struct_fss_driver.sio` — \(f_{\mathrm{ss}}(3)=0.972676\), Var=0.000050, \(E_2=0.971912\); \(n_{90}=1.918820\), Var=0.019145. Gate: `fo_pk_struct_fss_driver_gate.sh`.  
**R8b:** `fo_fss`/`fo_n90` multi-mod = method = peel (`fo_pk_import_fss_driver_gate.sh`).  
**PTR + DOF R9** (2026-08-01): `fo_pk_struct_ptr_driver.sio` — PTR=3.320113, Var=0.082541, \(E_2=3.338918\); DOF=2.320113, Var=0.082541. Gate: `fo_pk_struct_ptr_driver_gate.sh`.  
**R9b:** `fo_ptr`/`fo_dof` multi-mod = method = peel (`fo_pk_import_ptr_driver_gate.sh`).  
**MRT + t90 R10** (2026-08-01): `fo_pk_struct_mrt_driver.sio` — MRT=10, Var≈0.52, \(E_2=10.036\); \(t_{90}=23.025850\), Var=2.756987, \(E_2=23.108743\). Gate: `fo_pk_struct_mrt_driver_gate.sh`.  
**R10b:** `fo_mrt`/`fo_t90` multi-mod = method (peel MRT ULP 0.520000 vs 0.519999) (`fo_pk_import_mrt_driver_gate.sh`).  
**Loading dose R11** (2026-08-01): `fo_pk_struct_ld_driver.sio` — LD=715.507200, Var=916.939959, \(E_2=717.065032\); \(f_e=0.698804\), Var=0.000679 (=R6 f_rem). Gate: `fo_pk_struct_ld_driver_gate.sh`.  
**R11b:** `fo_ld`/`fo_fe` multi-mod = method = peel (`fo_pk_import_ld_driver_gate.sh`).  
**AUC_τ SS R12** (2026-08-01): `fo_pk_struct_auct_driver.sio` — AUC_τ=80, Var=114.6, \(E_2=80.688\) (= Css·τ FO); Css Var=0.795833. Gate: `fo_pk_struct_auct_driver_gate.sh`.  
**R12b:** `fo_auc_tau`/`fo_css_tau`/`fo_auc` multi-mod bit-agree (`fo_pk_import_auct_driver_gate.sh`).  
**Algebraic residual-4 closeout** (2026-07-31): pure surfaces definitionally equal + exact-ℚ FO freezes in
`formal/lean4/SounioFoCssSurfaceParity.lean`; executable certificate
`scripts/ci/fo_css_surface_parity_gate.sh` (17/17).  
**Semantic bridge L1** (2026-07-31): surfaces desugar to one `FoExpr`; FO var is AST-determined —
`formal/lean4/SounioFoSurfaceTransfer.lean` + `scripts/ci/fo_surface_transfer_gate.sh`.  
**L2-fragment** (2026-07-31): FO bytecode ops 1–6 stack machine — `SounioFoBytecodeFragment`.  
**L2 pure-emit** (2026-07-31): `fo_bc_compile_expr` pure fragment —
`SounioFoEmitPure` + `fo_emit_pure_gate.sh`.  
**L2 registration fragment** (2026-07-31): multipass FO_XFER expand —
`SounioFoRegistrationFragment` + `fo_registration_fragment_gate.sh`.  
**L2 engine-install fragment** (2026-07-31): multipass register —
`SounioFoEngineInstallFragment`.  
**L2 method peel + multi-mod model** (2026-07-31): `SounioFoMethodXferFragment`,
`SounioFoMultimodFragment`.  
**Oral Css residual §5.4 CLOSED** — closeout
`docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`.  
**Science R1–R4 + residual-4 stack** re-validated green 2026-07-31.  
**L2 full engine** (arbitrary programs) remains OPEN. Stack:
`scripts/ci/fo_residual4_stack_gate.sh`.  
**Dissertation index:** [`docs/research/fo_pk_method_science_receipts_2026-07-31.md`](../research/fo_pk_method_science_receipts_2026-07-31.md) — R1–R4 tables, re-run commands, citation sketch.

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
6b3. **~~Nested field FO~~ CLOSED** — recursive struct-lit bind (`o_inner_cl0`); path keys; recursive projections; let-bind of nested struct copies FO. Gate: `nested_field`.
6b4. **~~Deep call-result field FO~~ CLOSED** — recursive `fo_resolve_projected_fo` peels FieldAccess leftward to Call/MethodCall with arbitrary-depth path_suffix mangle (removes mid1/mid2 ceiling). Gate: `deep_field` (`make().c.b.a.x`, identity nest, method-return nests). Residual: non-pure constructors.
6b5. **~~PK struct e2e~~ CLOSED for Ident-bound Pk** — peels, nested methods, shared channel, call/nested field projections on a dissertation-shaped `Pk`. Gate: `pk_struct`.
6b6. **~~Method FO Call-result receiver~~ CLOSED for pure struct-return constructors** — op17 `LOAD_PARAM_FIELD` loads FO via `fo_load_field_sens_of_expr` (Ident binds **or** Call/MethodCall projected path); `let pk = make_pk(...)` binds projected FO onto `pk_*` via `fo_bind_call_result_field_fo`. Gate: `method_call_recv`.
6b7. **~~Free-fn struct-arg field FO~~ CLOSED** — op17 resolves free-fn `args[n].field` (not only method recv); Call/alias args via projected FO. Gate: `freefn_field`.
6b8. **~~Non-identity pure ctor field FO~~ CLOSED for pure field inits** — `fo_register_struct_lit_projections_at` compiles FO bytecode for non-param field exprs (`x*2`, `x*s`) under `mangle(fn,field)`. Gate: `nonpure_ctor`. Residual: let-bound field inits in ctor bodies (`let y = x*2; Pk{cl0:y}`) still miss projection.
6b9. **~~Correlate distinct measures sharing a latent~~ FROZEN** — science path is explicit `correlate(a,b,ρ)` (or a single shared peel). Gate: `correlate_latent` (ρ=1 matches shared-peel Var; PK-shaped CL·V product).
6b10. **~~Deep mutual FO~~ CLOSED for const depth 0..8** — multipass registers `name_d{d}` with `FO_SPECIALIZE_DEPTH` folding `n<=0` / `n<0.5` / `n<1` if-chains; else-arm expands `callee_d{d-1}`; call-site host-const IntLit selects specialization. Gate: `mutual_deep` (f/g n=0..5). Residual: runtime (non-const) depth still primary-arg on cycle; SELECT both-arms.
6b11. **~~Let-bound pure ctor FO~~ CLOSED** — field projections compile preceding lets then field expr (`fo_bc_compile_lets_then_expr`). Gate: `let_ctor`.
6b12. **~~Impure ctor pure-field FO~~ FROZEN** — Mut/IO on constructor does not block FO of pure field inits (identity / computed / let-bound). Gate: `impure_ctor`. No claim that FO tracks the effectful channel.
6c. **~~Nested method FO~~ CLOSED** — FO bytecode expands `ExprMethodCall` via mangled `Type_method` under `FO_BC_IMPL_TYPE` (+ multipass register). Gate: `nested_method`.
6d. **~~Mutual FO expand~~ CLOSED for fixed small depth + const deep** — expand stack cycle + missing-callee → primary-arg FO at generic register; const-depth specializations close deep unrolls. Gate: `mutual` + `mutual_deep`.
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
- **Deep call-result field FO.** `fo_resolve_projected_fo` + path_suffix mangle (no mid1/mid2 ceiling); identity multi-level peel. Gate: `deep_field`.
- **PK struct e2e.** Dissertation-shaped `Pk`: peels + nested methods + shared channel + call projections. Gate: `pk_struct`.
- **Method FO Call-result receiver.** op17 projected FO + let-bind of pure Call-return fields. Gate: `method_call_recv`.
- **Free-fn struct-arg field FO.** op17 free-arg path. Gate: `freefn_field`.
- **Non-identity pure ctor FO.** Projected bytecode for computed field inits. Gate: `nonpure_ctor`.
- **Correlate shared latent (freeze).** Distinct measures + `correlate` match shared peel. Gate: `correlate_latent`.
- **Deep mutual FO (const depth).** `f_d0..f_d8` specializations. Gate: `mutual_deep`.
- **Let-bound pure ctor FO.** Lets + field projection. Gate: `let_ctor`.
- **Impure ctor pure-field FO (freeze).** Mut ctors keep field FO. Gate: `impure_ctor`.

## Gate inventory

All files matching `tests/run-pass/madaros_gum_fo_*.sio` are members of the trust gate (42 files). Adding a new FO gate = drop a `madaros_gum_fo_*.sio` with a `MADAROS_GUM_FO_*_PASS` token.

## Next bold moves (ordered)

1. Runtime (non-const) mutual depth — still primary-arg on cycle under SELECT both-arms.
2. Depth specialization beyond 8 / non-integer host depths.
3. FO of effectful field values themselves (not in scope: impure_ctor only freezes pure field slices).

---

*Audit date: 2026-07-27. Numbers in individual gates are re-derivable by re-running each `.sio` under Madaros.*
