<!-- docs:meta
topic_id: repo.docs.research.fo-css-compiler-residual-half-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.fo-css-compiler-residual-half-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# FO Css residual §5.4 — compiler half: semantic bridge (2026-07-31)

**Status:** L0 + L1 + L2-fragment + **L2 pure-emit** CLOSED; **L2 registration/multipass** OPEN  
**Algebraic half:** `formal/lean4/SounioFoCssSurfaceParity.lean`  
**Semantic bridge:** `formal/lean4/SounioFoSurfaceTransfer.lean`  
**Bytecode fragment:** `formal/lean4/SounioFoBytecodeFragment.lean`  
**Pure emit:** `formal/lean4/SounioFoEmitPure.lean` (`fo_bc_compile_expr` pure fragment)  
**Stack gate:** `scripts/ci/fo_residual4_stack_gate.sh`  
**IR evidence:** R4 `scripts/ci/fo_pk_import_method_driver_gate.sh`

---

## 1. Residual split (five layers)

| Layer | Claim | Status | Evidence |
|-------|-------|--------|----------|
| **L0 Algebraic** | Pure Rat maps agree; freezes exact ℚ | **CLOSED** | `SounioFoCssSurfaceParity` + gate 17/17 |
| **L1 Semantic** | Surfaces desugar to one `FoExpr` | **CLOSED** | `SounioFoSurfaceTransfer` |
| **L2-fragment** | FO ops 1–6 stack machine; RPN run = desugar | **CLOSED** | `SounioFoBytecodeFragment` |
| **L2 pure-emit** | `fo_bc_compile_expr` pure fragment emits `cssSiteProg`; fo_css expand = site | **CLOSED** | `SounioFoEmitPure` + `fo_emit_pure_gate` |
| **L2 registration** | Multipass / multi-mod / method FO_XFER always feed that pure AST into compile | **OPEN** | R4 numerical witness |

Dissertation wording must not collapse pure-emit into full registration soundness.

---

## 2. L1 theorem (semantic)

For surfaces \(s \in \{\mathrm{Import},\mathrm{Site},\mathrm{Method},\mathrm{CallResult}\}\):

\[
\mathrm{desugar}(s) = E_{\mathrm{Css}}
\quad\Rightarrow\quad
\mathrm{foVar}(s) = \mathrm{foVarFromJac}(E_{\mathrm{Css}},\sigma)
\]

with \(E_{\mathrm{Css}} = (F\cdot\mathrm{Dose}/\tau)/(\mathrm{CL}_0\cdot e^\eta)\) as an FO AST, and

\[
\mathrm{foVarFromJac} = \sum_i \Bigl(\frac{\partial E}{\partial x_i}\Bigr)^2 \sigma_i^2
= \frac{191}{240}
\]

at default seeds. Surface independence is **AST identity**, not compiler IR identity.

---

## 3. L2-fragment (closed 2026-07-31)

Madaros FO bytecode header (`lower.sio`):
`OP_PARAM=1 CONST=2 ADD=3 SUB=4 MUL=5 DIV=6`.

Oral Css RPN (after pure-helper XFER expand of `fo_css` / method bodies):

```
PARAM F; PARAM Dose; MUL; PARAM τ; DIV; PARAM CL0; PARAM eEta; MUL; DIV
```

Lean stack machine interprets this to `desugarSite` (`native_decide`).
Import-expanded and method programs are definitionally the same RPN list.

**Honest scope:** this proves *if* Madaros emits this RPN, FO semantics match L1.
It does **not** prove Madaros emits it — that is L2-full.

## 3b. L2 pure-emit (closed 2026-07-31)

Formalises Madaros `fo_bc_compile_expr` pure path (`lower.sio` ~9358–9367):

```
compile(param i)     = [PARAM i]
compile(mul a b)     = compile a ++ compile b ++ [MUL]
compile(div a b)     = compile a ++ compile b ++ [DIV]
```

Proved: `compile(cssSite) = cssSiteProg`, and `fo_css` XFER expand
(`rate/clearance` bodies from `stdlib/epistemic/fo.sio`) is definitionally
`cssSite`. Round-trip `run(compile e) = toFo e` for oral Css.

## 3c. What would close L2 registration (remaining)

1. **Multipass registration** always installs pure-helper bodies as FO_XFER kind-6 for `fo_css` / methods before call-site expand.
2. **Multi-mod prepass** registers imported helpers with the same bytecode as same-module.
3. **Method FO_XFER** mangles `Pk_css` and peels `self.cl0` to param FO matching param-3 in the pure model.
4. Optional: EXP op-20 path for `exp(η)` vs eEta channel (FO-var equivalent at η=0).

**Until then:** R4 remains the **executable** registration witness.

---

## 4. Honest non-claims

- No claim that FO_XFER is sound for mutual recursion, effectful fields, or non-const depth.
- No claim that ΣH multi-site load matches solo-path 7.292592.
- L1 does not license “the compiler is correct”; it licenses “the science freezes are the right freezes for any surface that realises this AST.”

---

## 5. Re-run

```bash
bash scripts/ci/fo_residual4_stack_gate.sh         # full stack
# or piecemeal:
bash scripts/ci/fo_css_surface_parity_gate.sh      # L0
bash scripts/ci/fo_surface_transfer_gate.sh        # L1
bash scripts/ci/fo_bytecode_fragment_gate.sh       # L2-fragment
bash scripts/ci/fo_emit_pure_gate.sh               # L2 pure-emit
bash scripts/ci/fo_pk_import_method_driver_gate.sh # registration executable
```

---

*Spec version fo-css-compiler-residual-half-v3 (2026-07-31) — L2 pure-emit closed.*
