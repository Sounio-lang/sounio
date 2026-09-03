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

**Status:** **ORAL_CSS_CLOSED** (all fragment layers); **L2 full engine** OPEN  
**Closeout:** `docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`  
**Stack gate:** `scripts/ci/fo_residual4_stack_gate.sh`  
**IR evidence:** R4 `scripts/ci/fo_pk_import_method_driver_gate.sh`

---

## 1. Residual split (oral Css closed)

| Layer | Claim | Status | Evidence |
|-------|-------|--------|----------|
| **L0 Algebraic** | Pure Rat freezes | **CLOSED** | `SounioFoCssSurfaceParity` |
| **L1 Semantic** | FoExpr desugar | **CLOSED** | `SounioFoSurfaceTransfer` |
| **L2-fragment** | FO ops 1–6 stack | **CLOSED** | `SounioFoBytecodeFragment` |
| **L2 pure-emit** | pure `fo_bc_compile_expr` | **CLOSED** | `SounioFoEmitPure` |
| **L2 registration** | FO_XFER expand | **CLOSED** | `SounioFoRegistrationFragment` |
| **L2 engine-install** | multipass register | **CLOSED** | `SounioFoEngineInstallFragment` |
| **L2 method peel** | Pk.css / call-result peel | **CLOSED** | `SounioFoMethodXferFragment` |
| **L2 multi-mod model** | local ≡ import ≡ union registry | **CLOSED** | `SounioFoMultimodFragment` |
| **L2 full engine** | arbitrary programs | **OPEN** | R4 |

Oral Css science claim is **closed**; full engine is out of dissertation scope.

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

## 3c. L2 registration fragment (closed 2026-07-31)

Models multipass FO_XFER expand (`fo_bc_expand_xfer_call` / PARAM→arg subst):

```
Registry: fo_infusion_rate, fo_clearance, fo_css  (local ≡ import name-identity)
expand(fo_css(p0..p4)) = cssSite
compile(expand(...)) = cssSiteProg
method peel model = cssSite
```

## 3d. L2 engine-install fragment (closed 2026-07-31)

Models multipass register of the oral Css pure-helper fragment:

```
installPass(registry, items)  -- append each pure item body
multipass(items, k)           -- k sweeps (Madaros uses 4)
-- reverse definition order still installs all three in one pass
expand(install(...), fo_css(p0..p4)) = cssSite
compile(...) = cssSiteProg
```

## 3e. What would close L2 full engine (remaining)

1. Madaros multipass is sound for arbitrary pure programs (not just this fragment).
2. Multi-mod module loader + FO prepass for all loaded modules.
3. Method FO_XFER + op17 field peel in full generality.
4. Optional: EXP op-20 vs eEta channel.

**Until then:** R4 remains the **executable** full-engine witness for oral Css.

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
bash scripts/ci/fo_registration_fragment_gate.sh   # L2 registration fragment
bash scripts/ci/fo_engine_install_fragment_gate.sh # L2 engine-install fragment
bash scripts/ci/fo_method_xfer_fragment_gate.sh    # L2 method peel
bash scripts/ci/fo_multimod_fragment_gate.sh       # L2 multi-mod model
bash scripts/ci/fo_pk_import_method_driver_gate.sh # full-engine executable
```

---

*Spec version fo-css-compiler-residual-half-v6 (2026-07-31) — oral Css residual-4 closed.*
