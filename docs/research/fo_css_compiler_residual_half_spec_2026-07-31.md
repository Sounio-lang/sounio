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

**Status:** L0 + L1 + **L2-fragment** CLOSED; **L2-full** Madaros FO_XFER soundness **OPEN**  
**Algebraic half:** `formal/lean4/SounioFoCssSurfaceParity.lean` (kernel-checked)  
**Semantic bridge:** `formal/lean4/SounioFoSurfaceTransfer.lean`  
**Bytecode fragment:** `formal/lean4/SounioFoBytecodeFragment.lean`  
**Stack gate:** `scripts/ci/fo_residual4_stack_gate.sh`  
**IR evidence (unchanged):** R4 `scripts/ci/fo_pk_import_method_driver_gate.sh`

---

## 1. Residual split (four layers)

| Layer | Claim | Status | Evidence |
|-------|-------|--------|----------|
| **L0 Algebraic** | Pure Rat maps for import/site/method/call-result agree; freezes exact ℚ | **CLOSED** | `SounioFoCssSurfaceParity` + `fo_css_surface_parity_gate` 17/17 |
| **L1 Semantic** | Surfaces desugar to one `FoExpr` AST; FO var is a function of (AST, seeds) | **CLOSED** | `SounioFoSurfaceTransfer` + `fo_surface_transfer_gate` |
| **L2-fragment** | FO bytecode ops 1–6 stack machine: site ≡ import-expanded ≡ method RPN; run = L1 desugar; Var = 191/240 | **CLOSED** | `SounioFoBytecodeFragment` + `fo_bytecode_fragment_gate` |
| **L2-full Compiler** | Madaros `lower.sio` FO_XFER / multipass / multi-mod *emits* the L2-fragment program for the oral-Css fragment | **OPEN** | R4 green gates (numerical); no FO_XFER soundness proof |

Dissertation wording must not collapse L2-fragment into L2-full.

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

## 3b. What would close L2-full (compiler)

A future phase must show, for the oral-Css FO fragment under FO trust ≥42:

1. **Emit faithfulness.** Multi-mod `fo_css`, `Pk.css`, `make_pk(...).css`, and call-site composition lower to FO bytecode equal to `cssSiteProg` (or FO-equivalent under stack semantics).
2. **XFER composition.** Nested pure helper expand (`fo_bc_expand_xfer_call`) preserves FO of the expanded AST.
3. **Multi-mod registration.** Module-frontend FO prepass registers imported pure helpers with the same bytecode as same-module definitions.

**Until then:** R4 remains the L2-full **executable** witness; L2-fragment is the **semantic** FO bytecode model.

---

## 4. Honest non-claims

- No claim that FO_XFER is sound for mutual recursion, effectful fields, or non-const depth.
- No claim that ΣH multi-site load matches solo-path 7.292592.
- L1 does not license “the compiler is correct”; it licenses “the science freezes are the right freezes for any surface that realises this AST.”

---

## 5. Re-run

```bash
bash scripts/ci/fo_residual4_stack_gate.sh         # L0+L1+L2-fragment+R4
# or piecemeal:
bash scripts/ci/fo_css_surface_parity_gate.sh      # L0
bash scripts/ci/fo_surface_transfer_gate.sh        # L1
bash scripts/ci/fo_bytecode_fragment_gate.sh       # L2-fragment
bash scripts/ci/fo_pk_import_method_driver_gate.sh # L2-full executable
# optional:
FO_CSS_LEAN_BUILD=1 bash scripts/ci/fo_bytecode_fragment_gate.sh
```

---

*Spec version fo-css-compiler-residual-half-v2 (2026-07-31) — L2-fragment closed.*
