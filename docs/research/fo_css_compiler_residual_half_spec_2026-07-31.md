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

**Status:** semantic intermediate layer **CLOSED**; Madaros FO_XFER soundness **OPEN**  
**Algebraic half:** `formal/lean4/SounioFoCssSurfaceParity.lean` (kernel-checked)  
**Semantic bridge:** `formal/lean4/SounioFoSurfaceTransfer.lean`  
**Executable cert:** `scripts/ci/fo_surface_transfer_gate.sh`  
**IR evidence (unchanged):** R4 `scripts/ci/fo_pk_import_method_driver_gate.sh`

---

## 1. Residual split (three layers)

| Layer | Claim | Status | Evidence |
|-------|-------|--------|----------|
| **L0 Algebraic** | Pure Rat maps for import/site/method/call-result agree; freezes exact ℚ | **CLOSED** | `SounioFoCssSurfaceParity` + `fo_css_surface_parity_gate` 17/17 |
| **L1 Semantic** | Surfaces desugar to one `FoExpr` AST; FO var is a function of (AST, seeds) | **CLOSED** | `SounioFoSurfaceTransfer` + `fo_surface_transfer_gate` |
| **L2 Compiler** | Madaros `lower.sio` FO_XFER / multipass / multi-mod realises L1 for the oral-Css fragment | **OPEN** | R4 green gates (numerical); no FO_XFER soundness proof |

Dissertation wording must not collapse L1 into L2.

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

## 3. What would close L2 (compiler)

A future phase must show, for the oral-Css FO fragment under FO trust ≥42:

1. **Desugar faithfulness.** Multi-mod `fo_css`, `Pk.css`, `make_pk(...).css`, and call-site composition lower to FO bytecode whose FO channel Jacobian matches `jacCss` at means.
2. **XFER composition.** Nested pure helper expand (`fo_bc_expand_xfer_call`) preserves FO of the expanded AST.
3. **Multi-mod registration.** Module-frontend FO prepass registers imported pure helpers with the same bytecode as same-module definitions (gate `import` already measures this; needs a proof obligation, not a new number).

**Acceptance gate (proposed, not yet built):** a Lean model of FO bytecode ops 0–20+ that interprets to `FoExpr`, plus a hand-translated oral-Css program whose Madaros FO dump matches. Until then, **R4 remains the L2 executable witness**.

---

## 4. Honest non-claims

- No claim that FO_XFER is sound for mutual recursion, effectful fields, or non-const depth.
- No claim that ΣH multi-site load matches solo-path 7.292592.
- L1 does not license “the compiler is correct”; it licenses “the science freezes are the right freezes for any surface that realises this AST.”

---

## 5. Re-run

```bash
bash scripts/ci/fo_css_surface_parity_gate.sh      # L0
bash scripts/ci/fo_surface_transfer_gate.sh        # L1
bash scripts/ci/fo_pk_import_method_driver_gate.sh # L2 executable
# optional:
FO_CSS_LEAN_BUILD=1 bash scripts/ci/fo_surface_transfer_gate.sh
```

---

*Spec version fo-css-compiler-residual-half-v1 (2026-07-31).*
