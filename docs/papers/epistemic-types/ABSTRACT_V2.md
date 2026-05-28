# Epistemic Particle Physics — Abstract V2 (PL-first rewrite)

**Addresses**: §3.1 (wrong framing — GUM-first instead of PL-first) and §3.8 (3% PDG gap not acknowledged) from cycle-1 review.

---

## Rewritten Abstract (≤250 words)

Physics programs contain hidden approximations. Non-unitary amplitudes, perturbative truncations, and narrow-width assumptions are implicit in every cross-section calculation, yet they are invisible to callers and untracked by any existing type system. The result is that no physics library in any language can prevent a caller from silently treating a Breit-Wigner amplitude as a conventional S-matrix element.

We show that *algebraic effects are the right abstraction for tracking physical approximation in compiled physics programs*. In Sounio, `with NonUnitary` on a function signature is a compile-time obligation: any function in the call chain that extracts a cross-section from a Breit-Wigner propagator must declare the effect, or the compiler rejects the program. The same mechanism generalises to `with Perturbative`, `with NarrowWidthApproximation`, or any other physically-meaningful approximation class. This is not a one-off annotation — it is a *calculus of physical approximations*, composable in the effect algebra the compiler already enforces.

GUM uncertainty propagation (ISO Guide 98-3, delta method) serves as the technical substrate: variance threads through the effect-annotated call chain from PDG input uncertainty to observable variance in a single compile unit. The combined system gives a complete epistemic chain that is statically verified, not recovered at runtime.

We demonstrate the design across 129 tests in 7 suites (784 functions). The `NonUnitary` effect (compiler bit 22) is enforced across Z, W, H, top, and Higgs gg→H propagators. Hessian PDF uncertainty, NLO K-factors with scale variation, and detector material budgets participate in the same type-disciplined framework.

No other physics library in any language enforces physical approximation tracking through algebraic effects at compile time.

---

## Paper Pitch (PL-conference version)

**The one-sentence claim**: We introduce a calculus of physical approximations enforced by algebraic effects at compile time, realised in a self-hosting compiler with 129 verified tests across a complete Standard Model library.

**Why this is a PL paper, not a physics paper**: The physics is the *motivation*. The contribution is the abstraction: algebraic effects as the mechanism for surfacing hidden approximations at the type level. The Standard Model library is the evaluation corpus — chosen because (1) the approximations are precisely known, (2) the PDG provides ground-truth numbers for validation, and (3) the scale (784 functions, 7 suites, gg→tt̄ through detector simulation) demonstrates that the abstraction does not collapse under realistic physics.

**Why existing approaches fail**:
- Runtime uncertainty libraries (e.g., `uncertainties` in Python) propagate values but enforce nothing: a caller can silently drop the uncertainty wrapper.
- Probabilistic programming languages track distributional uncertainty but have no concept of *physical approximation class* — there is no `with NonUnitary` in Stan or Edward.
- Units type systems (F# units-of-measure, Pint) prevent dimension errors but say nothing about approximation regimes.
- Physics frameworks (ROOT, MadGraph) are C++/Python; no effect system, no compile-time enforcement.

**The generalisation argument**: The `NonUnitary` effect is one instance. The compiler bit (22) is one slot. `Perturbative` (next planned extension) would flag any function whose output depends on a finite-order perturbative expansion. `NarrowWidthApproximation` would flag factorised production × decay chains. Each is a physically-meaningful approximation class that currently lives in documentation, not in the type system. Our design moves them into the compiler.

---

## Tree-Level Gap Paragraph (for §Limitations)

### Perturbative order and theory uncertainty

All cross-section calculations in this library are at leading order (LO). The `Epistemic` type tracks *parametric* uncertainty — the propagation of PDG input value uncertainties (masses, couplings, widths) through the computation — but does not currently track theory uncertainty arising from truncation of the perturbative expansion. As a concrete example, `z_total_width_ep()` returns approximately 2.42 GeV against the PDG measured value of 2.4952 GeV; the 3% gap arises from missing NLO QED and electroweak radiative corrections, not from miscalibration of the GUM chain. This theory uncertainty would enter as an additional `Epistemic` term — constructible as `Epistemic::measured(nlo_correction, theory_uncertainty_band)` and propagated through the same GUM machinery — but the NLO correction itself must be supplied by the user, not derived from first principles within the library. The planned `Perturbative` effect (a future compiler extension, not present in the current implementation) would flag functions whose outputs depend on a finite-order truncation, making the theory-uncertainty gap visible at the type level rather than in documentation.
