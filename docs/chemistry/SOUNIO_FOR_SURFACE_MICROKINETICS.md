<!-- docs:meta
topic_id: repo.docs.chemistry.sounio-for-surface-microkinetics
authority: repo_only
audience: users
last_validated: 2026-09-09
validated_by: chemistry-surface-microkinetics
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.chemistry.sounio-for-surface-microkinetics
-->

# Surface microkinetics in Sounio, and what the language actually adds

**Audience:** a hydrogen group evaluating whether a new language is worth
their time. **Written for:** Dr. Emmanuel Stamatakis's group at NCSR
Demokritos, alongside `demos/hydrogen/README.md`, which covers the
metal-hydride compression and hydrogen-valley side. This file covers the
subsurface chemistry side.

**Everything below was measured on 2026-09-09 and every number names the
binary that produced it.** Where something does not work, it says so.

---

## 1. What this is not

Sounio is not a lattice kinetic Monte Carlo code and does not compete with
one. `stdlib/chemistry/surface.sio` is mean-field microkinetics: coverages
are continuous fractions, there is no lattice, no spatial correlation, no
adsorbate–adsorbate lateral interaction beyond site blocking. Anything
requiring the spatial statistics of a real surface needs a KMC code, and
this is not one.

It DOES do dimensional analysis, as of this branch, on the compiler you
would install -- which is a correction to what this document said on
2026-09-09, when that was the sharpest limitation listed here. A rate
declared `mol m^-2 s^-1` is checked against the arithmetic that produces
it, and two amounts of species declared disjoint cannot be added. What
that cost, and the defect it exposed in the process, is in S5.

What it does is make a microkinetic model **checkable in ways a script is
not**, and the rest of this document is the evidence for that claim,
including the four defects the checking found in this repository's own
merged code.

---

## 2. Site conservation, decided rather than assumed

Every microkinetic model rests on the site balance
`Σθᵢ + θ_* = 1`. In a conventional implementation this is an invariant the
author maintains by hand: the free-site fraction is computed as
`1 - sum(theta)` wherever it is needed, and nothing checks that the
mechanism's stoichiometry is consistent with it. A step that consumes one
site and returns two — a plausible transcription slip in a mechanism with
a dozen elementary steps — shows up as a slow drift in total coverage that
a plot at the wrong scale hides completely.

`surface.sio` gives the free site **its own index in the state vector and
its own column in the stoichiometry matrix**. Site conservation then stops
being a convention and becomes an arithmetic property:

```
sites are conserved  ⟺  every row of nu sums to zero
```

`sites_conserved()` decides it. `site_balance_holds()` checks the dynamic
counterpart on a state. And the property is proved, in Lean 4, as an **iff**
rather than an implication:

```lean
theorem sites_conserved_iff (a b c v : Rat) :
    (∀ r : Rat, totalSiteChange a b c v r = 0) ↔ a + b + c + v = 0
```

The reverse direction is the part that matters for trusting the checker: a
mechanism that fails the test leaks sites **at some rate**, so nothing is
being let through. A checker with only the forward direction could pass a
leaky mechanism. `test_sites_conserved_detects_a_leak` asserts both
directions on a concrete mechanism — a checker that cannot fail is not a
checker.

---

## 3. The hydrogen result: a reaction order without a rate constant

The pre-registered UHS study in the sibling repository decomposes abiotic
H₂ loss into four channels and reports of channel (c), abiotic methanation
CO₂ + 4H₂ → CH₄ + 2H₂O:

> `K(T)` is 1e28 to 1e37 over this temperature range, so the reaction is
> enormously favourable and the barrier is **entirely kinetic**. A rate is
> needed and **not supplied by any source in this study**.

That module refuses to emit a total abiotic band rather than treat the
unmeasured channels as zero. A rate constant cannot be invented.

**But the rate constant is not the only thing a mechanism determines, and
it is not the thing an operator most needs.** How the loss scales with
storage pressure follows from the mechanism alone — specifically from
whether H₂ adsorbs dissociatively. `examples/chemistry/h2_surface_reaction_order.sio`
derives it, integrating the elementary steps and differencing:

| p(H₂) | dissociative, measured | analytic | molecular, measured | analytic |
|---|---|---|---|---|
| 1e-4 | +0.490196 | +0.490196 | +0.999802 | +0.999802 |
| 1e-2 | +0.409909 | +0.409910 | +0.980391 | +0.980392 |
| 1    | +0.002488 | +0.002488 | +0.004975 | +0.004975 |
| 1e2  | −0.408264 | −0.408265 | −0.980001 | −0.980002 |
| 1e4  | −0.490001 | −0.490001 | −0.999798 | −0.999798 |

Two results, neither needing the missing constant:

1. **In the dilute limit the order is ½, not 1.** Doubling the storage
   pressure multiplies this loss channel by √2 ≈ 1.41, not by 2. A model
   that treats H₂ as adsorbing molecularly overstates how fast the channel
   grows with pressure, across the whole range a reservoir cycles over.
2. **Past a maximum the order goes negative.** Hydrogen crowds CO₂ off the
   surface and the reaction needs both, so beyond the peak, raising the
   pressure *lowers* this loss rate.

The ½ is not an assumption anywhere in the code. It emerges from `θ²`
against `v²` in the dissociative rate law, because a diatomic that
dissociates occupies two sites. It is proved in Lean without square roots
by working in the ratio `θ = s·v`:

```lean
theorem dissociative_isotherm_iff (kf kr p s v theta : Rat)
    (hv : v ≠ 0) (hratio : theta = s * v) :
    kf * p * v ^ 2 = kr * theta ^ 2 ↔ kf * p = kr * s ^ 2
```

---

## 4. Three independent routes to the same number

The repository's rule is that an oracle is never a translation of the code
it checks. For the dual-site Langmuir–Hinshelwood rate, three routes exist
and are compared at the same conditions in
`tests/stdlib/chemistry/test_surface_stdlib.sio`:

| route | what it is |
|---|---|
| `chemistry::catalysis::rate_law_eval` kind 5 | a closed form from a rate-law taxonomy, written months earlier |
| `chemistry::surface::lh_dual_site_rate` | a closed form derived as a mechanism's limit |
| `chemistry::surface::simulate_surface_mkm` | four elementary steps, RK4 to steady state |

The two closed forms agree to **1e-18**. The integrated mechanism lands on
the same number without being given the formula.

`stdlib/chemistry/oracles/surface_oracle.cpp` adds a fourth, and its
independence is **methodological rather than merely lexical**: it never
integrates anything. It solves the algebraic steady state
`νᵀ r(θ) = 0` directly, by damped Newton with a numerically differenced
Jacobian and Gaussian elimination, using the site-conservation row to
restore the rank the conservation law removes. Two methods with different
error structure — one has a time-discretisation error and no linear
algebra, the other has conditioning and no time step — reaching the same
coverages is evidence about the *physics*. Two RK4 implementations agreeing
would mostly be evidence about RK4.

And the reason they must agree is a theorem, not a coincidence:

```lean
theorem lh_quasi_equilibrium (k KA pA KB pB thetaA thetaB v : Rat)
    (hA : thetaA = KA * pA * v) (hB : thetaB = KB * pB * v)
    (hsum : thetaA + thetaB + v = 1) :
    k * thetaA * thetaB * (1 + KA * pA + KB * pB) ^ 2
      = k * (KA * pA) * (KB * pB)
```

**Read that precisely, because an adversarial math-review made us tighten
it.** What is proved is the exact *algebraic consequence* of imposing
quasi-equilibrium and the site balance — nothing is approximated, so the
two independently written closed forms are the same object rather than two
unrelated models. It is **not** a limit theorem: no limit is taken, no
residual is bounded, and no ODE trajectory is shown to approach this state.
Whether a real mechanism sits near the quasi-equilibrium point is a
numerical question, and `test_lh_reduces_to_closed_form` *measures* the
residual gap that opens when the surface reaction is fast enough to drain
the adsorbates. Measured, not proved.

Nine theorems, zero `sorry`, `#print axioms` naming only Lean's own three.

### The review is part of the evidence, not a footnote

The repository requires an adversarial math-review before a Lean file can be
committed. On this one, `mistral-large` passed all eight theorems that
existed then with no correction, and `xai/grok-4.6` passed all eight
**proofs** while rejecting four **docstring** claims and one redundant
hypothesis. All five were correct and all five were fixed:

- the Langmuir identity was called equivalent to `θ = Kp/(1+Kp)`, which
  needs `kr ≠ 0` — the degenerate `kf = kr = 0` satisfies both hypotheses
  for every `θ`, making the identity vacuously true where the quotient is
  meaningless;
- `kr·s² = kf·p` was read as `s = √(Kp)`, which over ℚ it is not: `kr` may
  vanish, `Kp` need not be a square, and both `s` and `−s` solve it;
- the coverage `θ = √(Kp)/(1+√(Kp))` was asserted while the file proved
  only its two halves and never assembled them — **a ninth theorem,
  `dissociative_coverage`, exists because of this review**;
- "coverage grows as the square root of pressure" holds only while `s ≪ 1`;
  `s/(1+s)` flattens toward 1 at saturation;
- `lh_quasi_equilibrium_div` carried a nonvanishing hypothesis that the site
  balance already forces.

The proofs were sound throughout, which is exactly why the prose needed
checking separately. Each correction is recorded at its own site in the file
rather than silently applied.

---

## 5. What does not work, measured

### Unit types now survive arithmetic on the default engine

**This section used to open by calling dimensional analysis the sharpest
limitation here, and that is no longer true.** The table below replaces the
one written on 2026-09-09; the old one is kept underneath it because the
difference is the point.

Measured 2026-09-10 on `artifacts/self-hosted/madaros-l9`, built from this
branch:

| operation | Madaros, this branch |
|---|---|
| `unit + unit`, same brand | accepted |
| `unit × f64` | accepted |
| `unit + unit`, **incompatible dimensions** | **rejected, E041** |
| `unit ÷ unit` against a declared derived unit | accepted |

What changed, and it was three separate defects:

1. **`ItemUnit` was never collected by the checker spine the default engine
   runs.** Every `unit X;` declaration was inert. Waking it took the unit
   system from 3 of 14 unit fixtures passing to 10, and from 4 of 15 to 11.
2. **The call-argument dimension check compared registry INDICES, not
   dimensions.** A quantity derived by arithmetic -- `distance / time` --
   gets an anonymous interned entry that never equals the named
   `unit velocity = m / s` it matches, so the diagnostic printed
   `argument unit: length^1*time^-1 / parameter unit: length^1*time^-1` and
   rejected it anyway. Both the boundary and the compatibility path now
   compare dimensions, using the same addability test the binary operator
   uses, so they agree by construction.
3. **Waking the units then BROKE that check, silently.** With `mg` and
   `second` both resolving to `TyF64`, a fast path keyed on `TypeKind` alone
   skipped all five call-argument boundary checks. `needs_mg(t: second)` is
   rejected by the committed baseline, was ACCEPTED after the units commit,
   and is rejected again now -- with E041, the real dimension error, rather
   than the E001 an unresolved name produced by accident.

The third is the one worth carrying into any evaluation of this language: the
checks are real, and turning a subsystem on can switch another one off
without a single test failing. See
`scripts/ci/checker_spine_parity_gate.sh`, which exists to make that class of
silence impossible, and
`docs/audit/SWEEP_STACK_LIMIT_CORRECTION_2026-09-09.md`, which corrects three
commit messages on this branch whose sweep numbers were measured under a stack
limit this project does not use.

**The old table, 2026-09-09, for contrast:**

| operation | Madaros v0.80.0 | lean_single seed | gen3.elf (unmerged) |
|---|---|---|---|
| `unit + unit`, same brand | rejected | accepted | accepted |
| `unit × f64` | rejected | accepted | accepted |
| `unit + unit`, incompatible | rejected | **accepted — unsafe** | rejected |
| `unit ÷ unit` | rejected | accepted | accepted |

`surface.sio` is still written without unit types, because it is meant to run
on the engines that ship on `main` and this work is not merged there yet.
`examples/hydrogen/h2_verified_surface_rate.sio` is the same physics written
WITH the rate declared as `mol m^-2 s^-1` and H2 and CO2 declared disjoint,
and it lists the four mistakes the compiler now refuses, each produced by
compiling the program rather than quoted from memory.

### Two limits on how far a model can be integrated

Both are properties of the generated runtime and both were found by
running out of memory, not by reading documentation:

- **A 2 GiB arena that never reclaims.** `MatNM` carries `data: [f64; 4096]`
  — 32 KiB per value regardless of the declared rows/cols — and
  `catalysis.sio` allocates about twenty per derivative evaluation, four
  evaluations per RK4 step. Measured ~2.7 MiB per step, ~800 steps per
  process; bisected at 750 running and 850 exhausting. `surface.sio` uses a
  flat `[f64; 256]` instead and does not have this cost.
- **A handle table.** A returned `[f64; 16]` is a heap handle, not a stack
  copy, so a by-value derivative function allocates eight handles per RK4
  step. Measured `madaros: handles full` after roughly 2.5 integrations of
  40000 steps. `integrate_to_steady_state()` hoists every buffer out of the
  loop and writes through `&![f64; 16]`, allocating nothing inside it.

### The largest models still do not link

`tests/stdlib/chemistry/test_kinetics_core.sio`,
`demos/hydrogen/uhs_brine_calcite.sio` and
`demos/hydrogen/site_screening.sio` now type-check cleanly on Madaros but
stop at `multimodule native thin-link compilation failed` — a codegen scale
limit on large import graphs. They remain `//@ check-only` for that reason,
and their headers say which reason.

---

## 6. What the checking actually caught

This is the part that is hard to argue with, because it is not a claim
about what a type system could find. Pointing the default compiler at
chemistry code that had been merged, trusted, and cited took
`stdlib/chemistry/kinetics.sio` from 61 type errors to 0, and three of them
were real defects the other engine accepted silently:

| defect | what it was |
|---|---|
| **stdlib visibility** | `constants/physical.sio` declared 23 physical constants and **zero** `pub`, while six chemistry and physics modules imported and called them. A constants module nobody could legally use. |
| **an engine-specific workaround breaking the other engine** | `ontology.sio` carried a `&str`/`&string` workaround written *for* lean_single's checker, and that workaround was precisely what Madaros rejected. It is what made the two UHS demos lean_single-only. |
| **an out-of-bounds read** | `simulate_big_crn` returns `[f64; 6]`; it was passed to a `&[f64; 8]` parameter that copies all eight. Two elements past the end of the array, on every call. |
| **an effect-system violation** | `stdlib/ode/rk4.sio` declared its callback `with Mut` while every right-hand side handed to it is `with Mut, Div, Panic`. An effect system that lets a panicking callback through a `with Mut` hole is not enforcing anything. |

Regression-swept across all 50 files calling `rk4_step` or `rk4_integrate`:
37 passing became 39, and the two changes are both improvements. Nothing
regressed.

**And a wrong number.** `kinetics.sio::test_enzyme_crn` asserted a product
concentration of ≈0.7 where the true value is 0.154054667769 — wrong by a
factor of 4.5, and analytically impossible (Michaelis–Menten bounds it at
≈0.165). It had a second defect that explains the first's survival: the
check sat in statement position as dead code, so only a different assertion
was returned. It survived merge because its driver was `//@ check-only` —
type-checked, never executed.

That is the cost of a test that is checked and never run, and it is the
argument for the rest of this document in one sentence.

---

## 7. Reproducing everything here

```bash
# the surface module and its nine tests
bin/souc run stdlib/chemistry/surface.sio                     # SURFACE ALL PASS
bin/souc run tests/stdlib/chemistry/test_surface_stdlib.sio   # SURFACE_STDLIB_OK

# the hydrogen reaction-order result
bin/souc run examples/chemistry/h2_surface_reaction_order.sio

# the catalysis suite, no longer check-only
bin/souc run tests/stdlib/chemistry/test_catalysis_stdlib.sio          # CATALYSIS_STDLIB_OK
bin/souc run tests/stdlib/chemistry/test_catalysis_turnover_stdlib.sio # CATALYSIS_TURNOVER_OK

# the independent C++23 oracle (Newton, no time integration)
g++ -std=c++23 -O2 -o /tmp/surface_oracle stdlib/chemistry/oracles/surface_oracle.cpp
/tmp/surface_oracle                                            # SURFACE_ORACLE ALL PASS

# the formal proofs
cd formal/lean4 && lake build SounioSurfaceKinetics
```

Provenance of every number above: Madaros v0.80.0,
`bin/madaros-linux-x86_64`, md5 `92f2c5664c6c6a8d8ba7c36bc0d503b8`, the
committed gate-receipted ELF (receipt
`artifacts/self-hosted/madaros.gate-receipt`, `source_commit=67caef4b74`),
on `origin/main` @ `ed1fa84234`. The surface module and its driver were
additionally run under `gen3.elf` (md5 `bdcdb9d642783187251fd469e3a56d73`)
and produce identical output; they type-check clean under the lean_single
seed.

Full engine measurement, with the bisections behind every limit quoted
here: `docs/audit/CHEMISTRY_MADAROS_ENGINE_MEASUREMENT_2026-09-09.md`.
