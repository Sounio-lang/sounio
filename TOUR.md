# A measured tour of Sounio

Twenty claims, each one a program you can run, each one checked by the
sentinel it prints. Nothing on this page was carried over from an earlier
document; every timing and result was measured on 2026-10-04 at `main`
(Linux x86-64, the only platform the compiler targets).

```bash
bash scripts/tour.sh            # sections 1-3, about 1 minute
bash scripts/tour.sh --full     # adds the slow simulations and the compiler rebuild: 16 claims, about 4 minutes
bash scripts/tour.sh --full --lean   # adds the 4 Lean 4 checks of section 5: all 20 claims (needs elan; toolchains are pinned per lakefile)
```

Last run with `--full --lean`: **passed 20, failed 0, 4 min 13 s.** `--full` alone
prints `passed 16`; the other four claims are the Lean checks, which run only with
`--lean`.

Section 6 lists what does **not** work, with issue numbers. Read it before
drawing conclusions from sections 1–5.

---

## What Sounio is, in one paragraph

A self-hosted compiler for a language whose values can carry their own
uncertainty (`Knowledge<T>`, propagated to first order by the GUM rules),
their physical units, and declared effects (`IO`, `Mut`, `Div`, `Panic`, `GPU`,
…). It compiles to static x86-64 ELF. The compiler is written in Sounio and
rebuilds itself to a fixed point starting from the committed prebuilt compiler
(`bin/souc-linux-x86_64`, section 4). The original C seed, `bootstrap/stage0.c`,
is not on that path; `scripts/ci/bootstrap_chain_gate.sh` exercises it. Two engines
share the front end: **Madaros**, the modular default behind `bin/souc`, and
**lean_single**, the bootstrap seed, selected with `SOUNIO_SOUC_ENGINE=lean_single`.
Where they differ, this page says which one ran.

---

## 1. The epistemic core

| claim | program | time |
|---|---|---|
| uncertainty propagates analytically by the GUM, at compile-time-known cost | `demo_incerteza.sio` | 3 s |
| derived units (`unit velocity = m / s;`) type-check; `m + s` is rejected | `demo_unidades.sio` (lean_single) | <1 s |
| a `Knowledge<Patient where { age >= 18 }>` built with age 13 aborts before the next line runs | `demo_portas_rejeicao.sio`, via `scripts/ontology/expand_knowledge_runtime_guards.sh` | 3 s |

The last two come with caveats that matter (section 6, #2751–#2753).

## 2. Knowledge graphs: a verified EL+ reasoner, executed

`formal/OntologyELPlusClosureComplete.lean` proves, for **any** EL+ TBox over
finite signatures, that the boolean saturation oracle answers *true* **if and
only if** the subsumption is derivable (`subBPlusC_iff`: soundness from the
closure invariant, completeness from the canonical model). `lake build
OntologyELPlusClosureComplete` takes 11 s and uses no `sorry`.

`stdlib/ontology/elplus.sio` is the executable mirror of that saturation. Every
demo below runs it on a different domain and checks its own inferences:

| domain | program | time |
|---|---|---|
| role-aware closure on a SNOMED-shaped TBox | `examples/ontology_elplus_closure_demo.sio` | 2 s |
| Pericarditis ⊑ ∃finding_site.Heart, through a `part_of` role chain the old BFS missed | `examples/ontology/biomedical/snomed_elplus_demo.sio` | 3 s |
| W3C PROV-CONSTRAINTS inferences written as EL+ role axioms | `examples/epistemic/prov_elplus_demo.sio` | 3 s |
| VIM3 metrological traceability, an "unbroken chain of calibrations", as role composition | `examples/epistemic/traceability_elplus_demo.sio` | 2 s |
| drug–drug interaction on ChEBI ids (smoke test, not a screening service) | `examples/clinical/ddi_elplus_demo.sio` | 3 s |
| pharmacogenomic diplotype → phenotype → safety (smoke test) | `examples/clinical/pgx_elplus_demo.sio` | 4 s |
| closure, then epistemic repair, then query, end to end | `examples/ontology_pipeline_demo.sio` | 3 s |

**What the proof does and does not cover.** It covers the saturation algorithm
in Lean. The `.sio` file is a hand-written mirror, not extracted code, so the
link between the two is by construction and by test, not by proof.

## 3. Simulation: hydrogen chemistry, against external oracles

| claim | program | engine | time |
|---|---|---|---|
| the seven-stage metal-hydride compressor of Gkanas et al. (2020) is bounded by its last stage's desorption plateau; computed from the paper's measured Table 3 with **no fitted parameter**, the bound orders the paper's six simulated delivery pressures in 15/15 pairs, and the S6→S7 driving force orders its cycle times in 15/15 pairs | `demos/hydrogen/mh7_coupled_ceiling.sio` | Madaros | 2 s |
| van 't Hoff extrapolation of an equilibrium constant, with an honest interval instead of a point | `demos/hydrogen/vanthoff_gate.sio` | Madaros | 3 s |
| methanation log K: naive extrapolation against the interval | `demos/hydrogen/methanation_logk_gate.sio` | Madaros | 3 s |
| H₂–brine–calcite network over 30 years, with GUM bands and p-boxes; it catches an interior maximum that corner-only interval propagation would miss | `demos/hydrogen/uhs_brine_calcite.sio` | lean_single | 50 s |
| GRI-Mech 3.0 hydrogen ignition with uncertainty | `examples/chemistry/h2_ignition_uq_demo.sio` | Madaros | 134 s |

Each result has an independent oracle outside the language:

- **Cantera 3.2 for GRI-Mech 3.0.** The full 53-species mechanism agrees within the
  oracle's own measured resolution. The headline of the cross-validation is that
  **the implementation was right and its test oracle was wrong**: the Python
  replica added parameter uncertainty in quadrature at every step, so its band
  scaled with √dt, while Sounio's native band is step-size invariant. The reported
  parity gap was one rounded gas constant, not the integrator, and a reported
  1 bar vs 1 atm defect was checked and does not exist. It is frozen with a DOI:
  [`Sounio-lang/sounio-gri30-crossvalidation`](https://github.com/Sounio-lang/sounio-gri30-crossvalidation),
  v1.0.3, [10.5281/zenodo.22263060](https://doi.org/10.5281/zenodo.22263060).
- **C++23 re-implementations written from the protocol**, not translated from the
  `.sio` files: `benchmarks/chemistry/cpp/` and `demos/hydrogen/tools/`.
- **A pre-registered falsification study.** In
  [`Sounio-lang/sounio-uhs-coupled`](https://github.com/Sounio-lang/sounio-uhs-coupled),
  `bash tools/verify.sh` rebuilds nothing and trusts nothing. It checks that the
  vendored compiler's md5 matches the pin, then reproduces 11 captured outputs
  byte for byte. That compiler rebuilds bit-identically from its two-month-old
  source commit in 7.5 s.

## 4. Self-hosting

```bash
make build      # prebuilt bin/souc-linux-x86_64 -> gen1 -> gen2 -> gen3; checks gen2 == gen3
```

32 s on `main`, `FIXED POINT OK (4b6b478d…)`. A fixed point is what makes a
pinned compiler reproducible years later from its source commit, which section 3's
last bullet relies on.

## 5. Lean 4

Besides the EL+ closure, the hydrogen demos have machine-checked companions:
`formal/lean4/SounioHydrogenPbox.lean`, `SounioHydrogenReceipt.lean` and
`SounioHydrogenVanthoff.lean`, with 30 theorems in all, no `sorry`, about 1 s
each. The repository holds 255 Lean files under `formal/lean4/`; this tour runs
only the ones it cites.

## 6. What does not work, measured the same day

| area | state | tracking |
|---|---|---|
| **units, soundness** | on lean_single, `let v: velocity = distance * elapsed` (m·s into m/s) is **accepted** and prints a wrong-dimension number. `+` of incompatible units is rejected, `*` into a derived-unit `let` is not | [#2751](https://github.com/Sounio-lang/sounio/issues/2751) |
| **units, default engine** | Madaros rejects `let d: m = 100.0` (E001), so the units demo runs only on lean_single | [#2752](https://github.com/Sounio-lang/sounio/issues/2752) |
| **`where` refinements** | `souc run` accepts `Knowledge<T where {…}>` and **does not enforce it** on either engine. Enforcement today is a source-to-source script | [#2753](https://github.com/Sounio-lang/sounio/issues/2753) |
| **engine divergence** | with the lowering fix of #2750, three chemistry tests that pass on lean_single fail at run time on Madaros (illegal instruction, wrong result, arena exhaustion), and so does the UHS demo | `demos/hydrogen/README.md` |
| **GPU** | `souc build --backend gpu` emits valid PTX, but only for kernels with **empty bodies**: no arithmetic, no memory access. It is an emission skeleton, not a backend | `examples/kernel_vec_add.sio` |
| **first-order uncertainty across calls** | first-order variance channels do not cross user function calls | KL-11, `docs/compiler/KNOWN_LIMITATIONS.md` |
| **variance channel, lean_single shim** (measured 2026-10-06) | `scripts/ci/souc-seq-leansingle.sh` runs `bin/souc-linux-x86_64`, a 2026-06-16 snapshot. It drops the Knowledge variance at `.value`, so `variance_of` returns 0: `rapamycin_rk4_budget` and `rapamycin_epistemic_adaptive` refuse with `EPISTEMIC_FABRICATION`, and `rapamycin_iso_budget` printed `var=0` and still passed until P0.5. It also returns wrong non-zero values in `gum_fo_imported_div_variance` (0.1 vs 0.045) and `madaros_gum_fo_interproc` (0.0025 vs 0.01). Madaros and the current seed `bin/souc-lean-single-x86_64` keep the channel | `docs/dissertation/pbpk_claim_truth_table.md` (suite gate: engine per test) |
| **variance through struct fields, lean_single** | both lean_single ELFs return variance 0 for a `Knowledge` read back through a struct field, a constructor `let`, or a field-returning call. Madaros keeps it. Witnesses: `gum_fo_field_chain_variance`, `madaros_gum_fo_{struct_field,deep_field,nested_field,field_call,impure_ctor,nonpure_ctor,let_ctor}`. On lean_single each prints its own `*_FAIL` marker **and exits 0** | `tests/run-pass/` (those files) |
| **variance through `if`, lean_single** | `madaros_gum_fo_div_if`: the `if` branch gives variance 0 (Madaros 0.0325), and the quotient's variance is 0.0025 where 0.000401 is expected. Exits 0 with `MADAROS_GUM_FO_DIV_IF_FAIL` | `tests/run-pass/madaros_gum_fo_div_if.sio` |
| **correlated first-order, lean_single** | repeated use of one input is treated as independent. `gum_correlated` gives var(h·h) = 0.000613, where Madaros gives 0.001225 = 4h²σ². The seed's `rapamycin_iso_budget` path (a) is about 0.55× Budget64, where Madaros agrees to 0.2 % | `tests/run-pass/gum_correlated.sio` |
| **one-hop struct-field variance, both engines** | `madaros_wide_struct_variance_field`: one_hop = two_hop = 0 on Madaros as well as on lean_single (direct = 0.09) | `tests/run-pass/madaros_wide_struct_variance_field.sio` |
| **performance** | not benchmarked on this page. The one same-algorithm comparison measured during this tour's preparation was dominated by the model code, not the compiler, so no number is given | — |
| **platform** | Linux x86-64 only. The binaries are static ELF and do not run on macOS | — |

If you find something that belongs in this table, it is the most useful issue
you can open.
