<!-- docs:meta
topic_id: repo.docs.papers.gri30-cantera-crossvalidation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.gri30-cantera-crossvalidation
-->

# The instrument concealed the defect: ten ways a cross-validation against Cantera certified the errors it was built to catch

**Demetrios Chiuratto Agourakis**
ORCID [0009-0001-8671-8878](https://orcid.org/0009-0001-8671-8878) · demetrios@agourakis.med.br

**Preprint.** Target venue: SSRN. Version of 2026-09-03, last revised
2026-09-23. Both the 2026-09-22 and 2026-09-23 revisions are **post-version
correction passes** against reviewer findings, not re-measurements of the
chemistry: the first fixes the reproduction commands (§2.7, §8.2) and
re-verifies the Lean development's build path and axiom dependencies (§5.5);
the second, driven by an external adversarial review (§9), clarifies which
producer measured which residual figure where the manuscript had cited two
(§4.2/§4.5) without naming them, and scopes the Lean-theorem claim of §5.4–5.5
to the checkpoint it was actually measured at. Every change in both passes is
dated inline and marked **[W]** where it corrects a prior statement. No
parity, residual, band-scaling or instrument-count *value* from the
2026-09-03 version changed in either pass — what changed is which of two or
three already-published values a given sentence is licensed to call "the"
result, and the scope of one claim about what the Lean development
establishes.

**Data and code:** frozen snapshot `Sounio-lang/sounio-gri30-crossvalidation`,
concept DOI [10.5281/zenodo.22236607](https://doi.org/10.5281/zenodo.22236607)
(resolves to the latest version); the version cited throughout is **v1.0.3**,
DOI [10.5281/zenodo.22263060](https://doi.org/10.5281/zenodo.22263060).

**Measurement record:** every number in this manuscript is transcribed from
`benchmarks/chemistry/RESULTS.md` in the upstream repository `Sounio-lang/sounio`,
where it appears beneath the command that produced it. Unless a sentence says
otherwise, that command was run at commit
`98aa8e4d5151bbc61815bf910b6c31c3d0789f5f` on 2026-09-01; numbers dated
2026-09-02, 2026-09-03, 2026-09-22 or **2026-09-23** are re-measurements,
newly-added verifications or reviewer-driven corrections, and are marked in
place, each with its own command. Nothing here is carried forward from an
earlier log, a prior session or a draft.

**Notation used throughout.** **[B]** marks a claim bounded by the instrument —
it may not be stated without its bound. **[W]** marks a retraction or
correction that is part of the result rather than an erratum to it. **[U]**
marks a quantity left explicitly unexplained.

---

## Abstract

A published parity gap of 2.2 × 10⁻⁷ to 2.7 × 10⁻⁶ between a Sounio chemistry
module, a fixed-step Python replica and Cantera 3.2 on the GRI-Mech 3.0 H/O
submechanism, attributed in the source documentation to fixed-step RK4 versus
CVODE, is shown here to be one rounded activation-energy gas constant: aligning
`R_cal` from the CHEMKIN-conventional 1.9872041 to Cantera's own
8.31446261815324/4.184 = 1.9872042586408316 improves every species by 155,031×
to 295,899× in an isolated substitution experiment — all eight post-alignment
figures sit below the oracle's single-state floor, and three of them sit
below the lowest floor value measured anywhere in its ensemble sweep, so none
is citable as resolution, only as measured **[B]**. The truncation hypothesis
is falsified directly rather than by
extrapolation: halving the step at the operating point moves the deviation by a
ratio of 1.000 on all eight species, and the replica's self-convergence there is
2.7 × 10⁻¹⁵ to 2.2 × 10⁻¹⁴, eight orders below the gap. After alignment the
residual is 2.074 × 10⁻¹¹, but the oracle's own resolution, measured over an
ensemble of initial states perturbed by one part per million, spans 3.730 ×
10⁻¹² to 4.142 × 10⁻¹¹ — so the residual lies inside the instrument's noise band
and is reported as a bound, not as agreement **[B]**. Separately, this
project's most-repeated numerical claim ("majors 0.2–2 %, radicals ~3 %,
H2O2 ~16 %" — no citation count is claimed for it) is shown to be the
per-species error profile of a historical reverse-rate defect,
reproduced here at 0.21 % / 0.18 % / 3.44 % / 16.17 %, mislabelled as a parity
table **[W]**. A second result concerns uncertainty composition: the Python and
C++23 replicas give an uncertainty band that scales as √dt (ratio 1.9999–2.0084
under a factor 4 in step), while the Sounio native band is step-invariant
(0.999999–1.000001); the √N understatement law behind that contrast is
machine-checked in Lean 4 (15 theorems, zero `sorry`), and here the replicas'
per-step-independent quadrature is shown architecturally wrong for a
persistent parameter — the evidence is ratio-only, so it does not by itself
establish that the Sounio band's magnitude is correct **[W]**. The organising
finding is
methodological: in ten separate instances the instrument built to detect a
defect had less resolution than the defect, and the reading was then attributed
to the object rather than to the instrument.

**Keywords:** combustion kinetics; GRI-Mech 3.0; Cantera; cross-validation;
uncertainty quantification; JCGM combination law; measurement resolution;
reproducibility.

> **[W] Corrected 2026-09-23, a reviewer finding.** This keyword line
> previously read "GUM" — a term the body never once uses, defines, or
> connects to anything. What §5.5 actually contains is narrower and precise:
> theorems about which *combination law* (additive vs. quadrature) JCGM
> eq. 13 licenses under a given correlation structure, not a full GUM
> uncertainty budget with a stated coverage factor or expanded uncertainty.
> The keyword now names what is there.

---

## 1. Introduction

### 1.1 What a cross-validation is, and which side is the oracle

A cross-validation of a numerical implementation is an asymmetric instrument.
One side is designated the **oracle** — taken as correct for the purpose of the
comparison — and the other is the **object under test**. The asymmetry is rarely
stated, and it is not free: whatever error the oracle carries is attributed to
the object, and whatever the oracle cannot resolve is invisible in both.

In the work reported here, Cantera 3.2.0 with CVODE is the oracle for the
*trajectory*, and two independent replicas (Python and C++23) are diagnostics.
For the *uncertainty band*, §5 shows that the designation has to be reversed
for the replicas' **accumulation rule**: their per-step-independent quadrature
is architecturally wrong for a persistent parameter. **[W] Corrected
2026-09-23, a reviewer finding.** That does not license reversing the
designation wholesale to "the implementation under test is right" — the
evidence for the Sounio module is ratio-only (step-invariance under a
factor-4 dt change) and does not establish its band's *magnitude*, per §7.5's
own Threats to Validity note. A single project therefore contains both
orientations *and* a partial one, and the reader should not carry any of them
over to the others.

### 1.2 The claims in circulation

Four statements about this system were in circulation before the measurements
reported here, and are quoted in their original form because their wording is
part of the finding. **Their source, stated rather than left implicit:** the
task dispatch that commissioned this cross-validation work, issued by the
project's operator to a prior working session. That dispatch is not a tracked
repository artefact — no commit, path or URL identifies it, and none is
invented here to look like one. §3.2 records the same absence for the
`reac − nu` code commit itself, which the same investigation found does not
exist under any ref; claims (A)–(D) are of a different kind — testimony about
what was asked, not a lost code state — and are sourced no further than that
for the same reason: the source is outside this repository's history.

- **(A)** "majors within 0.2–2 %, radicals ~3 %, H2O2 ~16 %", attributed to
  fixed-step RK4 versus CVODE;
- **(B)** "agreement to 5–6 significant figures, deviations 2 × 10⁻⁷ to
  6 × 10⁻⁶";
- **(C)** "H2O2 agrees to 5.9 × 10⁻³ relative";
- **(D)** "factor 4 in dt, ratio √2", and "the underestimation law √(T/dt) is
  exact for dt = 10⁻⁸ at T = 10⁻⁶, 10⁻⁵, 10⁻⁴".

(A) and (B) are mutually incompatible as descriptions of the same checkpoint —
four orders of magnitude apart — which is what prompted the work. The resolution
is that they describe different things: §3 shows (A) is a real measurement of a
real defect wearing the wrong label, and §4 confirms (B) and refutes its stated
cause. (C) is contradicted by three orders of magnitude. (D) conflates two
sweeps and overstates a law's domain by more than a decade.

### 1.3 What this work does

It measures rather than reads. Each number carries the command that produced it
and the commit at which that command was run, and a committed auditor
(`audit_provenance.py`, §8.3) exits non-zero — **FAIL** — when a section's own
command block names a file that is not in the released tree, is placeholder
prose rather than a runnable command, or names no file at all. **[W] Corrected
2026-09-23, a reviewer finding.** A section that reports numbers but carries no
command block of its own — because it relies on a command given earlier in the
document — is counted separately as **INHERIT** and does not fail the
auditor's exit code; that case still needs a reader to locate the inherited
command, which the auditor cannot verify automatically. The auditor's actual
coverage is therefore narrower than "exits non-zero whenever a section lacks a
producer in the tree" states: it catches a *wrong* or *absent-from-tree*
producer, not the weaker case of *no producer stated at all*. That auditor was
written after — and because — a published table in this very work turned out
to have been measured from a working copy that was never committed (§6,
instance 10, third row).

### 1.4 Contributions

1. **A closed attribution.** The published parity gap is derived analytically
   from one constant and confirmed by substitution, with the improvement
   measured per species (§4.2).
2. **A direct falsification** of the truncation hypothesis at the step actually
   used, rather than by extrapolating a coarse-step convergence law (§4.3).
3. **The oracle's resolution as an interval, not a number** — the resolution of
   an adaptive integrator is not a smooth function of the initial state, and
   admits no value independent of the state it is run from (§4.5) **[B]**.
4. **Ten instances** of a single methodological pattern: the instrument had less
   resolution than the defect it was asked to measure (§6).
5. **A frozen, DOI-bearing snapshot as the unit of publication**, with an
   offline schema check of the archive metadata that is itself a result (§6,
   instance 9; §8).

### 1.5 Order of presentation

The results are presented as a descent through the instrument, each section
moving the defect one layer inward: §3 treats a defect in the *object* found by
an external oracle; §4 a defect in the *convention constants* shared by object
and oracle; §5 a defect in the *oracle itself*, where the replica's accumulation
rule is shown architecturally wrong, though the implementation under test's
band magnitude is not thereby validated (§7.5). §6 then treats the
instruments, which is where the generalisable content of the work lies.

---

## 2. Methods

### 2.1 Mechanism and protocol

Two mechanisms are used: the GRI-Mech 3.0 **H/O submechanism** (10 species, 29
reactions) and the **full mechanism** (53 species, 325 reactions). The mixture
is 2 % H2 / 1 % O2 / 97 % N2 by mole, at T = 1500 K, constant volume, with an
**additive** H seed of 10⁻¹¹ mol cm⁻³.

The primary comparison point is a **pre-front checkpoint at t = 10⁻⁴ s**, which
sits at 79 % of the ignition delay (126 µs) and therefore inside the exponential
growth phase but before the front. Isothermal integration is used for parity;
adiabatic constant-volume integration is used for the ignition-delay anchors of
§3.4.

For the full mechanism the step is dt = 2 × 10⁻⁹ s; dt = 10⁻⁸ s lies outside
the RK4 stability limit there because of `NNH ⇌ N2 + H`.

### 2.2 The five implementations

| implementation | integrator | role |
|---|---|---|
| Sounio native module (`stdlib/chemistry/gri30_h2.sio`, `gri30_full.sio`) | fixed-step RK4 | object under test; carries coherent uncertainty propagation |
| Python replica (`gri30_h2_python_replica.py`, `gri30_full_python_replica.py`) | fixed-step RK4 | diagnostic |
| C++23 cross-check (`cpp/gri30_h2_band_crosscheck.cpp`) | fixed-step RK4 | independent third implementation, written from the published protocol |
| Cantera 3.2.0 | CVODE, `rtol = 10⁻¹²`, `atol = 10⁻²²` | oracle for the trajectory |
| Gragg–Bulirsch–Stoer in Sounio (`examples/chemistry/gbs_oracle.sio`) | modified midpoint + Richardson extrapolation in h² | second integrator, added 2026-09-02 (§4.6) |

"Independent" for the C++23 cross-check means independent **code**, written
from the protocol description rather than copied — it shares the same RK4
formula, step, mechanism JSON and (deliberately, in §5) the same
per-step-independent quadrature construction as the Python replica. Its role
is a bug check on the replica's own arithmetic (§5.1's "reproduces the
Python replica's deterministic checkpoint to all 17 printed digits" licenses
using it that way), not an epistemically independent method for the band
claims of §5 — GBS is the only integrator here built by a different method.

The C++23 cross-check reproduces the Python replica's deterministic checkpoint
**to all 17 printed digits on all 8 species** (H2: `1.45202682104479838e-07`
against `1.4520268210447984e-07`), which is what licenses its use as an arbiter
in §5. It requires `-std=c++23`: `std::expected` and the multidimensional
`operator[](std::size_t, std::size_t)` are both C++23-only, and `-std=c++20`
fails with three errors. That correction is itself recorded, because the
environment line of the measurement record originally said C++20.

### 2.3 Independence of the integrators, established mechanically

Two independent integrators agreeing to 2 × 10⁻¹¹ requires an explanation
rather than an assumption; the natural failure mode is that the "replica" is in
fact calling the oracle. Checked by parsing rather than by reading, on
`gri30_h2_cantera_parity.py`:

```
params            ['gas', 'T', 't_end', 'dt']
params used       ['gas', 'T', 't_end']        <- dt is NEVER referenced
method calls      ['IdealGasReactor', 'ReactorNet', 'advance']
loops in body     0                            <- there is no stepping at all
```

and `grep -c 'cantera\|ct\.'` over the whole replica returns **0**. The Cantera
side makes one `net.advance(t_end)` call and does its own adaptive stepping; the
replica's `rk4_step` calls only its own `dc_dt`. There is no path by which
Cantera could act as a rate evaluator inside the RK4 loop.

### 2.4 Initialisation: TDY against TPX

The as-shipped parity script initialised Cantera with `gas.TPX = T, P0, X`,
which renormalises the mole fractions and pins P = 101325 Pa exactly. The
protocol documented for the comparison initialises through `TDY` so that the
seeded concentrations are *not* renormalised, which makes the realised initial
pressure 101325.576758 Pa. The two differ in the initial state:

| initialisation | worst deviation from intended initial concentrations | realised pressure (Pa) |
|---|---|---|
| `TPX` (as shipped) | **5.692129 × 10⁻⁶**, uniform over species | 101325.000000 |
| `TDY` (as documented) | **0.000000** | **101325.576758** |
| `TDY`, aligned constants | **1.629030 × 10⁻¹⁶** (one ULP) | 101325.124717 |

The TDY path reproduces the documented pressure to all six decimals, which
independently confirms it is the intended protocol. A uniform −5.69 × 10⁻⁶
initial-density error is amplified by chain branching to 3.9 × 10⁻⁵ in the
radicals by the checkpoint — an amplification of about 6.9× — and that single
line accounted for the whole ~15× gap between the as-shipped script's answer and
the documented one. Ignition delays are unaffected (169.66 / 126.34 / 98.29 /
79.00 / 65.08 µs at 1400–1800 K).

This three-row table is used again in §6 as a **provenance signature**: it
identifies, from the result alone, which merge produced the tree that produced
the result.

### 2.5 The two constant regimes

All results are reported in one of two regimes, and no number is quoted without
its regime:

| | published regime | aligned regime |
|---|---|---|
| activation-energy gas constant `R_cal` | 1.9872041 cal mol⁻¹ K⁻¹ | 8.31446261815324/4.184 = **1.9872042586408316** |
| molar volume | `1/(82.057·T)` | `P0/(R_SI·T)·10⁻⁶` with `R_SI` = 8.31446261815324 |

The two changes were measured **factorially**, not jointly. Under TDY
initialisation the molar-volume constant contributes **exactly zero** to the
*parity gap* — the trajectory-endpoint comparison of §4: `R_cal` is the entire
effect there. The molar-volume change is therefore justified on its own
terms — it removes a truncated constant — and not by a parity improvement it
does not produce. This is a distinct claim from the realised initial
pressure (§2.4, Table on p. TDY/TPX), which **does** change with the
molar-volume alignment (101325.576758 → 101325.124717 Pa) and remains a
valid identifying value precisely because it is a different quantity than
parity — §6 instance (7)'s signature is built on the initial-*concentration*
deviation, not on parity, so its usefulness as a provenance marker is
unaffected by parity's insensitivity to this constant.

One caveat is carried explicitly and is **not** closed by this work: whether
GRI-Mech 3.0's published rate parameters were themselves regressed under this or
another rounding of R cannot be established from the regression documentation,
which is not available here. The alignment was chosen with that uncertainty
explicit.

### 2.6 Reproduction contract

Each section of the measurement record names a committed producer. Two probes
**fail closed** on provenance: `rep_traj_bug.py` and `rep_tolerance.py` draw
their initial state from the oracle's `initial_concentrations()` and *raise*
rather than report when that helper is absent, because its absence is the
signature of the TPX variant of §2.4. They refuse to compare two protocols that
were never the same.

### 2.7 Environment

Linux x86-64; Python 3.11; `cantera 3.2.0`; `numpy 2.4.6`; `g++ (Ubuntu 13.3.0)
-std=c++23 -O2`; Lean `leanprover/lean4:v4.33.0`.

Sounio compiler: **`bin/souc-lean-single-x86_64`, md5
`458d82bc22e44caaca1161231f56d82d` at tree `98aa8e4d`**. Every Sounio command in
§8.2 exports `SOUNIO_SOUC_ENGINE=lean_single`, and `bin/souc` routes that
setting to the legacy single-file ELF at that path; the modular Madaros ELF
`bin/madaros-linux-x86_64` (md5 `ff69dae4c9d733ef21c0f04c9678f34b`) is the
default engine of the same wrapper but **was not executed for any measurement
reported here**.

> **Corrected 2026-09-22, and it is an instance of §6.** The environment line of
> the measurement record named the Madaros ELF and its md5 while also naming the
> `lean_single` override that routes past it, so the artefact identified was not
> the artefact run. Nothing numerical changes — the runs were always made by the
> `lean_single` engine, which is what §6 instance (8) would call a documentation
> number with no resolution label, here applied to a binary identity. The md5
> above is `git cat-file -p 98aa8e4d:bin/souc-lean-single-x86_64 | md5sum`,
> computed rather than recalled. Note that this ELF also lags its source: per the
> repository's own guidance, a committed binary is not evidence about the
> compiler source at the same commit.

---

## 3. The reverse-rate defect and the provenance of "0.2–2 %"

### 3.1 The defect, at the exponent

A historical revision of the adiabatic replica computed reverse-rate exponents
as `reac − nu`. Three forms exist, not two:

**Table 1 — the three exponent forms.**

| form | expression | equals |
|---|---|---|
| shipped | `p = prod[r][s]` | `prod` |
| proposed "fix" | `p = reac + nu` | **≡ `prod` — an identity** |
| reported bug | `p = reac − nu` | `2·reac − prod` |

`reac − nu ≡ 2·reac − prod`, and the guard is `if p > 0`, so a negative exponent
is *skipped*. In **all 29 of 29** reactions the products (correct exponent ≥ 1)
go negative and drop out, while the reactants (correct exponent 0) enter at +2
or +4. The reverse term stops depending on product concentrations altogether.

For `H + HO2 ⇌ O2 + H2` the exponents become H2 −1 (skipped), O2 −1 (skipped),
H +2, HO2 +2. Its products are the majors, so its reverse direction is
H2 + O2 → H + HO2 — chain initiation. In a majors-only state the correct reverse
rate is large and the buggy one is ∝ c[H]²·c[HO2]² ≈ 0, which is precisely the
documented symptom ("zeroed the reverse channels of H2 + O2 → H + HO2"),
re-derived here independently.

**The proposed "fix" is a no-op**, confirmed by measurement and not only by
algebra: the delta is **exactly 0, bit-for-bit**, on all eight species, at the
checkpoint and in the 1-σ band.

### 3.2 The per-species profile, and what claim (A) actually is

Producer: `benchmarks/chemistry/rep_traj_bug.py`, which runs all three exponent
forms.

**Table 2 — per-species deviation under `reac − nu`, isothermal checkpoint
(T = 1500 K, t = 10⁻⁴ s, dt = 10⁻⁸ s), against claim (A).**

| species | shipped (mol cm⁻³) | under `reac − nu` (mol cm⁻³) | delta | claim (A) says |
|---|---|---|---|---|
| H2 | 1.45202409180648259e-07 | 1.44896760284528512e-07 | **2.105e-03** (0.21 %) | "majors 0.2–2 %" |
| O2 | 7.39988119297947390e-08 | 7.38675293103348018e-08 | **1.774e-03** (0.18 %) | "majors 0.2–2 %" |
| HO2 | 1.70510868870023519e-11 | 1.70748634715518421e-11 | 1.394e-03 | — |
| O | 1.45235755728164051e-09 | 1.47399979085128111e-09 | 1.490e-02 | — |
| H2O | 1.17789779335583026e-08 | 1.19776593065772939e-08 | 1.687e-02 | — |
| H | 9.78118787162766235e-09 | 9.95290506208524791e-09 | 1.756e-02 | — |
| OH | 1.22456521149193150e-09 | 1.26670674554832556e-09 | **3.441e-02** (3.44 %) | "radicals ~3 %" |
| H2O2 | 1.62463513786769239e-13 | 1.88736066163527837e-13 | **1.617e-01** (16.17 %) | "H2O2 ~16 %" |

Three figures, three matches, the last to three significant figures.

> **[W] Retraction that is part of the result.** An earlier revision of the
> measurement record stated that claim (A) "has no provenance and should not be
> cited". That was wrong, and the correction matters more than the original
> statement. **Claim (A) is not a parity table. It is the per-species error
> profile of the `reac − nu` reverse-rate defect, presented as though it were a
> Sounio-versus-Cantera comparison.** The number was right; the label was not.
> No pairing of Sounio, replica and Cantera produces percent-level deviations —
> that part of the earlier finding stands, and §4 measures it.

A second method error is recorded with the same weight: the defect was first
reported *absent* because the file named in the brief did not contain
`reac − nu`, generalising from "absent here" to "never existed". The buggy
revision is not recoverable — history for these files in this clone begins at
`d25b43a4`, and no version under any ref contains the expression — so the
documentation is testimony, not artefact.

### 3.3 Why the shipped and fixed forms are identical yet the bug is not

At the checkpoint, R16's reverse channel is 0.16 % of its forward channel
(forward 5.22176484288829602e-06, reverse 8.54651292554704534e-09, ratio
1.637 × 10⁻³). The *fix* is the same expression, so its delta is exactly zero.
The *bug* does not merely delete that 0.16 %: it substitutes a term keyed to
reactant rather than product concentrations, and ten thousand steps of chain
branching turn that substitution into the percent-level profile of Table 2.

### 3.4 Adiabatic anchors reproduce; one protocol variable was undeclared

Producer: `benchmarks/chemistry/rep_adiabatic_bug.py`, which monkey-patches the
versioned module's `uv_rhs` rather than carrying its own chemistry, so it cannot
drift from the replica it characterises.

**Table 3 — ignition delay at the time of maximum d[H2O]/dt, dt = 5 × 10⁻⁹ s.**

| T₀ (K) | correct (µs) | under the bug (µs) | error | documented anchor |
|---|---|---|---|---|
| 1100 | 674.9575 | 675.6025 | **+0.096 %** | **0.1 %** |
| 1400 | 169.6625 | 173.7175 | +2.390 % | — |
| 1700 | 78.9775 | 81.9575 | +3.773 % | — |
| 2000 | 46.3075 | 50.2675 | **+8.552 %** | **8.6 %** |

Both documented anchors reproduce. On this criterion every sign is positive: the
defect always delays ignition, consistent with removing a radical source.

The replica exposes **two** delay definitions — time of maximum d[H2O]/dt and
time of maximum dT/dt — and the source documentation does not say which it used.
They disagree, and at 1100 K the sign inverts. The comparison is given in
**Appendix G**; the short statement is that a defect shifting the H2O-rate delay
by +0.096 % shifts the temperature-rise delay by −2.374 %, which says the dT/dt
peak is broad and ill-conditioned there at this step, not that the defect
accelerates ignition.

### 3.5 Two claims that do not reproduce

- "d[HO2]/dt changes by −34 %" is **not reproduced**: measured −9.86 %
  (shipped −8.13560209958764591e-08, under the bug −8.93741991782262909e-08).
  It is also distinct from the −50.5 % figure, which is R16's *net rate of
  progress* at a radical-loaded probe state — a different quantity at a
  different state. Three numbers, three quantities; −34 % matches none of them
  and is **[U] left unexplained rather than forced into agreement**.
- The shipped isothermal reverse path is correct, verified against the oracle
  rather than by inspection: all 29 net rates of progress against
  `Cantera.net_rates_of_progress` at a radical-loaded state agree to
  **7.877 × 10⁻⁷** in the published regime and **8.442 × 10⁻¹⁵** after
  alignment, while the buggy form sits at **1.838** (184 %) in *both* regimes,
  to four figures. **The defect is a property of the stoichiometry; the floor
  beneath it was a property of the constant.**

### 3.6 Standard-state reference pressure: no defect present

A suspected 1 bar / 1 atm confusion was measured rather than assumed. Cantera
reports `reference_pressure = 101325.0 Pa` for all 53 species of `gri30.yaml`,
and both replicas already set `P0 = 101325.0`. Kc agrees with Cantera on all 29
H/O reactions: worst 1.418 × 10⁻¹⁴ on the 17 reactions with Δn = 0, and
**1.843 × 10⁻¹¹ on exactly the 12 with Δn ≠ 0** — the only ones carrying the
factor `c0 = P0/(R·T)` — which is fully explained by the replica's truncated
`R_SI = 8.314462618` against Cantera's `8.31446261815324`, a relative difference
of 1.84 × 10⁻¹¹.

The counterfactual is worth stating because the reasoning around it was subtly
wrong. Had P0 been 10⁵ Pa, Kc would scale by (10⁵/101325)^Δn: +1.3250 % at
Δn = −1, −1.3077 % at Δn = +1, exactly zero at Δn = 0. It is **not** true that
Δn ≠ 0 is rare in the H/O submechanism — it is 12 of 29, i.e. 41 %. What makes
such a defect unobservable is not scarcity but **direction of flux**: at 1500 K
in the induction period these recombinations run overwhelmingly forward, and Kc
enters only the reverse term, which is 6 to 15 orders below the forward one.
The defect would become observable only where an equilibrium pins a population —
NNH (Δn = +1) in the full mechanism, whose population is set by a fast
quasi-equilibrium rather than by accumulated flux. No such defect exists in this
code, so no magnitude is claimed for the shipped artefacts.

---

## 4. Parity: the published gap is one rounded constant

### 4.1 The three-way measurement

**Table 4 — relative deviations at the pre-front checkpoint, published regime.**
(Absolute concentrations for all three implementations are in the measurement
record, §1.2.)

| species | Sounio vs replica (display precision — see below) | Sounio vs Cantera (as-shipped, TPX) | Sounio vs Cantera (documented, TDY) |
|---|---|---|---|
| H2 | ~~3.305e-12~~ → 1.823e-16 | 1.739e-06 | 2.854e-07 |
| H | ~~4.573e-11~~ → 2.368e-15 | 3.851e-05 | 2.365e-06 |
| O | ~~3.467e-11~~ → 4.271e-15 | 3.922e-05 | **2.660e-06** |
| O2 | ~~1.091e-11~~ → 1.789e-16 | 2.435e-06 | 2.387e-07 |
| OH | ~~2.639e-10~~ → 4.559e-15 | 3.808e-05 | 2.624e-06 |
| H2O | ~~3.427e-11~~ → 2.949e-15 | 3.911e-05 | 2.400e-06 |
| HO2 | ~~1.004e-11~~ → 1.895e-16 | 8.918e-06 | 2.247e-07 |
| H2O2 | ~~4.890e-12~~ → 3.884e-15 | 3.705e-05 | 1.891e-06 |
| **range** | ~~3.3e-12 … 2.6e-10~~ → **1.789e-16 … 4.559e-15** | 1.7e-06 … 3.9e-05 | **2.2e-07 … 2.7e-06** |

> **[W] Corrected 2026-09-23, a reviewer finding.** The struck-through values
> are the raw print-resolution figures the original demo emitted; a reader
> quoting them alone, without the paragraph that follows this table, would
> cite a retracted number. The arrow gives each species' actual value at full
> precision, transcribed from the 16-digit reprint in the measurement record
> (`RESULTS.md` §1.3), not re-derived here. This column is otherwise the same
> measurement as bullet 1 below, tabulated per species rather than as a
> range.

Replica-vs-Cantera is identical to Sounio-vs-Cantera to three significant
figures in every cell, because Sounio and the replica agree about 10⁴× more
tightly than either agrees with Cantera.

Two conclusions follow, and both correct the claims of §1.2:

1. **Sounio vs the replica is 1.8 × 10⁻¹⁶ to 4.6 × 10⁻¹⁵ — 1 to 30 ULP, i.e.
   15 significant figures, not 5–6.** The apparent 3.3 × 10⁻¹² … 2.6 × 10⁻¹⁰ of
   the demo was **entirely a print-resolution artefact**; re-printing the same
   checkpoint at 16 digits gives 1 ULP on H2, O2 and HO2 and at most 30 ULP on
   O. The two implementations share the integrator, the step, the mechanism JSON
   and the summation order, so the only residual is the transcendental-function
   implementations (`exp`, `log`, `pow`) — and at 1–30 ULP against a
   double-precision ε of 2.220 × 10⁻¹⁶, that is exactly what is measured. There
   is no cross-language discrepancy to explain.
2. **Claim (C) is contradicted.** H2O2 agrees to **1.891 × 10⁻⁶**, three orders
   of magnitude better than the published 5.9 × 10⁻³.

Claim (B) — "5–6 significant figures, 2 × 10⁻⁷ to 6 × 10⁻⁶" — is **confirmed**
for the Sounio-vs-Cantera pairing under the documented protocol. Its stated
*cause* is not.

### 4.2 The attribution, closed analytically and by substitution

The replica and the Sounio module use the CHEMKIN-conventional rounded
`R = 1.9872041` cal mol⁻¹ K⁻¹; Cantera converts cal → J at exactly 4.184 and
divides its own gas constant, giving 8.31446261815324/4.184 =
**1.9872042586408316**, a relative difference of **7.983 × 10⁻⁸** sitting inside
`exp(−Ea/(R·T))`. With max Ea/(RT) ≈ 9.87 at this state, the first-order
estimate is 7.877 × 10⁻⁷ per rate, amplified ≈ 3.4× by chain branching over the
trajectory to ≈ 2.7 × 10⁻⁶ — the observed gap.

Substituting Cantera's value and re-running the identical trajectory:

**Table 5 — deviation from Cantera before and after aligning `R_cal`.**

| species | R = 1.9872041 | R = 1.9872042586408316 | improvement |
|---|---|---|---|
| H2 | 2.854e-07 | **1.082e-12** | 263,862× |
| H | 2.365e-06 | **9.001e-12** | 262,784× |
| O | 2.660e-06 | **9.197e-12** | 289,178× |
| O2 | 2.388e-07 | **8.952e-13** | 266,715× |
| OH | 2.624e-06 | **8.867e-12** | 295,899× |
| H2O | 2.400e-06 | **9.280e-12** | 258,573× |
| HO2 | 2.247e-07 | **1.246e-12** | 180,396× |
| H2O2 | 1.891e-06 | **1.220e-11** | 155,031× |

**The entire published parity gap is one rounded constant.** It is the same root
cause as the 1.843 × 10⁻¹¹ Kc floor of §3.6, where the truncated constant was
`R_SI` instead: two independent truncated gas constants, two residual floors,
both fully accounted for.

> **[B] The right-hand column must not be read as resolution.** Recomputed
> 2026-09-23 rather than trusted: **all eight** of these figures sit below
> the oracle's *single-state* floor of 1.473 × 10⁻¹¹ (§4.5) — the largest,
> H2O2 at 1.220 × 10⁻¹¹, still under it. Three of the eight — H2, O2, HO2 at
> 1.082 × 10⁻¹², 8.952 × 10⁻¹³, 1.246 × 10⁻¹² — sit below even the *lowest*
> value the ensemble sweep of §4.5 ever measured for that floor
> (3.730 × 10⁻¹²), meaning those three are unresolvable against any floor
> value this document reports, not only the typical one. None of these eight
> numbers should be read as resolution.

> **[W] Corrected 2026-09-23, a reviewer finding.** Table 5 is an isolated
> substitution — `R_cal` alone, changed inside a reconstructed probe and
> re-run, historically, before the molar-volume alignment landed in the
> committed tree. It is **not** the same measurement as §4.5/Table 8's "total
> residual" (2.074 × 10⁻¹¹, H2O), which comes from a later run against the
> fully-committed, both-constants-aligned code — a different producer, a
> different commit state. Both numbers are real; they are not the same
> experiment, and their difference (9.28 × 10⁻¹² here vs 2.074 × 10⁻¹¹ there,
> both nominally "H2O, aligned") is not a discrepancy to resolve — it is
> consistent with, and additional evidence for, §4.5's own finding that the
> oracle's floor at this scale varies by up to 11× under perturbations of one
> part per million. Citing either number alone as *the* aligned residual,
> without naming its producer, is the defect §6 catalogues.

### 4.3 It is not the integrator — falsified at the operating step

The natural alternative explanation is fixed-step RK4 versus CVODE. It is
excluded three ways, in increasing strength.

*Weakest — the order test.* Against CVODE at `rtol = 10⁻¹³`, the replica's
worst relative error falls 1.384e-06 → 7.738e-08 → 4.554e-09 → 2.739e-10 →
**6.580e-12** as dt goes 8e-7 → 4e-7 → 2e-7 → 1e-7 → 1e-8, with successive-error
ratios of 15.15–20.36 against the 2⁴ = 16 expected of a fourth-order method.
Fourth order is confirmed; but this argues from the coarse-dt regime and then
extrapolates.

*Decisive — bisection at the operating step.* Producer:
`benchmarks/chemistry/rep_tolerance.py`.

**Table 6 — deviation from Cantera under halving of the step.**

| species | \|dev\| dt = 1e-8 | \|dev\| dt = 5e-9 | ratio |
|---|---|---|---|
| H2 | 2.854e-07 | 2.854e-07 | **1.000** |
| H | 2.365e-06 | 2.365e-06 | **1.000** |
| O | 2.660e-06 | 2.660e-06 | **1.000** |
| O2 | 2.388e-07 | 2.388e-07 | **1.000** |
| OH | 2.624e-06 | 2.624e-06 | **1.000** |
| H2O | 2.400e-06 | 2.400e-06 | **1.000** |
| HO2 | 2.247e-07 | 2.247e-07 | **1.000** |
| H2O2 | 1.891e-06 | 1.891e-06 | **1.000** |

A fourth-order truncation error falls by 16 under this halving. It does not move
— ratio 1.000 on all eight species, to four significant figures. The residual is
a **fixed offset, invariant under step size**. The same run gives the replica's
self-convergence as **2.7 × 10⁻¹⁵ … 2.2 × 10⁻¹⁴**, eight orders below the
2.66 × 10⁻⁶ gap: truncation cannot account for it even in principle.

*And the bisection repeats in the aligned regime*, where the 2.66 × 10⁻⁶ offset
is gone and could no longer mask a smaller step-dependence: across a factor of
**four** in step (1e-8 / 5e-9 / 2.5e-9) the ratios are 0.996–1.003, where a
fourth-order error would have fallen by 256. The remaining 2 × 10⁻¹¹ is a fixed
offset at both scales.

Finally, claim (A) is not rescued by this route either: extrapolating the
established dt⁴ law, reaching 0.2 % would require dt ≈ 4.9 × 10⁻⁶ s — 493× the
step used, far outside RK4 stability for this mechanism.

### 4.4 The full mechanism reproduces the H/O result

A Cantera parity script for the full 53-species / 325-reaction mechanism did not
previously exist; one was written, and its initial state was later found to be
unaligned with the module it is the oracle for (§6, instance 10). The aligned,
committed version gives:

**Table 7 — full mechanism, worst relative deviation Sounio ↔ Cantera.**

| checkpoint | published regime | aligned regime |
|---|---|---|
| t = 4 × 10⁻⁶ s | 5.268e-07 | **3.175e-11** |
| t = 2 × 10⁻⁵ s | 8.242e-07 | **1.024e-11** |

Sounio agrees with the Python replica at **0–16 ULP** across 10 reported species
(four are bit-identical at one or both checkpoints), so the 10⁻⁷-scale figure is
the RK4-versus-CVODE integrator difference, not a cross-language one. **NNH** —
the Δn = +1 quasi-equilibrium species of §3.6 — tracks to 5.129 × 10⁻¹² in the
aligned regime, confirming that its equilibrium-pinned population is reproduced.

The full-mechanism uncertainty section of the source documentation, by contrast
with the H/O prose, is **not** stale: the coherent Sounio band against an
independent Cantera central-difference referee reproduces exactly, every figure
— largest σ deviation **8.724 × 10⁻⁷** (H), the other seven spanning
1.559 × 10⁻⁷ … 7.171 × 10⁻⁷, and the referee's own H2 value reproducing the
published figure to all 16 digits.

### 4.5 The oracle's resolution, and what it costs this work's headline

§4.2's aligned figures were originally described as agreement "at the floor of
CVODE's own `rtol = 10⁻¹²`". That framing is retired here. The floor was never
measured; it is not a property of the tolerance alone.

Measured, ten species, fresh `gas` object per run
(`rep_resolution.py`): Cantera at `rtol = 10⁻¹²` against `10⁻¹³` differs by
**1.473 × 10⁻¹¹** (H2O); `10⁻¹³` against `10⁻¹⁴` by 5.839 × 10⁻¹²;
`10⁻¹²` against `10⁻¹⁴` by 2.057 × 10⁻¹¹.

> **[W] Corrected 2026-09-23, twice.** This paragraph originally said "ten
> species," which a first pass changed to "eight species (the H/O checkpoint
> set)" on the mistaken belief that `rep_resolution.py`'s eight-member
> `REPORT` list was what fed these three comparisons. Re-checked against the
> producer's actual call: `worst(runs[a], runs[b], ALL10)`
> (`rep_resolution.py:120,124`, `ALL10 = REPORT + ["N2", "AR"]`) — the
> oracle-resolution figures in this paragraph are measured over **all ten**
> species, and "ten species" was correct the first time. `REPORT` (eight
> species) is what the *separate* step-bisection further down in the same
> script iterates over; conflating the two producer-internal species lists is
> what caused this. Separately, the underlying point about this 1.473 × 10⁻¹¹
> figure and the 1.416 × 10⁻¹¹ floor in §4.5's own decomposition table below
> still holds: they are **two independent invocations** of the same script —
> this one a bare call at the working state, the other `--dir <aligned tree>`
> inside the dt-bisection — not the same measurement re-quoted. They differ by
> 4%, which is not a discrepancy: it is the next paragraph's finding, restated
> a section early.

And the floor is not even a number. Perturbing the initial density by
δ ∈ [−10⁻⁶, +10⁻⁶] over ten states, fresh `gas` each (`rep_floor_spread.py`),
the floor ranges over **[3.730 × 10⁻¹², 4.142 × 10⁻¹¹] — a factor of 11.1 from
perturbations of one part per million**, with no monotone trend in δ.

**Table 8 — decomposition of the aligned residual. [B]**
Every entry is a measurement; nothing is inferred by subtraction, because the
parts are not orthogonal and a subtraction would manufacture a number.

| part | how measured | value | share of residual |
|---|---|---|---|
| total residual, replica vs Cantera `rtol = 10⁻¹²` | worst species (H2O) | **2.074e-11** | 100 % |
| oracle's own resolution at one state | `10⁻¹²` vs `10⁻¹³`, fresh `gas` | 1.416e-11 | 68 % *at that one state* |
| oracle's own resolution, ten states at ±10⁻⁶ density | `rep_floor_spread.py` | **3.730e-12 … 4.142e-11** | **18 % … >100 %** |
| replica truncation + roundoff, upper bound | \|c(1e-8) − c(5e-9)\|/\|c\|, worst (H2O2) | **≤ 3.465e-14** | 0.17 % |
| unaccounted | — | **undefined**: the oracle's band spans the residual | — |

> **[W] A 68 %/32 % partition was published and is withdrawn.** It was a single
> sample from a distribution that swallows it. The correct statement replaces a
> number with a range: **the oracle explains between 18 % and more than 100 % of
> the aligned residual.**

The generalisable observation: **the resolution of an adaptive integrator is not
a smooth function of the initial state.** CVODE re-selects its step sequence
under a perturbation of 10⁻⁶, and the re-selection moves the tolerance-induced
error by an order of magnitude. An adaptive oracle's resolution therefore has no
value independent of the state it is run from, and must be measured *in regime,
every time*, over an ensemble rather than at a point.

### 4.6 A second integrator, and the step-refinement anomaly

The replica's self-difference *grows* under refinement for five of eight species
— by 12× for H, O and H2, by 140× for H2O (Appendix F). That was first
attributed to roundoff, which cannot be right: a random-walk roundoff gives
√2 ≈ 1.41× per halving, a systematic one 2×, fourth-order truncation 1/16;
nothing known gives 12× to 140×. The one hypothesis with a mechanism — **step
stagnation**, where dt·|dc/dt| falls below half an ULP of c and the update rounds
back — was tested and **not confirmed**: the per-step increment is above 10¹¹
half-ULPs for every species at every step tried, where stagnation needs it below 1.

A self-difference cannot say which of two runs is wrong, so the question was
closed on 2026-09-02 with a genuinely independent method: **Gragg–Bulirsch–Stoer**
(modified midpoint with Richardson extrapolation in h²), implemented in Sounio,
sharing the replica's right-hand side verbatim so that only the time stepping
differs.

The oracle is characterised before it is used, because Richardson extrapolation
divides by ((n_k/n_j)² − 1) and so amplifies roundoff as depth grows. Sweeping
depth against subdivision sequence, both sequences bottom out at **depth 6** and
rise afterwards — that minimum *is* the truncation-to-roundoff crossover,
measured in place. The wider sequence (2,4,6,8,12,16,24,32) bottoms out four
times lower than the harmonic one, at **1.421 × 10⁻¹⁴** worst over species, and
is the instrument used.

Against that independent method, at dt = 5 × 10⁻⁹ **seven of eight species sit
below the oracle's own per-species resolution**; four halvings later, at
dt = 1.25 × 10⁻⁹, **seven of eight are above it, by 2.6× to 24×**. The replica's
distance to an external reference therefore has a minimum and then grows as the
step shrinks. Growth measured against a *different* method cannot be truncation
being resolved, and cannot be an artefact of comparing a run with itself.

So the 12–140× is the **right-hand branch of the total-error curve of a
fixed-step method**, where accumulation over 10⁴ to 8 × 10⁴ steps overtakes a
truncation term already spent. **[U] What is still owed is the exponent, not the
mechanism**: the observed per-halving factors are 2.1× to 6.6× where systematic
accumulation predicts 2×, and no model here derives 6.6×. The location of the
minimum is bracketed between 10⁻⁸ and 2.5 × 10⁻⁹ and is not pinned, because the
5 × 10⁻⁹ row lies at the instrument's floor.

The bound that Table 8 depends on is unchanged and now has independent support:
at dt = 10⁻⁸ the replica is 3.263 × 10⁻¹⁴ from the second method, three orders
below the residual, so **the replica contributes nothing measurable to it**. It
also follows that dt = 10⁻⁸ was a fortunate choice — refining it does not
improve this replica, it degrades it.

### 4.7 The checkpoint sits at the minimum of the truncation curve

**Figure 1 — truncation against time.** `|c(dt = 1e-8) − c(dt = 5e-9)|/|c|`,
worst over the eight reported species; to be plotted log–log from these five
measured points. No plot file forms part of the record.

> **[W] Corrected 2026-09-23, a reviewer finding.** The table below is measured
> entirely in the **published** regime, matching the row it is drawn from in
> the two-column published/aligned table of §4.3. Its own "regime" column
> names the *trajectory phase* (induction, pre-front, front), not the
> constant-alignment regime — the same word doing two jobs. That collision is
> why this table's checkpoint figure (2.222 × 10⁻¹⁴, H2O2, published) and
> Table 8's truncation-bound figure (3.465 × 10⁻¹⁴, H2O2, **aligned** — its
> own title says so) look like the same quantity measured twice
> inconsistently. They are the same functional at the same checkpoint in two
> different constant regimes, both correct, neither a re-quote of the other.

| t (s) | worst | on | trajectory phase |
|---|---|---|---|
| 1.00e-06 | **1.608e-11** | H2O | early induction |
| 1.00e-05 | 7.330e-14 | O | induction |
| **1.00e-04** | **2.222e-14** | H2O2 | **the pre-front checkpoint (published regime)** |
| 1.20e-04 | 8.508e-14 | HO2 | approaching the front |
| 1.30e-04 | 2.088e-13 | HO2 | into the front |

Truncation at the checkpoint is **724× lower than at t = 10⁻⁶ s and 9.4× lower
than just past the front**. Both readings of that fact are stated because both
are true. For the question this work asks — constant or integrator? — the
checkpoint is well chosen, since truncation there is five orders below the
published-regime gap. For a claim that the two integrators *agree*, it is the
most favourable point available and must not be presented as representative: a
parity table at t = 1.3 × 10⁻⁴ s would sit an order worse, and no such table is
reported here. The rise at t = 10⁻⁶ s is the sharper caution — anyone re-using
this protocol at a shorter horizon inherits a much weaker bound.

---

## 5. Uncertainty-band scaling: the oracle's accumulation rule is wrong

### 5.1 Scaling in dt

The replicas' predicted law is band ∝ dt·√(T/dt) = √(T·dt), so a factor **f** in
dt gives **√f**.

**Table 9 — measured band ratios, Python replica (C++23 cross-check reproduces
every figure to all six printed digits), T = 10⁻⁶ s fixed.**

| species | 4e-9 → 2e-9 (f = 2, √f = 1.414214) | 2e-9 → 1e-9 (f = 2) | 4e-9 → 1e-9 (**f = 4**, √f = **2.000000**) |
|---|---|---|---|
| H | 1.414792 | 1.414503 | **2.001227** |
| O | 1.414585 | 1.414399 | **2.000787** |
| OH | 1.415047 | 1.414630 | **2.001769** |
| H2O | 1.418167 | 1.416197 | **2.008405** |
| HO2 | 1.414183 | 1.414198 | **1.999935** |
| H2O2 | 1.417520 | 1.415872 | **2.007027** |
| H2 | 1.000000 | 1.000000 | 1.000000 |
| O2 | 1.000000 | 1.000000 | 1.000000 |

Claim (D) is therefore **contradicted in its labelling**: a factor 4 gives
1.9999–2.0084, i.e. 2.0 = √4; the ratio √2 is what a factor **2** gives. The
logged entry "factor 4 in dt, ratio √2" conflates two sweeps — √2 is a correct
measurement wearing the wrong factor.

A refinement the √dt statement needs: the law holds **only for species whose
band is generated by accumulation**. H2 and O2 give exactly 1.000000 under every
dt change, because their variance is dominated by the 1 % initial-condition
uncertainty seeded at t = 0, which neither accumulates nor scales with dt.
Stating "the band scales as √dt" without that qualifier is false for 2 of the 8
species reported.

### 5.2 Scaling in T: the √(T/dt) law is not exact

**Figure 2 — measured band ratio per decade against the predicted
√10 = 3.162278**, to be plotted from these measured values.

| species | T: 1e-6 → 1e-5 | T: 1e-5 → 1e-4 |
|---|---|---|
| O | 3.227593 | 577.01 |
| OH | 3.532908 | 573.25 |
| HO2 | 3.552849 | 186.64 |
| H | 6.150410 | 639.96 |
| H2O | 15.342525 | 766.11 |
| H2O2 | 27.409671 | (overflowed the reported set) |
| H2 | 0.999927 | 0.891875 |
| O2 | 0.999927 | 0.908202 |

Over the first decade only O, OH and HO2 land near 3.16 (2–12 % high); H is 1.9×
high, H2O 4.9×, H2O2 8.7×. Over the second decade the measured ratios exceed the
prediction by **59× to 242×**.

The mechanism is not mysterious. √(T·dt) describes the pure quadrature
accumulation of the per-step parameter term with a frozen Jacobian, valid only
in the induction period. Between t = 10⁻⁵ and 10⁻⁴ s the trajectory enters
chain-branching growth, and the Jacobian terms `2·J_ii·v_i·dt` and
`Σ_k (J_ik·dt)²·v_k` dominate the parameter term entirely. **The claim spans not
two decades of validity but less than one**, and should be stated as a property
of the quiescent limit rather than as a law holding across the sweep.

### 5.3 The Sounio native band is step-invariant

**Table 10 — band under a factor 4 in dt, three implementations.**

| implementation | band ratio |
|---|---|
| Python replica (per-step independent quadrature) | **1.99994 – 2.00841** |
| C++23 cross-check (same formula, independent code) | **1.99994 – 2.00841** |
| **Sounio native (coherent sensitivity propagation)** | **0.999999 – 1.000001** |

The bands are non-trivial — H2 and O2 carry ~1 % of their value, the radicals
sit at 3 × 10⁻¹⁴ to 6 × 10⁻¹⁹ — so this is invariance, not a degenerate zero.
The Sounio module carries the property as a declared, tested invariant
(`test_g30_epistemic_step_invariance` asserts agreement to 10⁻² across a factor
2); the sweep shows it holds to **~10⁻⁶ across a factor 4**, four orders tighter
than the test asserts.

### 5.4 What the √dt behaviour measures is the defect

The √dt dependence is not a property of the chemistry. The replicas add an
**independent** quadrature source `Σ_r (ν_ir·net_r·dt·u_r)²` at **every** step,
treating successive steps as independent when they share the same rate
parameters. Persistent parameter uncertainty is not independent across steps, so
quadrature is invalid here, and the artefact is precisely that the band acquires
a spurious √dt dependence. **A band that changes when you change the step size is
reporting the integrator, not the chemistry.**

This is the one place in this work where the oracle designation inverts: **the
implementation under test's architecture is right — independent quadrature is
not applied to a persistent parameter — and its reference implementation's is
wrong.** A reviewer taking the replica as ground truth and the √dt scaling as a
physical result would have drawn the opposite conclusion.

**[W] Corrected 2026-09-23, a reviewer finding.** "The implementation under
test is right" overstated what §7.5's own Threats to Validity note qualifies:
the evidence here is ratio-only (step-invariance under a factor-4 dt change),
which cannot exclude that Sounio's module simply dropped the
persistent-parameter term rather than computing it correctly — H2 and O2 are
ratio-1 in every table *regardless* of whether persistent-parameter UQ is
computed at all, because their band is dominated by the step-invariant
initial-condition seed. Absolute band *widths* against a finite-difference or
GBS-referee magnitude are not measured here. What this section establishes is
narrower and still true: the replica's per-step-independent quadrature is
architecturally wrong for a persistent parameter, which the derivation above
proves directly rather than by comparison. Whether the Sounio band's
*magnitude* is also right is not yet shown.

### 5.5 The measured law is a theorem

The per-step parameter term carries the **same** rate parameter at every step, so
successive contributions are correlated with ρ = +1, not 0. For N contributions
of equal uncertainty u: the truth is variance (N·u)², uncertainty N·u;
quadrature gives N·u² and √N·u; the ratio is **√N**. With N = T/dt that is
**√(T/dt)** — the law measured in §5.2, and read along dt instead of T, the √dt
dependence of §5.1.

`SounioIndepComposition.lean` machine-checks this and the rest of
the contract: **15 theorems, zero `sorry`**, Mathlib-free core Lean 4, built
under `leanprover/lean4:v4.33.0`. Every claim is stated on *variances* rather
than standard uncertainties — since √ is monotone on the non-negatives,
comparing variances is comparing uncertainties, and the square root never has to
be constructed; in that form every statement is polynomial and the linear ones
close under `omega`.

The axiom dependencies are **measured here rather than quoted**, by appending
`#print axioms` for all fifteen names and re-running (§8.2). Nine arithmetic
theorems, six d-separation theorems:

| axioms reported | theorems |
|---|---|
| `[propext, Quot.sound]` | `quadrature_iff_zero_covariance`, `quadrature_sound_of_independent`, `quadrature_understates_of_positive_covariance`, `quadrature_sound_iff_nonpositive_covariance`, `additive_sound`, `additive_tight_at_unit_correlation`, `quadrature_below_additive` |
| **`[propext]` alone** | `quadrature_understates_correlated_sum`, `accumulation_agrees_at_one_step` |
| **none** | all six d-separation theorems: `chain_blocked_by_conditioning`, `fork_blocked_by_conditioning`, `collider_blocked_marginally`, `collider_opened_by_conditioning`, `collider_inverts_the_others`, `conditioning_not_monotone` |

> **[W] The published statement was right in substance and wrong in both
> directions of detail, and is corrected from measurement.** It said `propext`
> *and* `Quot.sound` on the arithmetic theorems: two of them, including the √N
> law that §5.1–5.2 measure, need only `propext`. And it named three
> axiom-free d-separation theorems: all **six** are axiom-free. Nothing in
> §5.1–5.4 depends on the difference; it is corrected because it is now
> measurable and was not measured before.

Principal statements: `quadrature_iff_zero_covariance` (quadrature agrees with
JCGM eq. 13 **iff** cov = 0 — it is the independence law, not an approximation);
`quadrature_understates_of_positive_covariance` (cov > 0 ⇒ quadrature is
*strictly tighter than the truth*, i.e. unsound); `quadrature_sound_iff_
nonpositive_covariance`; `additive_sound` and `additive_tight_at_unit_
correlation` (the additive default is sound for every admissible ρ and tight at
ρ = +1, hence the least such bound); `quadrature_understates_correlated_sum`
(the √N law); `accumulation_agrees_at_one_step` (at N = 1 the two agree — why
the defect is invisible in a single composition);
`collider_opened_by_conditioning` (Berkson 1946) and `conditioning_not_monotone`
(∃ a junction where conditioning turns a blocked path active, so a reachability
check with a blocklist is unsound).

**What this does not establish.** The theorems say what follows *given* a
correlation structure; they do not certify that any particular program's declared
graph matches the world. Nor does the development verify the floating-point
numerics of the integrator: it is exact arithmetic over `Int`, and the numerical
agreement is the business of §3–§5.

**A scope this manuscript states once here rather than at every occurrence:**
the theorem's algebraic model — N contributions of *equal* uncertainty, all
correlated at ρ = +1 — is the mechanism §4.7 and §5.2 identify with the
**quiescent, pre-chain-branching regime**, and Tables 9 and 10 measure it
there: fixed T = 10⁻⁶ s, before §5.2's own finding that the Jacobian terms
`2·J_ii·v_i·dt` and `Σ_k(J_ik·dt)²·v_k` take over past ≈ 10⁻⁵ s. Neither the
theorem nor the quiescent-regime tables say anything about band behaviour at
the chain-branching checkpoint (t = 10⁻⁴ s) that §3–§4's parity claims use —
that is a different quantity, not measured here. What §5.4's "implementation
right, reference wrong" claim rests on, and what *does* generalise beyond the
quiescent window, is architectural rather than numerical: the replicas add an
independent quadrature term at *every* step regardless of what regime that
step is in, which is unsound whenever steps are correlated, not only in the
window where this document happens to measure its magnitude. The theorem
formalises the magnitude in the window it was measured; the soundness
argument is not confined to that window, and the two should not be read as
the same claim.

---

## 6. The instrument concealed the defect — ten instances

Every defect reported in this work was invisible to the instrument that was
supposed to find it, and in each case **the instrument had less resolution than
the defect it was asked to measure**. The readings then came out attributed to
the object rather than to the instrument. This is the pattern worth carrying out
of the work; the individual numbers are secondary.

> **A note on the count, kept rather than absorbed.** The section was
> commissioned with two new findings, which would have brought it to six. It
> brought it to seven — it already carried five, not four — then to eight, when
> the author's own falsified premise was recorded as an instance; to nine, when
> the archive layer failed silently under the release meant to freeze the other
> eight; and to **ten** on 2026-09-03, when three separate harnesses turned out
> to have been comparing states they did not share while printing agreement. The
> miscount is kept because the instance most easily dropped from a mental list is
> (4), where the *reference's* own error was being attributed to the method under
> test — and that is the one this record had to correct twice.

**Table 11 — the two classes.**

| | (1)–(4) | (5)–(10) |
|---|---|---|
| signature | a magnitude | none in the syntax |
| detected by | a tighter setting | a **behavioural** invariant, printed |
| the fix | calibrate | instrument deliberately, and **fail closed** |

Instances (1)–(4) are instruments set too coarse: the resolution is a number,
the defect is a number, and the first is larger. They are fixable by tightening
a tolerance or choosing a finer probe.

Instances (5)–(10) have **no syntactic signature**. They are invisible to every
tool that reads the program as text or as types — no token to grep, no dimension
to check, no diff to review, no unit test that could have been written against
the source alone, because each branch, file and literal involved is individually
well-formed and correct.

**They are not, however, undetectable.** An earlier revision of this section said
they were, and then the work built an instrument that catches one of them, which
is a contradiction and is corrected here **[W]**. The constructive form, and the
reason the section is worth more than its diagnosis: *a defect with no syntactic
signature can still be given a behavioural one, by refusing to proceed when the
provenance of the initial state cannot be established.* `rep_tolerance.py`,
`rep_traj_bug.py` and `rep_resolution.py` are the reference implementation — each
draws its initial state from the oracle's `initial_concentrations()` and **raises
rather than reports** when that helper is absent, because its absence *is* the
signature of instance (7). They do not compare two protocols that were never the
same and then print a plausible number; they refuse. That is the whole recipe,
and it costs four lines.

### Instruments set too coarse

**(1) A reduced mechanism hides a standard-state error.** 12 of 29 H/O reactions
carry Δn ≠ 0, so a 1 bar/1 atm confusion would be *present* in 41 % of the
submechanism — yet unobservable there, because Kc enters only the reverse term,
which at 1500 K in the induction period is 6–15 orders below the forward one
(§3.6). It becomes observable only where an equilibrium pins a population: NNH,
Δn = +1, in the full mechanism.

**(2) A tolerance 57× too loose hides a constant.** `test_g30_sim` asserts
1.452024295 × 10⁻⁷ at 10⁻⁴ relative tolerance. The molar-volume shorthand moved
the module by 1.740 × 10⁻⁶ — **1.7 % of the tolerance**. The gate could not have
failed on it. After alignment the same gate sits at 0.14 % of tolerance, 12.4×
tighter, and now has room to see a regression of *this size* — a shift on the
order of 10⁻⁶. It remains roughly seven orders looser than the 10⁻¹¹-scale
residual §4 treats as the object of interest; 12.4× tighter is a real
improvement in headroom for constant-sized regressions, not a claim that the
gate resolves the scientific residual this manuscript reports.

**(3) A shared initialisation hides an initial-state error.** Under TDY both
sides are built from the *same* total concentration, so an error in it cancels
and the comparison is blind to it. Aligning `R_cgs` moves the TDY column **not at
all** — 2.660 × 10⁻⁶ before and after — while under TPX, which does not share
it, the same change is worth 3.922 × 10⁻⁵ → 6.576 × 10⁻⁶. The protocol choice,
not the constant, decided what the test could see.

**(4) A loose reference tolerance hides the integrator's own error.** A harness
trusting CVODE's default tolerance would report the replica as 2.5 × 10⁻⁸ off;
all of that is the *reference's* error, and the replica's own contribution is
bounded at 3.5 × 10⁻¹⁴.

> **[W] Corrected twice, and the second correction is instance (7) catching the
> author of this section.** The table was first published with no committed
> producer. A re-measurement gave 3.251 × 10⁻⁹ for the oracle's own spread, and
> the section *withdrew* the published 2.515 × 10⁻⁸ as unreproducible. That
> withdrawal was wrong: the re-measurement ran in the **published** regime, the
> published table had been measured in the **aligned** one, and had not said so.
> Re-run in the aligned regime the oracle's spread is **2.515 × 10⁻⁸, to four
> figures the number that was withdrawn.** The number was right; it lacked a
> producer and a regime label, and the corrector supplied a third regime error on
> top of the two missing labels.

That episode also yields a result in its own right: **the oracle's tolerance
spread is not a constant of the instrument.** It is 2.515 × 10⁻⁸ in one regime
and 3.251 × 10⁻⁹ in the other — a factor of 7.7 from a change in initial density
of 5.7 × 10⁻⁶. Quoting one number for "the oracle's resolution" without its
regime is exactly the defect this instance describes.

### Defects with no syntactic signature

**(5) Integrated observables hide what per-reaction comparison exposes.** The
`reac − nu` defect was invisible to every Sounio↔replica pin and fell in one
shot to a per-reaction rate comparison against Cantera. That path was reproduced
here without knowing it was being repeated: the shipped reverse path matches
Cantera to 7.877 × 10⁻⁷ per reaction, the buggy one to 1.838 (§3.5).

**(6) A convention constant has no syntactic signature, so it is not
searchable.** The alignment of §2.5 changed constants at **30 sites**. One
survived: `adiabatic_init` in `examples/chemistry/h2_ignition_uq_demo.sio`,
twelve lines below the `demo_init` the same commit did change, still carrying a
truncated `R_SI = 8.314462618` against CODATA's `8.31446261815324` — 1.5 × 10⁻¹⁰
relative. Both are well-formed `f64` literals with the same type, dimension,
units and magnitude, differing only in digits a reader's eye compresses to
"8.31446…". The 30 sites were found by searching for the *old* spelling; a site
already half-migrated to a different wrong value matches neither the old pattern
nor the new one. **A constant is the one kind of program content whose
correctness is invisible to every tool that reads the program.** Detecting it
needs a value-level invariant — a unit-carrying type, a named constant with a
single definition site, or a gate that *recomputes* the constant rather than
matching its text.

**(7) Provenance of a number is a property of the git topology, and no type
system sees it.** The branch aligning the constants and the branch carrying the
TPX → TDY fix are **each internally correct**: every file in each is
self-consistent, both pass CI, and a reviewer reading either diff sees nothing
wrong, because nothing in either diff is wrong. The defect exists only in the
**ancestry** — the alignment branch is cut from a base that never carried the TDY
fix, so its copy of the parity script restores `gas.TPX` and deletes
`initial_concentrations`. Merging it, or building a release artefact from it,
silently reverts the only real defect this work fixed, and the result is a
*plausible* number, not a crash. No type system, test suite or review process
examines the topology of the graph that produced a working tree. The only
signature is numerical — the three-row table of §2.4, which identifies which
merge produced the tree that produced the result, **from the result alone**, with
ten orders of magnitude between the failing case and the passing ones. *A
numerical result inherits its meaning from a merge history that no static
analysis of the merged tree can recover.* The remedy is not a stronger type
system; it is an invariant printed alongside the result, chosen so that
different ancestries give different values.

**(8) A documentation number, never measured, propagates into a reviewer's
hypothesis.** The seven instances above are failures of the *authors* of a
measurement; this one is a failure of its **reviewer**, and it is the same
defect, which is why it belongs in the list rather than in an acknowledgement.
The reviewer raised a well-posed physical objection — that the two sides might
share an integrator, and the residual be RK4 truncation — which was correct to
raise and forced the measurements of §4.3. Its **premise** was a documentation
figure: "dt-convergence: dt = 1e-8 vs 5e-9 agree to 4 significant figures". Four
significant figures reads as a truncation error near 10⁻⁴, from which an
agreement at 2 × 10⁻¹¹ between two independent integrators looks impossible.
Measured, that self-convergence is **2.7 × 10⁻¹⁵ … 2.2 × 10⁻¹⁴ — about fifteen
significant figures, eleven orders from the documented four**. The documented
figure was about *ignition delays* (126.315 against 126.317 µs), a
phase-sensitive quantity at the front, and was never a statement about the
pre-front checkpoint at all. The pattern is identical to (1)–(4) one level up:
an unmeasured number of low resolution was taken as the resolution of a different
quantity, and the discrepancy was attributed to the object — here, the harness's
integrator, which was innocent. Two things follow. First, the reviewer's move was
right regardless: the hypothesis was falsifiable, was put in falsifiable form,
and was killed by measurement in one run; *a wrong hypothesis that names its own
test is worth more than a right one that does not.* Second, the defect reached
the reviewer through the same channel it reached the authors. **Prose reports
numbers without their resolution, and a reader supplies a plausible one.**

**(9) The archive layer failed silently, and reported it where no check could
see.** The eight instances above are failures *inside* the measurement; this one
is a layer further out, in the machinery meant to freeze the other eight, and it
is a new kind: **the failure was reported, but only on a page requiring the
owner's login, while every public signal read as success.** v1.0.0 of the frozen
snapshot received its DOI 45 seconds after the GitHub release. v1.0.1 — the
version carrying every remediation — received none in ten minutes, and would
never have: its `.zenodo.json` had gained two fields the schema forbids, both
added while inserting the ORCID and DOIs the archive was asked to carry — a
`related_identifiers` relation `isVersionOf`, which does not exist (the legacy
deposit schema enumerates 33, `isNewVersionOf` among them), and a top-level
`version` key, which `additionalProperties: false` rejects outright. The GitHub
release published normally with its tarball and checksum; the public API listed
one version of the concept and said nothing about a second. Every check that
existed had passed — the snapshot verifier, the provenance auditor, and the
release workflow's own guard, because the *tree* was correct. **A silent failure
with a private error message is, from the outside, indistinguishable from
success.**

Remedied in two rounds, because the first remedy had a blind spot of exactly the
kind it was built to remove. The schema is vendored into the snapshot and
`verify_snapshot.py` validates `.zenodo.json` against it **offline, before a
release**. The first version of that check used full `jsonschema` validation when
the library was present and an enumeration of the `relation` vocabulary
otherwise; the author's machine lacked the library, the enumeration caught
`isVersionOf` and passed the file. The release workflow's guard runs on a runner
that *has* the library, ran full validation, and **refused the v1.0.2 tag** on
the `version` key the enumeration could not see. Both paths are now proved to
discriminate by negative control. *The last layer in a pipeline is the one no
earlier layer can check, and when it reports failure only to an authenticated
view, the public record certifies what it was asked to reject.* The remedy is to
pull that layer's contract — here, a schema — down to where it can be checked
before the irreversible step.

**(10) A cross-check went on printing agreement after its two sides stopped
sharing a state.** Instance (6) is about a constant that cannot be enumerated by
reading the program. This is what happened to the sites that enumeration missed,
and it is a distinct failure because of *where* they were — not in the code under
test, but in the **harnesses that measure it**. Three of them, found separately,
all with the same shape:

- `examples/chemistry/rep_adiabatic_bug.sio` held `1.0/(82.057*t)` while the
  module it cross-checks had moved to `P0/(R_SI·T)·10⁻⁶`. Found by re-running it
  after the merge. Cost: the dT/dt delay at 1100 K, 701602 → 701597 ns, so the
  published −2.375 % becomes **−2.374 %** once both sides share the state.
- `benchmarks/chemistry/rep_prodfix.py`, same shorthand. Found by the snapshot
  verifier, not by any gate in the tree. Cost: the shipped reverse path's floor,
  9.204 × 10⁻¹⁵ → **8.442 × 10⁻¹⁵**; the buggy column did not move by a single
  bit, because R16's forward term depends only on the fixed radical seeds.
- `benchmarks/chemistry/gri30_full_cantera_parity.py`, same shorthand, putting
  *the oracle* at a different initial density from the module it is the oracle
  for — the one thing an oracle may not do. Found by re-running it and finding it
  print the published-regime pressure against a section measured in the aligned
  one; the section had been measured from a **working copy that was never
  committed**.

The first of the three carried a comment asserting that the two sides were "kept
identical". The comment was true when written and false when read, and nothing in
the tree marked the transition.

This is not simply (6) recurring. In (6) the instrument was a `grep`, and it
failed because a truncated literal has no signature. Here the instrument was
**the cross-check itself**, and it failed in a way a cross-check is supposed to
be immune to: its whole warrant is that two independently constructed quantities
agree. All three went on agreeing — to every digit they print — while charging
*different initial states*, because 5.7 × 10⁻⁶ is below the printed resolution of
almost everything either side reports. The one quantity that moved at all was the
most ill-conditioned number in the work, a delay read off a broad dT/dt peak, and
it moved by 5 ns in 701 µs.

*Agreement between the two sides of a cross-check is evidence about the two
computations only if the two sides are charged from the same state, and nothing
in the agreement itself establishes that they were.* A cross-check that does not
print its own initial-state deviation is reporting a comparison whose premise it
never checked.

The scope of the existing remedy is narrower than that sentence, and is stated
rather than rounded up: `initial_state_deviation` is implemented **oracle-side
only**, in the two Cantera parity harnesses, and the replicas and Sounio modules
carry no counterpart. None of the three divergences above was caught by it —
they were found by re-running and by the snapshot verifier. The invariant is
what these instances argue *for*; it is not a gate that already covers them.

The residual honest statement: these three were found by re-running and by the
snapshot verifier, one at a time, over three days. **No gate in the tree would
have caught any of them**, and the count reaching ten this way is itself the
measurement — the sweep of §2.5 reported 30 sites and closed, and four more
surfaced afterwards, one of them in the demo the reproduction section tells a
reader to run first.

### The hazard that is not an instance

**Sharing the integrator would hide the integrator's error**, and would do it
invisibly, because agreement *improves*, which reads as confirmation. Checked,
and it does not occur here: all three Cantera harnesses call `IdealGasReactor` +
`ReactorNet` + `net.advance(t_end)`, and the replica's `rk4_step` calls only its
own `dc_dt` (§2.3). Two independent integrations, no contact. Recorded as a
hazard to guard, not as a finding, because reporting it as observed would be
inventing it.

---

## 7. Discussion

### 7.1 What stands unqualified, and what is bounded

- **Unqualified.** The published-regime gap of 2.660 × 10⁻⁶ is real, is one
  rounded constant, and is **180,000× the oracle's resolution**. Its existence,
  its magnitude and its attribution are far above every noise source measured
  here. The falsification of the truncation hypothesis is likewise unqualified:
  ratio 1.000 under halving, in both regimes, across a factor of four in step.
- **Bounded [B].** After alignment the residual is 2.074 × 10⁻¹¹; the oracle's
  own uncertainty, over an ensemble of initial states, spans 3.730 × 10⁻¹² to
  4.142 × 10⁻¹¹; the replica accounts for at most 3.465 × 10⁻¹⁴. The residual
  lies *inside* the oracle's noise band. **No part of it can be attributed to a
  real difference between the two implementations, and no part can be excluded
  from being one.** The admissible statement is: *with the constants aligned,
  replica and Cantera agree to within the oracle's resolution.* Not "to
  2 × 10⁻¹¹"; not "at the floor of `rtol`"; **to within the resolution of the
  instrument used to compare them**. Resolving the last ~6.6 × 10⁻¹² would need a
  reference at least an order tighter — Cantera at `rtol = 10⁻¹⁴`, whose own
  floor is 6.068 × 10⁻¹² and so would not suffice either, or an
  arbitrary-precision integration of the same mechanism. Neither is done here.

### 7.2 An implication for any validation against an adaptive solver

The resolution of an adaptive integrator is not a smooth function of the initial
state (§4.5). CVODE re-selects its step sequence under a perturbation of one part
per million, and the re-selection moves the tolerance-induced error by an order
of magnitude with no monotone trend. Two consequences for practice:

1. A single measurement of "the oracle's floor" is a sample from a distribution,
   not a property of the tolerance. It must be measured **in regime, every time,
   over an ensemble**.
2. Any residual within about one order of that ensemble's upper end is a bound
   on disagreement, not a measurement of agreement, and should be reported as
   such.

### 7.3 The favourable checkpoint

The parity comparison is made at the single sampled point where the integrator
is under the least stress (§4.7). That is the right choice for isolating a
constant from an integrator and the wrong one for claiming the integrator has
been validated. The distinction is worth naming because the two claims are
routinely conflated, and the second is the more attractive.

### 7.4 In what sense the oracle was "wrong"

The title's claim is not that Cantera is wrong. It is that **the designation of
oracle is a choice, and the choice was wrong in one of the two comparisons
reported here.** For the trajectory, Cantera is the oracle and the replicas are
diagnostics; that orientation holds throughout §3 and §4. For the uncertainty
band, the Python replica was the *de facto* oracle — the reference implementation
against which the Sounio module's step-invariance would have been judged — and it
is wrong, demonstrably and for a derivable reason, machine-checked in §5.5. A
reviewer who carried the first orientation into the second would have reported
the Sounio module as defective for a property — step-invariance — that its
derivation shows the reference implementation itself lacks; whether the
Sounio module's band is otherwise correct, magnitude included, is a separate
question §7.5 leaves open.

### 7.5 Threats to validity

- **One mechanism, one checkpoint, one oracle.** Everything here is GRI-Mech 3.0
  at a pre-front, constant-volume checkpoint against CVODE. The truncation curve
  of §4.7 shows the bound is horizon-dependent by nearly three orders.
- **The GRI-Mech regression convention is not established.** Whether GRI-Mech
  3.0's rate parameters were themselves regressed under a particular rounding of
  R could not be determined from available documentation. The alignment was made
  with that uncertainty explicit, not because it was closed. If the mechanism was
  regressed under the rounded value, then aligning to Cantera's makes *this*
  comparison better and the mechanism's own self-consistency worse.
- **[U] Items left unexplained**, and deliberately not forced into agreement: the
  −34 % d[HO2]/dt figure (§3.5); the exponent of the step-refinement growth, 2.1×
  to 6.6× per halving where systematic accumulation predicts 2× (§4.6); the
  location of the minimum of the total-error curve, bracketed but not pinned.
- **Step-invariance is reported as ratios, not magnitudes, on the H/O
  problem.** Table 10 shows the Sounio band's ratio under a factor-4 step
  change is 1.000000 ± 10⁻⁶, against the replicas' 1.9999–2.0084 — but a ratio
  alone cannot exclude an implementation that dropped a persistent-parameter
  term entirely: H2 and O2 are ratio-1 in *every* table here (Tables 9 and 10
  alike) because their band is dominated by the 1 % initial-condition seed,
  which is step-invariant regardless of whether persistent-parameter UQ is
  computed at all. `test_g30_epistemic_step_invariance`'s own tolerance
  (10⁻²) is not tight enough to rule this out either. What would: band
  *widths*, not ratios, for all eight species against a finite-difference or
  GBS-referee magnitude at one dt. Not measured here.
- **One benchmark is deliberately left divergent.** `flame1d_replica.py` keeps
  the truncated `R = 8.314462618`, because it is a different benchmark with its
  own published reference values, and aligning it would move those numbers by an
  amount not measured here. Changing a separate benchmark's published results as
  a side effect of a chemistry-constant fix is exactly the unmeasured widening
  this work argues against everywhere else. The divergence is recorded so that it
  is not read as an oversight.

### 7.6 Self-correction as a reported result

Four statements in the source documentation were corrected in the course of this
work, and three statements *of this work* were corrected by it. The latter are
reported in place rather than silently fixed, because they are instances of the
phenomenon the work is about:

| corrected | what happened |
|---|---|
| §4.5 partition | a 68 %/32 % split was a single sample from a distribution that swallows it; replaced by an interval **[W]** |
| §6 instance (4) | a published figure was withdrawn for four hours on a re-measurement made in the wrong regime, then reproduced to four figures in the regime it was measured in **[W]** |
| §6 preamble | an earlier revision called instances (5)–(10) undetectable, and the same document then built an instrument that catches one **[W]** |

The provenance auditor (§8.3) found **nine** sections reporting high-precision
numbers without a producer present in the released tree, including one inside the
work that was correcting exactly that pathology. Every one was remediated by
supplying a committed producer, not by softening the claim: no published number
was ultimately withdrawn.

---

## 8. Reproducibility and data availability

### 8.1 The snapshot

`Sounio-lang/sounio-gri30-crossvalidation`, Apache-2.0. **Cite the concept DOI
[10.5281/zenodo.22236607](https://doi.org/10.5281/zenodo.22236607)**, which
resolves to the latest version. Version DOIs: v1.0.0 →
[10.5281/zenodo.22236608](https://doi.org/10.5281/zenodo.22236608); **v1.0.3 →
[10.5281/zenodo.22263060](https://doi.org/10.5281/zenodo.22263060)**. v1.0.1 and
v1.0.2 carry no DOI, for the reason given in §6 instance (9), and the snapshot's
README says so rather than hiding it.

### 8.2 Commands, in order

```sh
git rev-parse HEAD           # 98aa8e4d5151bbc61815bf910b6c31c3d0789f5f
pip install 'cantera==3.2.0' 'numpy==2.4.6'      # the versions §2.7 records

export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/h2_ignition_uq_demo.sio        # ~90 s
python3 benchmarks/chemistry/gri30_h2_python_replica.py          # ~4 s
python3 benchmarks/chemistry/gri30_h2_cantera_parity.py          # ~1 s
python3 benchmarks/chemistry/gri30_full_python_replica.py        # ~20 s
python3 benchmarks/chemistry/gri30_full_cantera_parity.py        # ~1 s
python3 benchmarks/chemistry/gri30_full_cantera_uq_reference.py --jobs 4   # ~8 s

( cd benchmarks/chemistry/cpp && \
  g++ -std=c++23 -O2 -o band_crosscheck gri30_h2_band_crosscheck.cpp && \
  ./band_crosscheck ../gri30_h2_mechanism.json )               # ~6 min
```

The C++ block is parenthesised into a subshell so it does not leave the
working directory changed for what follows — an earlier revision `cd`'d in
place and left every subsequent command in this section resolving under
`benchmarks/chemistry/cpp/`, silently, which is exactly the kind of defect §6
catalogues **[W]**.

The Lean development does **not** build through Lake in either tree, and the
command originally printed here is withdrawn **[W]**. Measured, not inferred:

```sh
cd formal/lean4 && lake build SounioIndepComposition
# error: unknown target `SounioIndepComposition`     (exit 1)
```

`formal/lean4/` in the upstream repository carries a Lake project but not this
module's source — only stale `.lake/build/` artefacts from a build made
elsewhere, which is worse than absence because it looks like evidence.

**It is nevertheless reproducible, standalone, against the frozen snapshot**,
which is what this block now prints:

```sh
elan toolchain install leanprover/lean4:v4.33.0    # the pin in formal/lean4/lean-toolchain
elan run leanprover/lean4:v4.33.0 lean formal/SounioIndepComposition.lean
# snapshot v1.0.3, path relative to the repo root; exit 0, no diagnostics
```

`elan run <toolchain> <cmd>` is the command actually needed: `elan toolchain
install` only fetches a toolchain, it does not select it, and a bare `lean`
afterwards runs whatever `elan`'s default or an ambient override resolves to —
not necessarily v4.33.0. `elan run` pins the invocation explicitly, which is
what was verified.

Verified at 2026-09-22 under `Lean (version 4.33.0, x86_64-unknown-linux-gnu,
commit d8b18978322de05a8f3dba51ef03cf5461676c17, Release)`: the file compiles
clean, carries **15 `theorem` declarations and zero occurrences of `sorry`**,
and the per-theorem axiom table of §5.5 is produced by appending
`#print axioms Sounio.IndepComposition.<name>` for each of the fifteen and
re-running. Being Mathlib-free core Lean 4, it needs no Lake project and no
dependency fetch — which is why the absence of one is not an obstacle.

> **[W] Corrected 2026-09-22, and the error was mine, not the record's.** The
> first version of this paragraph said no substitute invocation could be offered
> because "the Lean toolchain is not installed". That was `command -v lake lean
> elan` returning nothing — and `elan` was installed all along, at
> `~/.elan/bin`, simply not on `PATH`, with the pinned toolchain v4.33.0 and
> v4.33.1 both present. A negative result from the wrong instrument was read as
> a property of the environment. That is §6 instance (1) committed by this
> manuscript about itself, and it cost the reader a reproduction path that
> existed.

Diagnostic probes: `rep_traj_bug.py` (§3.2, §3.3, §3.5), `rep_adiabatic_bug.py`
(§3.4), `rep_prodfix.py` (§3.5), `rep_1atm.py` (§3.6), `rep_tolerance.py`
(§4.3), `rep_resolution.py` (§4.5), `rep_floor_spread.py` (§4.5),
`rep_stagnation.py` (§4.6), `examples/chemistry/gbs_oracle.sio` (§4.6).

### 8.3 The provenance contract, and its result

```sh
python3 benchmarks/chemistry/audit_provenance.py            # audits the working tree
python3 benchmarks/chemistry/audit_provenance.py --tree .   # or an unpacked release
```

The auditor is committed and exits non-zero on a finding, so the contract is
checkable rather than asserted. Run against the release it reports **one declared
failure**, §7.5's `flame1d_replica.py`, which lives upstream only and is
deliberately not shipped. Run against the working branch it additionally reports
the C++23 section, whose producer lives in a sibling pull request — that
difference is the audit working, not a defect.

### 8.4 What is not in the snapshot, and why

- Six probe artefacts referenced by the original preprint
  (`h2_precision_probe.sio`, `h2_probe2.sio`, `full_probe.sio`, `band_sweep.sio`,
  `rep_prodfix.py`, `rep_1atm.py`) were **never committed to the repository**
  in their original, preprint-cited form: an object scan of the repository's
  history finds no blob matching the original content under those names in
  any tree reachable from any ref, and there are no stashes, worktrees or
  dangling objects. **[W] Corrected 2026-09-23, a reviewer finding.** That is
  not the same as the names being absent from the *current* tree: all six now
  exist at those same paths as committed reconstructions, each carrying a
  header stating verbatim that it is a reconstruction, not the original, and
  citing this section. `audit_provenance.py` therefore finds a producer
  present for sections that cite these six files — correctly, since the
  reconstruction is what a reader would actually run — which does not mean
  the *original* measurement is recovered. The wording to be used for the
  distinction: *reconstructed from the described protocol on 2026-09-01; the
  original artefacts were not recoverable from the repository history.* The
  reconstructions reproduce the **protocols**, not the originals.
- `flame1d_replica.py` and `stdlib/constants/physical.sio` are upstream-only
  (§7.5), so any command citing them must be run against the upstream tree, not
  an unpacked snapshot. This is recorded explicitly because a section citing
  files a reader does not have is precisely the defect §8.3 audits for — here it
  is intentional and scoped.
- The **Lean development** does not build through Lake in either tree: the
  upstream `formal/lean4/` has a Lake project but no
  `SounioIndepComposition.lean` source, and the snapshot has the source at
  `formal/SounioIndepComposition.lean` with no Lake project. The `lake build`
  command is withdrawn and replaced by the standalone `lean` invocation of
  §8.2, which is measured and does reproduce §5.5 in full **[W]**. What this
  cost is worth recording: the auditor of §8.3 passed the section, because its
  criterion is a *named file present in the released tree* and a `cd` into a
  directory that exists — holding a Lake project but not the module — satisfies
  it. The gap was found by a reviewer, not by the instrument built for it.
- This manuscript is not part of the snapshot; the snapshot is the measurement
  record and its producers.

---

## 9. AI disclosure (GAIDeT-ICMJE 2025)

Generative AI assistance (Claude, Anthropic) was used under the author's
direction for: drafting and revision of prose; implementation of the diagnostic
probes and the C++23 and Gragg–Bulirsch–Stoer cross-checks; and the Lean 4
development. All numerical results reported here were produced by executing the
named commands and were transcribed from their output; no value was generated,
interpolated, or carried forward by a language model. The author is responsible
for the design of the measurements, for every claim made from them, and for the
decisions recorded in §2.5 and §7.5.

**[W] Corrected 2026-09-23, a reviewer finding.** This section previously
pointed to a root-level `AI_DISCLOSURE.md` as where repository-level
disclosure "is maintained." Verified: no such file exists in this repository
— only artefact-specific disclosures under `tools/lsp/AI_DISCLOSURE.md` and
`tools/mcp/AI_DISCLOSURE.md`, neither of which covers this manuscript.
`CLAUDE.md` §9 directs that papers and external submissions "update
`AI_DISCLOSURE.md` per artefact," which presupposes a repository-level file
this repository does not yet have. Creating that file is a repository-wide
governance decision outside this manuscript fix's scope; this disclosure is
recorded here, in full, as the paper's own AI-disclosure statement, with the
gap flagged rather than papered over with a citation to a file that does not
exist.

The 2026-09-23 revision additionally used a second, independent language model
(xAI Grok 4.6) as an adversarial reviewer against the 2026-09-22 manuscript
text, under a hostile-reviewer task prompt asking for BLOCKER/MAJOR/MINOR/NIT
findings rather than commentary. Findings were not accepted at face value:
each was checked against the actual manuscript text, `RESULTS.md`, and the
named producer scripts before any correction was made, and at least one
finding (the Lean-theorem scope objection) was substantially disputed and
narrowed rather than accepted as stated, with the reasoning given in place.
Logged in `.claude/llm_offload_log.md`.

---

## Appendices

### Appendix A — per-reaction exponents under the three forms

**[W] Corrected 2026-09-23, a reviewer finding.** This appendix previously
promised "the full 29-reaction table of correct and `reac − nu` exponents"
without including it — a table it did not contain cannot be called full, and
a reader had no way to inspect or reproduce the claimed per-reaction values
from the manuscript alone. `benchmarks/chemistry/rep_traj_bug.py` does not
itself print a per-reaction exponent table (it prints per-species deviations
under the three forms, §2.3–§2.5 of its own output); the 29-reaction exponent
table this appendix pointed to is not a committed artefact. §3.1's worked
case (`H + HO2 ⇌ O2 + H2`) remains the one exponent comparison this document
actually derives and shows. Reproducing the full per-reaction table would
require a dedicated script against `stdlib/chemistry/gri30_full.sio`'s
mechanism data, which does not yet exist — until it does, this appendix is a
pointer to that gap, not a table.

### Appendix B — tolerance ladders and the ten perturbed states

The full `rep_floor_spread.py` output underlying the interval of §4.5:

| δ | floor | δ | floor |
|---|---|---|---|
| −1.00e-06 | 6.281e-12 | +1.11e-07 | 2.214e-11 |
| −7.78e-07 | 8.964e-12 | +3.33e-07 | 8.260e-12 |
| −5.56e-07 | **4.142e-11** | +5.56e-07 | 1.104e-11 |
| −3.33e-07 | **3.730e-12** | +7.78e-07 | 7.592e-12 |
| −1.11e-07 | 8.728e-12 | +1.00e-06 | 1.045e-11 |

The table is not sorted by floor, because it cannot be: there is no order to
recover.

### Appendix C — RK4 order at coarse steps

Successive-error ratios against CVODE at `rtol = 10⁻¹³`, which must be 2⁴ = 16:

| pair | H2 | H | O | O2 | OH | H2O | HO2 | H2O2 |
|---|---|---|---|---|---|---|---|---|
| 8e-7/4e-7 | 15.53 | 15.44 | 15.57 | 15.64 | 16.60 | 15.46 | 20.36 | 17.89 |
| 4e-7/2e-7 | 15.72 | 15.68 | 15.73 | 15.78 | 16.27 | 15.69 | 18.26 | 16.99 |
| 2e-7/1e-7 | 15.19 | 15.17 | 15.15 | 15.23 | 15.75 | 15.15 | 17.49 | 16.62 |

The order test fails at dt ≤ 10⁻⁸ only because the differences fall below the
roundoff floor — which is itself the finding that truncation there is ~10⁻¹²,
not that the method misbehaves.

### Appendix D — statements of the 15 Lean theorems

Transcribed from `formal/SounioIndepComposition.lean` (snapshot v1.0.3), not
re-derived; each carries its Lean signature and the file's own docstring
gloss. §5.5 gives the axiom table; this appendix gives the statements
themselves, which that section names but does not reproduce. Built under
`leanprover/lean4:v4.33.0` (commit `d8b18978322de05a8f3dba51ef03cf5461676c17`);
15 theorems, zero `sorry`, reproduced by the standalone command in §8.2.

Three definitions the statements are built on: `varGeneral v₁ v₂ cov :=
v₁ + v₂ + 2·cov` (JCGM eq. 13, the general combination law, in variance form);
`varQuadrature v₁ v₂ := v₁ + v₂` (JCGM eq. 10, the independence-only law);
`varAdditive v₁ v₂ p := v₁ + v₂ + 2·p`, with `p` the product `u₁·u₂` kept as
an atom so no square root is constructed.

**§1 — quadrature is exactly the independence law.**

1. `quadrature_iff_zero_covariance (v₁ v₂ cov : Int) : varQuadrature v₁ v₂ =
   varGeneral v₁ v₂ cov ↔ cov = 0` — quadrature agrees with the general law
   precisely when the covariance vanishes; it is the ρ = 0 case, not an
   approximation that is usually fine.
2. `quadrature_sound_of_independent (v₁ v₂ cov : Int) (h : cov = 0) :
   varQuadrature v₁ v₂ = varGeneral v₁ v₂ cov` — restated in the direction a
   compiler needs: given independence, the tight bound is sound, the only
   hypothesis under which it is.

**§2 — with positive covariance, quadrature is unsound.**

3. `quadrature_understates_of_positive_covariance (v₁ v₂ cov : Int) (h : 0 <
   cov) : varQuadrature v₁ v₂ < varGeneral v₁ v₂ cov` — positively correlated
   inputs make quadrature report a strictly smaller variance than the truth,
   a bound tighter than the truth being the unsound direction.
4. `quadrature_sound_iff_nonpositive_covariance (v₁ v₂ cov : Int) :
   varGeneral v₁ v₂ cov ≤ varQuadrature v₁ v₂ ↔ cov ≤ 0` — the converse:
   quadrature is sound exactly when the covariance is non-positive, so
   assuming independence without proof buys the tight bound on an unchecked
   premise.

**§3 — the additive bound is never wrong, only wide.**

5. `additive_sound (v₁ v₂ cov p : Int) (hcs : cov ≤ p) : varGeneral v₁ v₂
   cov ≤ varAdditive v₁ v₂ p` — with `p = u₁·u₂`, Cauchy–Schwarz (`cov ≤ p`,
   i.e. ρ ≤ 1) makes the additive bound sound for every admissible
   correlation.
6. `additive_tight_at_unit_correlation (v₁ v₂ p : Int) : varGeneral v₁ v₂
   p = varAdditive v₁ v₂ p` — the additive bound is tight at ρ = +1, hence
   the least sound upper bound available without an independence proof.
7. `quadrature_below_additive (v₁ v₂ p : Int) (hp : 0 ≤ p) : varQuadrature
   v₁ v₂ ≤ varAdditive v₁ v₂ p` — quadrature sits below the additive bound
   whenever `u₁·u₂` is non-negative, so swapping the default from quadrature
   to additive can only widen a reported band, never narrow it.

**§4 — the N-step accumulation law, the bridge to measurement.** Two further
definitions: `varQuadN n u := n·u²` (N equal-uncertainty contributions
combined in quadrature); `varCorrN n u := (n·u)²` (the same N contributions
fully correlated, ρ = +1, so the uncertainties add).

8. `quadrature_understates_correlated_sum (n : Nat) (u : Int) : varCorrN n
   u = n · varQuadN n u` — **the underestimation law**: for N fully-correlated
   contributions the true variance is exactly N times the quadrature
   variance, so the true uncertainty is √N times the quadrature uncertainty.
   With N = T/dt this is the √(T/dt) law of §5.2; read along dt it is the
   √dt dependence of §5.1.
9. `accumulation_agrees_at_one_step (u : Int) : varCorrN 1 u = varQuadN 1
   u` — the degenerate reading that makes the law easy to miss: at N = 1 the
   two agree exactly, so a single composition is no evidence that repeated
   composition is sound.

**§5 — d-separation: the collider row.** `Junction` is `chain | fork |
collider` (Pearl's classification of how two edges meet at a middle node);
`active : Junction → Bool → Bool` gives whether a path through that junction
is unblocked as a function of whether the middle node is conditioned on:
chain and fork return `!conditioned`, collider returns `conditioned` — the
inversion is the whole content of d-separation (Berkson 1946).

10. `chain_blocked_by_conditioning : active .chain true = false`
11. `fork_blocked_by_conditioning : active .fork true = false` — conditioning
    on the middle node blocks a chain or a fork.
12. `collider_blocked_marginally : active .collider false = false` — a
    collider path is already blocked without conditioning.
13. `collider_opened_by_conditioning : active .collider true = true` — **the
    discriminating case**: conditioning on a collider opens the path, the
    one row a reachability check with a blocklist gets wrong.
14. `collider_inverts_the_others (b : Bool) : active .collider b = !(active
    .chain b)` — the inversion stated as the asymmetry itself: at every
    junction the collider is active exactly when the other two are not.
15. `conditioning_not_monotone : ∃ j : Junction, active j false = false ∧
    active j true = true` — conditioning is therefore not monotone in the
    blocking direction: there is a junction where adding to the conditioning
    set turns a blocked path active, which is unsound for any implementation
    whose search only ever removes edges as the conditioning set grows.

### Appendix E — the sections remediated by the provenance audit

**[W] Corrected 2026-09-23, a reviewer finding.** This appendix previously
implied it contained the nine-section list §7.6 refers to ("with the producer
supplied for each") without actually including a list, mapping, or per-section
producer — an appendix that names no entries cannot substantiate which nine
sections were found or what fixed each one. `audit_provenance.py`
(§8.3), run against the current tree, reports 0 FAIL — every section it
checks now names a producer present in the released tree — but the tool has
no mode that lists which sections historically failed before remediation, so
that specific nine-item history is not reconstructable from this document or
its producers as they stand. §7.6's aggregate count (**nine**, found and all
remediated) is what this document can actually support; this appendix is a
cross-reference to that claim and to the reproducible command below, not an
itemized list.

### Appendix F — the step-refinement anomaly (moved from §4.6)

Replica self-difference at three step sizes, `|c(1e-8) − c(5e-9)|` against
`|c(5e-9) − c(2.5e-9)|`:

| species | \|c(1e-8)−c(5e-9)\| | \|c(5e-9)−c(2.5e-9)\| | ratio |
|---|---|---|---|
| H2 | 1.823e-16 | 2.188e-15 | 0.083 |
| H | 2.537e-15 | 3.247e-14 | 0.078 |
| O | 2.705e-15 | 3.218e-14 | 0.084 |
| O2 | 2.683e-15 | 8.943e-16 | 3.000 |
| OH | 1.283e-14 | 3.293e-14 | 0.390 |
| H2O | 2.809e-16 | 3.905e-14 | 0.007 |
| HO2 | 3.411e-15 | 7.580e-16 | 4.500 |
| H2O2 | 3.465e-14 | 2.626e-14 | 1.320 |

Not one ratio is near 16; halving the step makes the self-difference **worse**
for five of eight species. Resolved as to mechanism in §4.6; **[U]** as to
exponent.

Gragg–Bulirsch–Stoer depth sweep (self-difference between macro-steps
H = 10⁻⁶ and 5 × 10⁻⁷, worst over species):

| depth | order | seq 2,4,6,8,10,12,14,16 | seq 2,4,6,8,12,16,24,32 |
|---|---|---|---|
| 3 | 6 | 5.730e-09 | 5.730e-09 |
| 4 | 8 | 8.413e-11 | 8.413e-11 |
| 5 | 10 | 8.942e-13 | 6.257e-13 |
| 6 | 12 | **5.734e-14** | **1.421e-14** |
| 7 | 14 | 3.066e-13 | 2.314e-14 |
| 8 | 16 | 2.581e-13 | 1.556e-14 |

Per-species resolution of the chosen instrument (depth 6, wider sequence): H2
6.198e-15, H 1.421e-14, O 3.702e-15, O2 5.007e-15, OH 5.910e-15, H2O 1.334e-14,
HO2 2.273e-15, H2O2 3.107e-15.

Halving ladder against that independent method, relative distance × 10¹⁸:

| dt | H2 | H | O | O2 | OH | H2O | HO2 | H2O2 |
|---|---|---|---|---|---|---|---|---|
| 1e-8 | 10755 | 4904 | 4841 | 357 | 16380 | 5898 | 2652 | 32629 |
| 5e-9 | 10573 | 7442 | 1851 | 2861 | 3546 | 5618 | 757 | 1709 |
| 2.5e-9 | 8385 | 40254 | 30185 | 2146 | 29552 | 45365 | 1705 | 24083 |
| 1.25e-9 | 729 | 63257 | 78596 | 13056 | 77005 | 60955 | 14212 | 73805 |

Step-stagnation test (per-step increment in units of half an ULP of the state;
stagnation requires < 1):

| species | dt = 1e-8 | dt = 5e-9 | dt = 2.5e-9 |
|---|---|---|---|
| H2 | 8.29e+11 | 4.15e+11 | 2.07e+11 |
| H2O | 9.07e+12 | 4.54e+12 | 2.27e+12 |
| HO2 | 5.04e+11 | 2.52e+11 | 1.26e+11 |
| *(all eight)* | > 1e+11 | > 1e+11 | > 1e+11 |

### Appendix G — the undeclared delay criterion (moved from §3.4)

Error in ignition delay introduced by the `reac − nu` defect, under the two
criteria the replica exposes:

| T₀ (K) | error, d[H2O]/dt criterion | error, dT/dt criterion |
|---|---|---|
| 1100 | **+0.096 %** | **−2.374 %** |
| 2000 | **+8.552 %** | +9.947 % |

At 1100 K the sign flips. A defect that shifts the H2O-rate delay by 0.1 %
shifts the temperature-rise delay by 2.4 % *the other way*, which says the dT/dt
peak is broad and ill-conditioned there at this step, not that the defect
accelerates ignition. The documented anchors — 0.1 % and 8.6 % — are reproduced
**on the d[H2O]/dt criterion only**, and the documentation does not say which
criterion it used. That omission is a documented ambiguity rather than a silent
assumption, and it is the kind of unstated protocol variable §6 instance (8) is
about.

> The dT/dt figure at 1100 K was −2.375 % until 2026-09-03; the change is
> §6 instance (10), not a rounding.

---

## Mapping to the measurement record

| manuscript | `benchmarks/chemistry/RESULTS.md` |
|---|---|
| §2.4 | §1.4 |
| §2.5 | §1.5 |
| §3.1–3.3 | §2.1–2.5, §2.7 |
| §3.4, App. G | §2.6 |
| §3.6 | §3, §3.1, §3.2 |
| §4.1 | §1.1a, §1.2, §1.3 |
| §4.2 | §1.5 |
| §4.3 | §7.3, §7.5b |
| §4.4 | §6, §6.1, §6.2b |
| §4.5, App. B | §7.5, §7.7 |
| §4.6, App. F | §7.7 |
| §4.7 | §7.6 |
| §5 | §5.1–5.5, §6.2 |
| §6 | §6.3 |
| §7.5 | §6.4, §8 |
| §7.6, §8.3 | §7.4 |
| §8.4 | §4 |
