# GRI-Mech 3.0 cross-validation — measured results

**Every number on this page was produced by the command printed above it, at
commit `98aa8e4d5151bbc61815bf910b6c31c3d0789f5f` (branch `claude/gri-mech-cantera-preprint-776qb9`), on
2026-09-01.** Nothing is carried forward from an earlier log, a prior session,
or a draft. Where a run was not performed, the row says so.

Environment: Linux x86-64, Python 3.11, `cantera 3.2.0`, `numpy 2.4.6`,
`g++ (Ubuntu 13.3.0) -std=c++20 -O2`. Sounio compiler: the committed ELF
`bin/madaros-linux-x86_64` (md5 `ff69dae4`, tree `98aa8e4d`), run under
`SOUNIO_SOUC_ENGINE=lean_single`.

> **Frozen relative to PR #1758.** PR
> [Sounio-lang/sounio#1758](https://github.com/Sounio-lang/sounio/pull/1758)
> ("Independência na composição: quadratura passa a exigir prova de
> d-separação") is OPEN and unmerged at this commit. It is the language-level
> remedy for the uncertainty-accumulation defect measured in section 5 below.
> These results describe the pre-remedy behaviour and must be re-measured if
> #1758 lands.

---

## 1. The parity contradiction — resolved

### 1.1 What was claimed

Two incompatible descriptions were in circulation for the H/O pre-front
checkpoint (T = 1500 K, t = 1e-4 s):

- **(A)** majors within 0.2–2%, radicals ~3%, H2O2 ~16%, attributed to
  fixed-step RK4 versus CVODE;
- **(B)** agreement to 5–6 significant figures, deviations 2e-7 to 6e-6.

**Claim (A) does not appear anywhere in this repository.** It is absent from
the current `benchmarks/chemistry/README.md` and from every version of that
file reachable from any ref in this clone:

```sh
git log --all --format='%h' -- benchmarks/chemistry/README.md | while read h; do
  git show "$h:benchmarks/chemistry/README.md" | grep -qiE "H2O2.*1[0-9]%|~?16 ?%" && echo "$h"
done          # -> no output
```

No pairing measured below produces percent-level deviations. Claim (A) is
**not reproduced and has no provenance in this repository**; it should not be
cited.

### 1.2 The three-way measurement

```sh
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/h2_ignition_uq_demo.sio      # Sounio
python3 benchmarks/chemistry/gri30_h2_python_replica.py        # replica
python3 benchmarks/chemistry/gri30_h2_cantera_parity.py        # Cantera oracle
```

Absolute concentrations at T = 1500 K, t = 1e-4 s, mol/cm³:

| species | Sounio (native) | Python replica (RK4 dt=1e-8) | Cantera 3.2 (CVODE) |
|---|---|---|---|
| H2   | 1.45202682104e-07 | 1.4520268210447984e-07 | 1.452026406661632e-07 |
| H    | 9.781440907e-09   | 9.7814409074473e-09    | 9.781464042547194e-09 |
| O    | 1.452395690e-09   | 1.4523956900503582e-09 | 1.452399552929886e-09 |
| O2   | 7.3998984676e-08  | 7.399898467680751e-08  | 7.399896700947298e-08 |
| OH   | 1.224596458e-09   | 1.2245964583232146e-09 | 1.224599671250092e-09 |
| H2O  | 1.1779287709e-08  | 1.1779287709403675e-08 | 1.177931597531644e-08 |
| HO2  | 1.7051211792e-11  | 1.7051211792171253e-11 | 1.705120796024241e-11 |
| H2O2 | 1.62467687132e-13 | 1.6246768713279443e-13 | 1.624679943884135e-13 |

Relative deviations, per pairing:

| species | Sounio vs replica | Sounio vs Cantera (as-shipped, TPX) | Sounio vs Cantera (documented, TDY) |
|---|---|---|---|
| H2   | 3.305e-12 | 1.739e-06 | 2.854e-07 |
| H    | 4.573e-11 | 3.851e-05 | 2.365e-06 |
| O    | 3.467e-11 | 3.922e-05 | 2.660e-06 |
| O2   | 1.091e-11 | 2.435e-06 | 2.387e-07 |
| OH   | 2.639e-10 | 3.808e-05 | 2.624e-06 |
| H2O  | 3.427e-11 | 3.911e-05 | 2.400e-06 |
| HO2  | 1.004e-11 | 8.918e-06 | 2.247e-07 |
| H2O2 | 4.890e-12 | 3.705e-05 | 1.891e-06 |
| **range** | **3.3e-12 … 2.6e-10** (print-limited; 1.8e-16 … 4.6e-15 at full precision, see 1.3) | **1.7e-06 … 3.9e-05** | **2.2e-07 … 2.7e-06** |

Replica vs Cantera is identical to Sounio vs Cantera to three significant
figures in every cell, because Sounio and the replica agree ~10⁴× more
tightly than either agrees with Cantera.

### 1.3 Verdict

The working hypothesis — that the tight figure is Sounio-vs-replica and the
loose one is Sounio-vs-Cantera — is **half right, and both halves are off in
magnitude**:

1. **Sounio vs the Python replica: 1.8e-16 to 4.6e-15 — 1 to 30 ULP.** Not
   5–6 significant figures; **15**. The demo's apparent 3.3e-12 … 2.6e-10 was
   **entirely a print-resolution artefact**, which is what
   `examples/chemistry/h2_precision_probe.sio` was reconstructed to settle
   (section 6). Re-printing the same checkpoint at 16 digits gives:

   | species | Sounio (16 digits) | Python replica | rel dev | ULP |
   |---|---|---|---|---|
   | H2   | 1.45202682104479811e-07 | 1.45202682104479838e-07 | 1.823e-16 | 1 |
   | H    | 9.78144090744727741e-09 | 9.78144090744730058e-09 | 2.368e-15 | 14 |
   | O    | 1.45239569005035204e-09 | 1.45239569005035824e-09 | 4.271e-15 | 30 |
   | O2   | 7.39989846768074966e-08 | 7.39989846768075098e-08 | 1.789e-16 | 1 |
   | OH   | 1.22459645832320899e-09 | 1.22459645832321457e-09 | 4.559e-15 | 27 |
   | H2O  | 1.17792877094036405e-08 | 1.17792877094036753e-08 | 2.949e-15 | 21 |
   | HO2  | 1.70512117921712495e-11 | 1.70512117921712527e-11 | 1.895e-16 | 1 |
   | H2O2 | 1.62467687132793797e-13 | 1.62467687132794429e-13 | 3.884e-15 | 25 |

   The two implementations share the integrator (RK4), the step (dt = 1e-8),
   the mechanism JSON and the summation order, so the only residual is the
   transcendental-function implementations (`exp`, `log`, `pow`) — and at
   1–30 ULP against a double-precision eps of 2.220e-16, that is exactly what
   is measured. There is no cross-language discrepancy to explain.
2. **Sounio vs Cantera: 2.2e-07 to 2.7e-06** under the protocol the README
   documents — a **5-to-6-significant-figure** agreement, not 0.2–2%. The
   published claim of "2e-7 through 6e-6" is **confirmed** for the major
   species and radicals. But its stated *cause* is wrong: this is **not** the
   fixed-step-RK4-versus-CVODE difference. RK4 truncation at dt = 1e-8 is
   2.3e-14, seven orders of magnitude too small. The gap is one rounded
   activation-energy gas constant, and it collapses to ~1e-11 when the same
   constant is used on both sides. See section 1.5.
3. The published claim that **"H2O2 agrees to 5.9e-3"** is **not reproduced**.
   H2O2 agrees to **1.891e-06** — three orders of magnitude better than
   stated. That figure is stale and is superseded here.

### 1.4 A real defect found on the way: the parity script did not implement its own documented protocol

The as-shipped `gri30_h2_cantera_parity.py` initialised Cantera with
`gas.TPX = T, P0, X`. The README documents initialisation "through `TDY` so
those concentrations are not renormalized", and records that "the seed makes
the actual initial pressure 101325.576758 Pa".

`TPX` renormalises the mole fractions and pins P = 101325 Pa exactly.
Measured effect on the initial state:

```sh
python3 benchmarks/chemistry/gri30_h2_cantera_parity.py | head -4
```

| initialisation | worst deviation from intended initial concentrations | realised pressure |
|---|---|---|
| `TPX` (as shipped) | **5.692129e-06** (uniform, every species) | 101325.000000 Pa |
| `TDY` (as documented, now implemented) | **0.000000e+00** | **101325.576758 Pa** |

The TDY path reproduces the documented pressure to all six decimals, which
independently confirms it is the intended protocol.

A uniform −5.69e-06 initial-density error is amplified by chain branching to
**3.9e-05** in the radicals by t = 1e-4 s (amplification ≈ 6.9×; the
checkpoint sits at 100 µs against a 126 µs ignition front, i.e. inside the
exponential growth phase). That single line accounted for the entire
**~15× gap** between the as-shipped script's answer and the README's claim.

**Fixed at this commit.** `initial_state()` now sets unnormalised mole
fractions and then fixes the density, and asserts the realised deviation is
exactly 0.0. Ignition delays are unaffected (169.66 / 126.34 / 98.29 / 79.00 /
65.08 µs at 1400–1800 K, unchanged).

### 1.5 The residual gap to Cantera is one rounded constant, not the integrator

This was settled by the reconstructed `examples/chemistry/h2_probe2.sio`
(section 6), whose purpose is to separate RK4 truncation from everything else.

```sh
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/h2_probe2.sio
```

**Step 1 — how big is RK4 truncation at dt = 1e-8?** Halve the step to 5e-9
(20000 steps) and measure how far the checkpoint moves:

| species | dt = 1e-8 vs dt = 5e-9 |
|---|---|
| H2 | 2.917e-15 |
| H | 6.765e-15 |
| O | 1.381e-14 |
| O2 | 8.227e-15 |
| OH | 2.280e-14 |
| H2O | 5.899e-15 |
| HO2 | 1.289e-14 |
| H2O2 | 2.098e-14 |

**RK4 truncation at dt = 1e-8 is at most 2.3e-14.** The gap to Cantera is
2.2e-07 … 2.7e-06 — between **10⁷ and 10⁸ times larger.** The gap is therefore
**not** RK4-versus-CVODE.

**Step 2 — what is it, then?** The Arrhenius activation-energy gas constant.
The replica and the Sounio module use the CHEMKIN-conventional rounded value
`R = 1.9872041` cal/mol/K. Cantera converts cal → J at 4.184 exactly and
divides its own gas constant, giving:

```
8.31446261815324 / 4.184 = 1.9872042586408316 cal/mol/K
```

a relative difference of **7.983e-08**, sitting inside `exp(-Ea/(R·T))`.

**Step 3 — substitute Cantera's value and re-run the identical trajectory:**

| species | replica with R = 1.9872041, vs Cantera | replica with R = 1.9872042586408316, vs Cantera | improvement |
|---|---|---|---|
| H2   | 2.854e-07 | **1.082e-12** | 263,862× |
| H    | 2.365e-06 | **9.001e-12** | 262,784× |
| O    | 2.660e-06 | **9.197e-12** | 289,178× |
| O2   | 2.388e-07 | **8.952e-13** | 266,715× |
| OH   | 2.624e-06 | **8.867e-12** | 295,899× |
| H2O  | 2.400e-06 | **9.280e-12** | 258,573× |
| HO2  | 2.247e-07 | **1.246e-12** | 180,396× |
| H2O2 | 1.891e-06 | **1.220e-11** | 155,031× |

**The entire published parity gap is one rounded constant.** With the same R
on both sides, Sounio, the Python replica and Cantera agree to
**8.9e-13 … 1.2e-11** — at the floor set by CVODE's own `rtol = 1e-12`, and
about five orders of magnitude better than the figure the README publishes.

This is the same root cause as the 1.843e-11 Kc floor in section 3.1 (there it
was `R_SI = 8.314462618` against Cantera's `8.31446261815324`). Two independent
truncated gas constants, two residual floors, both fully accounted for.

**Not changed in the shipped code, deliberately.** `R = 1.9872041` cal/mol/K
is the rounded value the Sounio module and both replicas use, of the kind
conventional in CHEMKIN-family codes. *Whether GRI-Mech 3.0's published rate
parameters were themselves regressed under this or another rounding of R has
not been established here* — that would need the GRI-Mech regression
documentation, which is not in this repository. Until it is, retuning the
constant to Cantera's value would be changing a mechanism-semantics choice on
no evidence, in order to make one comparison look better. The correct action
is to **document** that exact agreement with Cantera requires Cantera's R, and
to stop attributing the residual to the integrator. Whether to align the
constant is a decision for the operator, not a defect to patch here.

---

## 2. Reverse-rate path in the H/O replica — no defect present

The reported defect was, in `rates_net()`:

```python
p = reac[r][s] - nu[r][s]     # reported as the buggy line
```

**This line does not exist and never existed in this repository.**

```sh
grep -n "reac\[r\]\[s\] - nu" benchmarks/chemistry/*.py           # -> no match
git log --all --format='%h' -- benchmarks/chemistry/gri30_h2_python_replica.py |
  while read h; do git show "$h:benchmarks/chemistry/gri30_h2_python_replica.py" |
  grep -q "reac\[r\]\[s\] - nu" && echo "BUG AT $h"; done      # -> no output
```

The H/O replica already reads `p = prod[r][s]` directly, exactly as the
full-mechanism replica does via `prod_nz`. The file entered the repository
already correct (added in `67dccf1e`, 2026-08-30); there was no back-port to
perform.

**Verified against the oracle rather than by inspection.** All 29 net rates of
progress, at T = 1500 K with a radical-loaded state chosen to exercise the
reverse path (H=1e-9, OH=1e-10, O=1e-10, H2O=1e-9, HO2=1e-12, H2O2=1e-13
mol/cm³), against `Cantera.net_rates_of_progress`:

| quantity | result |
|---|---|
| reactions compared | 28 non-zero of 29 (H + O2 + AR is exactly 0 both sides — no Ar present) |
| worst relative deviation | **7.877e-07** |
| median | ~4e-08 |
| pure-Arrhenius reactions (no falloff/third-body) | 1e-14 … 1e-16 |
| **H + HO2 <=> O2 + H2** (the initiation channel called out for watching) | replica 2.080890177001593e-08, Cantera 2.0808902365210865e-08, **rel 2.860e-08** |

The residual ~1e-07 floor is the CHEMKIN activation-energy gas constant: the
replica uses `R = 1.9872041` cal/mol/K where Cantera uses a marginally
different value. It is not a stoichiometry error.

**STEP 2 deltas: none. There was no patch to apply, so the checkpoint and the
1-σ band do not move.** The tables in section 1 are the unmodified replica.

---

## 3. Standard-state reference pressure — no defect present

**Measured, not assumed** — what Cantera itself reports for the GRI-Mech 3.0
NASA-7 polynomials:

```python
import cantera as ct
g = ct.Solution("gri30.yaml")
sorted({s.thermo.reference_pressure for s in g.species()})   # -> [101325.0]
```

`gri30.yaml` declares no `reference-pressure` key, so Cantera's default
applies, and it resolves to **101325.0 Pa (1 atm) for all 53 species** — not
1 bar.

Both replicas already use the correct value:

```sh
grep -n "^P0" benchmarks/chemistry/gri30_h2_python_replica.py \
              benchmarks/chemistry/gri30_full_python_replica.py
# gri30_h2_python_replica.py:30:P0 = 101325.0  # CHEMKIN/GRI standard state: 1 atm
# gri30_full_python_replica.py:46:P0 = 101325.0  # CHEMKIN/GRI standard state: 1 atm
```

The premise that "both replicas set P0 = 1.0e5 (1 bar)" is **false at this
commit**. Code and prose already agree, and they agree with the oracle.
**Decision: no change. There is no preserved defect to annotate.**

### 3.1 Kc verified against Cantera, all 29 reactions

```sh
python3 - < the harness in section 7.2
```

| Δn class | count | worst `|Kc_replica/Kc_Cantera − 1|` |
|---|---|---|
| Δn = 0  | 17 of 29 | 1.418e-14 (machine precision) |
| Δn ≠ 0  | **12 of 29** | **1.843e-11** |
| all | 29 | 1.843e-11 |

The 1.843e-11 floor appears on **exactly** the Δn ≠ 0 reactions and nowhere
else, because those are the only ones carrying the factor
`c0 = P0/(R·T)`. Its size is fully explained: the replica uses
`R_SI = 8.314462618` where Cantera uses `8.31446261815324`, a relative
difference of 1.84e-11. This is a documentation-precision artefact of one
constant, not a standard-state error.

### 3.2 The counterfactual, quantified

Had P0 been set to 1e5 Pa, Kc would scale by `(1e5/101325)^Δn`:

| Δn | Kc factor | error |
|---|---|---|
| −1 | 1.013250000 | **+1.3250%** |
| 0  | 1.000000000 | 0.0000% (exactly) |
| +1 | 0.986923267 | **−1.3077%** |

**The sharper claim, corrected.** The proposed formulation — that such a
defect would be "PRESENT but UNOBSERVABLE in a far-from-equilibrium
trajectory, becoming observable only where an equilibrium pins a population
(NNH, Δn = +1)" — is the right *shape* of argument, but its premise about the
H/O submechanism is wrong in a way worth stating precisely:

- It is **not** true that Δn ≠ 0 is rare in the H/O submechanism. **12 of 29
  reactions have Δn ≠ 0** (indices 0, 1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 21) —
  every three-body recombination and the 2 OH (+M) falloff. A 1.3% Kc error
  would be *present in 41% of the submechanism*.
- What makes it unobservable is **not** scarcity but **direction of flux**: at
  T = 1500 K in the induction period these recombinations run overwhelmingly
  forward, and Kc enters only the reverse term, which is smaller than the
  forward term by 6–15 orders of magnitude. A 1.3% error on a term that
  contributes ~1e-10 of the net rate cannot move the trajectory.
- The claim therefore becomes: **present in 12 of 29 reactions, but suppressed
  by the forward/reverse flux ratio in a far-from-equilibrium trajectory, and
  observable only where an equilibrium pins a population** — for which NNH
  (Δn = +1, `NNH <=> N2 + H`, k ≈ 3.3e8 s⁻¹ at 1500 K) in the full mechanism
  is the correct example, since its population is set by a fast quasi-
  equilibrium rather than by accumulated flux.

This remains a counterfactual. **No such defect exists in the code**, so no
magnitude is claimed for the shipped artefacts.

---

## 4. Missing probe artefacts

```sh
for f in h2_precision_probe.sio h2_probe2.sio full_probe.sio band_sweep.sio \
         rep_prodfix.py rep_1atm.py; do
  find . -name "$f" -not -path './.git/*'
  git rev-list --all --objects | grep "$f"
done
git stash list; git worktree list; git fsck --dangling
```

| artefact | in worktree | in any tree of any ref | in any dangling object | stash/worktree |
|---|---|---|---|---|
| `examples/chemistry/h2_precision_probe.sio` | no | no | no | none |
| `examples/chemistry/h2_probe2.sio` | no | no | no | none |
| `examples/chemistry/full_probe.sio` | no | no | no | none |
| `examples/chemistry/band_sweep.sio` | no | no | no | none |
| `benchmarks/chemistry/rep_prodfix.py` | no | no | no | none |
| `benchmarks/chemistry/rep_1atm.py` | no | no | no | none |

**None is recoverable.** They were never committed to this repository — the
object scan finds no blob under any of those names in any tree reachable from
any ref, and there are no stashes, no other worktrees and no dangling objects.
They exist only as references in the preprint.

See section 6 for what was reconstructed in their place, and how it is marked.

---

## 5. Band scaling — factor 2 vs factor 4

```sh
cd benchmarks/chemistry/cpp
g++ -std=c++23 -O2 -o band_crosscheck gri30_h2_band_crosscheck.cpp
./band_crosscheck ../gri30_h2_mechanism.json
```

This is an **independent third implementation** (C++23, no dependencies),
written from the published protocol rather than translated from either the
Sounio module or the Python replica. Its deterministic checkpoint reproduces
the Python replica **to all 17 printed digits on all 8 species** (e.g. H2
`1.45202682104479838e-07` vs `1.4520268210447984e-07`), which is what
licenses using it as the arbiter below.

### 5.1 Scaling in dt, at fixed T = 1e-6 s

Predicted: band ∝ dt·√(T/dt) = √(T·dt), so a factor **f** in dt gives **√f**.

| species | dt 4e-9 → 2e-9 (f=2, √f=1.414214) | dt 2e-9 → 1e-9 (f=2) | dt 4e-9 → 1e-9 (**f=4**, √f=**2.000000**) |
|---|---|---|---|
| H    | 1.414792 | 1.414503 | **2.001227** |
| O    | 1.414585 | 1.414399 | **2.000787** |
| OH   | 1.415047 | 1.414630 | **2.001769** |
| H2O  | 1.418167 | 1.416197 | **2.008405** |
| HO2  | 1.414183 | 1.414198 | **1.999935** |
| H2O2 | 1.417520 | 1.415872 | **2.007027** |
| H2   | 1.000000 | 1.000000 | 1.000000 |
| O2   | 1.000000 | 1.000000 | 1.000000 |

The Python sweep over the same three step sizes (using the replica's own
`propagate_unc`, driven by the STEP 5 harness) reproduces every ratio above to
all six printed digits — 1.414792 / 1.414585 / 2.001227 / 2.008405 and the
rest — so the two independent implementations agree exactly on this result.

### 5.2 Verdict

**The log is wrong, and the reasoning in the task is right.**

- A factor **4** in dt gives a measured ratio of **1.9999 – 2.0084**, i.e.
  **2.0**, matching √4 = 2. It does **not** give √2.
- A ratio of **√2 ≈ 1.4142** is what a factor **2** in dt gives — measured
  1.41418 – 1.41817 across two independent factor-2 pairs.
- The log entry "factor 4 in dt, ratio √2" therefore **conflates two
  different sweeps**. The ratio √2 is a correct measurement; the factor-4
  label attached to it is not. The correct pairing is *factor 2 → √2*.

**A refinement the √dt statement needs.** The law holds **only for species
whose band is generated by accumulation**. H2 and O2 give a ratio of exactly
1.000000 under every dt change, because their variance is dominated by the
1% initial-condition uncertainty seeded at t = 0, which does not accumulate
and does not scale with dt. Stating "the band scales as √dt" without that
qualifier is false for 2 of the 8 species reported.

### 5.3 Scaling in T, at fixed dt = 1e-8 s — the √(T/dt) law is NOT exact

The claim under test: "the underestimation law √(T/dt) is exact for dt = 1e-8
at T = 1e-6, 1e-5, 1e-4 (giving 10, 31.62, 100)."

The values 10, 31.62 and 100 are just √(T/dt) evaluated arithmetically; that
is true by construction and tests nothing. What matters is whether the
*measured band ratio* tracks them. Predicted ratio per decade: √10 = 3.162278.

| species | T 1e-6 → 1e-5 (predicted 3.162278) | T 1e-5 → 1e-4 (predicted 3.162278) |
|---|---|---|
| O    | 3.227593 | 577.01 |
| OH   | 3.532908 | 573.25 |
| HO2  | 3.552849 | 186.64 |
| H    | 6.150410 | 639.96 |
| H2O  | 15.342525 | 766.11 |
| H2O2 | 27.409671 | (overflowed the reported set) |
| H2   | 0.999927 | 0.891875 |
| O2   | 0.999927 | 0.908202 |

**The law is not exact over these decades — it is not even approximately true
past the first one.** Over the first decade only O, OH and HO2 land near
3.16 (2–12% high); H is 1.9× high, H2O 4.9× high, H2O2 8.7× high. Over the
second decade the measured ratios exceed the prediction by **59× to 242×**.

The mechanism is not mysterious: √(T·dt) describes the **pure quadrature
accumulation of the per-step parameter term with a frozen Jacobian**. It is
valid only in the induction period. Between t = 1e-5 and t = 1e-4 the
trajectory enters chain-branching growth (the ignition front is at 126 µs, so
t = 1e-4 s is at 79% of the delay), and the Jacobian terms
`2·J_ii·v_i·dt` and `Σ_k (J_ik·dt)²·v_k` dominate the parameter term
entirely. Exponential amplification, not quadrature, then sets the band.

**Correction to the framing.** The claim spans not "TWO decades" of validity
but **less than one**: it is already 8.7× off for H2O2 at the end of the first
decade, and it should be stated as a property of the quiescent limit, not as
a law holding across the sweep.

### 5.4 The Sounio native band is step-size invariant — the contrast, measured

```sh
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/band_sweep.sio
```

The same sweep on the **native Sounio** surface (T = 1500 K, t = 1e-6 s fixed,
dt = 4e-9 / 2e-9 / 1e-9), 1-σ bands in mol/cm³:

| species | dt = 4e-9 | dt = 2e-9 | dt = 1e-9 |
|---|---|---|---|
| H2   | 1.624884644233830e-09 | 1.624884644233830e-09 | 1.624884644233900e-09 |
| H    | 3.3546639095889176e-14 | 3.3546639095889176e-14 | 3.3546644941977180e-14 |
| O    | 4.4998255814766720e-14 | 4.4998255814766720e-14 | 4.4998257788051568e-14 |
| O2   | 8.124380305597014e-10 | 8.124380305597014e-10 | 8.124380305596639e-10 |
| OH   | 4.4369643744260896e-14 | 4.4369652359845808e-14 | 4.4369654512473184e-14 |
| H2O  | 2.0271889532571192e-14 | 2.0271868906078652e-14 | 2.0271863756670452e-14 |
| HO2  | 6.4503375999817024e-15 | 6.4503373814707224e-15 | 6.4503373268427008e-15 |
| H2O2 | 5.5192160918074848e-19 | 5.5192127150924152e-19 | 5.5192118711561208e-19 |

Measured ratios, all species, all pairs:

| dt pair | factor | predicted if quadrature-per-step | **measured (Sounio)** |
|---|---|---|---|
| 4e-9 / 2e-9 | 2 | 1.414214 | **0.999999 – 1.000001** |
| 2e-9 / 1e-9 | 2 | 1.414214 | **0.999999 – 1.000000** |
| 4e-9 / 1e-9 | **4** | **2.000000** | **0.999999 – 1.000001** |

**The bands are non-trivial** (H2 and O2 carry ~1% of their value; the radicals
sit at 3e-14 to 6e-19), so this is invariance, not a degenerate zero.

This is the whole contrast, in one sweep and on one page:

| implementation | band under a factor 4 in dt |
|---|---|
| Python replica (per-step independent quadrature) | **1.99994 – 2.00841** |
| C++23 cross-check (same formula, independent code) | **1.99994 – 2.00841** |
| **Sounio native (coherent sensitivity propagation)** | **0.999999 – 1.000001** |

The Sounio module carries this as a declared, tested property —
`test_g30_epistemic_step_invariance` in `stdlib/chemistry/gri30_h2.sio`
asserts agreement between dt = 1e-8 and dt = 5e-9 to a 1e-2 relative
tolerance. The sweep above shows it actually holds to **~1e-6**, four orders
of magnitude tighter than the test asserts, and across a factor 4 rather than
a factor 2.

### 5.5 What this measures is the defect itself

This √dt behaviour is **not** a property of the chemistry — it is the
signature of the defect the preprint reports. The replica (and the C++
cross-check, which reproduces it faithfully) adds an **independent** quadrature
source `Σ_r (ν_ir·net_r·dt·u_r)²` at **every** time step, treating
successive steps as independent when they share the same rate parameters.
Persistent parameter uncertainty is not independent across steps, so
quadrature is invalid here, and the artefact is precisely that the band
acquires a spurious √dt dependence — a band that changes when you change the
step size is reporting the integrator, not the chemistry.

`benchmarks/chemistry/README.md` states the same conclusion from the other
direction: the Sounio native module propagates persistent rate-parameter
sensitivities coherently, and "a quadrature source added independently at
every time step would incorrectly make radical uncertainties scale with the
square root of `dt`." **Section 5.1 is the direct measurement of that
"would", and section 5.4 is the direct measurement of the "does not".**

This is the preprint's thesis, measured on both sides: **the implementation
(Sounio) is right and its oracle (the Python replica) is wrong.** A reviewer
who took the replica as ground truth and the √dt scaling as a physical result
would have drawn the opposite conclusion. PR #1758 is the language-level
remedy — it makes quadrature require a proof of d-separation, so that
composing uncertainty terms that are not independent stops being expressible
rather than merely being wrong.

---

## 6. Full-mechanism Cantera oracle (new)

There was no Cantera parity script for the full 53-species / 325-reaction
mechanism; only the H/O one existed, so the full-mechanism results had no
reproduction path. `benchmarks/chemistry/gri30_full_cantera_parity.py` is
new at this commit and closes that gap. It uses the corrected (TDY,
non-renormalising) initialisation from section 1.4.

```sh
python3 benchmarks/chemistry/gri30_full_cantera_parity.py
python3 benchmarks/chemistry/gri30_full_python_replica.py
```

```
cantera 3.2.0
FULLMECH species=53 reactions=325
INIT worst relative deviation from intended concentrations = 0.000000e+00
INIT pressure = 101325.576758 Pa
```

Isothermal, constant volume, T = 1500 K, 2% H2 / 1% O2 / 97% N2, additive
H seed 1e-11 mol/cm³. Replica: fixed-step RK4, dt = 2e-9 (dt = 1e-8 is outside
the RK4 stability limit for this mechanism because of `NNH <=> N2 + H`).

### Checkpoint t = 4e-6 s (n = 2000 steps)

| species | replica (RK4 dt=2e-9) | Cantera 3.2 (CVODE) | rel dev |
|---|---|---|---|
| H2   | 1.62485959583121780e-07 | 1.62485959581675630e-07 | 8.900e-12 |
| H    | 1.05531109518200342e-11 | 1.05531114324938578e-11 | 4.555e-08 |
| O    | 1.24300232843104601e-12 | 1.24300281422727476e-12 | 3.908e-07 |
| O2   | 8.12421326765154836e-08 | 8.12421326755489925e-08 | 1.190e-11 |
| OH   | 1.09368254389978098e-12 | 1.09368302264047293e-12 | 4.377e-07 |
| H2O  | 1.83313117067441303e-12 | 1.83313213645402558e-12 | 5.268e-07 |
| HO2  | 1.20494092398029042e-13 | 1.20494094065092846e-13 | 1.384e-08 |
| H2O2 | 2.04379663825074701e-16 | 2.04379695675438026e-16 | 1.558e-07 |
| N2   | 7.88070081364192654e-06 | 7.88070081364194179e-06 | 1.935e-15 |
| NNH  | 1.57674597495852050e-17 | 1.57674604674652226e-17 | 4.553e-08 |

**worst = 5.268e-07**

### Checkpoint t = 2e-5 s (n = 10000 steps)

| species | replica (RK4 dt=2e-9) | Cantera 3.2 (CVODE) | rel dev |
|---|---|---|---|
| H2   | 1.62447037642099308e-07 | 1.62447037611745645e-07 | 1.869e-10 |
| H    | 3.26593676725774637e-11 | 3.26593842363430721e-11 | 5.072e-07 |
| O    | 4.72236112196927923e-12 | 4.72236467598050302e-12 | 7.526e-07 |
| O2   | 8.12251664034333963e-08 | 8.12251663896467307e-08 | 1.697e-10 |
| OH   | 4.02166906116787291e-12 | 4.02167237572375348e-12 | 8.242e-07 |
| H2O  | 2.78622581424163930e-11 | 2.78622784602024649e-11 | 7.292e-07 |
| HO2  | 8.65149682729337221e-13 | 8.65149874546233508e-13 | 2.217e-07 |
| H2O2 | 3.55244440868556859e-15 | 3.55244557601469086e-15 | 3.286e-07 |
| N2   | 7.88070081354257466e-06 | 7.88070081354255264e-06 | 2.795e-15 |
| NNH  | 4.87947004733045173e-17 | 4.87947252189819014e-17 | 5.071e-07 |

**worst = 8.242e-07**

NNH is reported because it is the Δn = +1 species of section 3.2. It tracks
the oracle to 4.6e-08 / 5.1e-07, confirming its quasi-equilibrium population
is reproduced.

### Sounio native column (via the reconstructed `full_probe.sio`)

```sh
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/full_probe.sio
```

| species | Sounio native (RK4 dt=2e-9) | vs replica | vs Cantera | ULP(S,replica) |
|---|---|---|---|---|
| **t = 4e-6 s** ||||
| H2   | 1.62485959583121701e-07 | 4.887e-16 | 8.900e-12 | 3 |
| H    | 1.05531109518200294e-11 | 4.593e-16 | 4.555e-08 | 3 |
| O    | 1.24300232843104399e-12 | 1.625e-15 | 3.908e-07 | 10 |
| O2   | 8.12421326765154042e-08 | 9.774e-16 | 1.190e-11 | 6 |
| OH   | 1.09368254389977896e-12 | 1.846e-15 | 4.377e-07 | 10 |
| H2O  | 1.83313117067441303e-12 | **0.000e+00** | 5.268e-07 | 0 |
| HO2  | 1.20494092398028789e-13 | 2.095e-15 | 1.384e-08 | 10 |
| H2O2 | 2.04379663825074306e-16 | 1.930e-15 | 1.558e-07 | 16 |
| N2   | 7.88070081364192654e-06 | **0.000e+00** | 1.935e-15 | 0 |
| NNH  | 1.57674597495852143e-17 | 5.863e-16 | 4.553e-08 | 3 |
| **worst** || **2.095e-15** | **5.268e-07** | **16** |
| **t = 2e-5 s** ||||
| H2   | 1.62447037642099308e-07 | **0.000e+00** | 1.869e-10 | 0 |
| H    | 3.26593676725774702e-11 | 1.979e-16 | 5.072e-07 | 1 |
| O    | 4.72236112196927115e-12 | 1.711e-15 | 7.526e-07 | 10 |
| O2   | 8.12251664034333963e-08 | **0.000e+00** | 1.697e-10 | 0 |
| OH   | 4.02166906116786806e-12 | 1.205e-15 | 8.242e-07 | 6 |
| H2O  | 2.78622581424163865e-11 | 2.319e-16 | 7.292e-07 | 2 |
| HO2  | 8.65149682729336615e-13 | 7.003e-16 | 2.217e-07 | 6 |
| H2O2 | 3.55244440868556386e-15 | 1.332e-15 | 3.286e-07 | 12 |
| N2   | 7.88070081354257466e-06 | **0.000e+00** | 2.795e-15 | 0 |
| NNH  | 4.87947004733045296e-17 | 2.526e-16 | 5.071e-07 | 2 |
| **worst** || **1.711e-15** | **8.242e-07** | **12** |

The full mechanism reproduces the H/O finding of section 1.3 at 53 species and
325 reactions: **Sounio agrees with the replica at 0–16 ULP** (four species are
bit-identical), and the 1e-7-scale figure is the RK4-vs-CVODE integrator
difference, not a cross-language one. NNH — the Δn = +1 quasi-equilibrium
species of section 3.2 — tracks both to 2.5e-16 / 5.1e-07.

### 6.1 Coherent band vs the independent Cantera central-difference referee

```sh
python3 benchmarks/chemistry/gri30_full_cantera_uq_reference.py --jobs 4
```

```
Cantera 3.2.0 gri30.yaml species=53 reactions=325 wall=7.908s
T=1500K t=4.0e-07s energy=off rtol=1e-12 atol=1e-22 delta=1e-03
```

| species | referee y (mol/cm³) | referee u (1-σ) |
|---|---|---|
| H2   | 1.624886332731066e-07 | 1.624886621846326e-09 |
| H    | 9.827370721628137e-12 | 1.728153747753403e-14 |
| O    | 1.963474015362450e-13 | 1.962035152319852e-14 |
| O2   | 8.124411817497948e-08 | 8.124413369241644e-10 |
| OH   | 1.921444969736858e-13 | 1.935251605949693e-14 |
| H2O  | 2.746525927522075e-14 | 3.727791931739965e-15 |
| HO2  | 1.212910166879070e-14 | 2.605446150229169e-15 |
| H2O2 | 2.482543463532573e-18 | 9.167378486142805e-19 |
| NNH  | 1.468598577480058e-17 | 2.625142477347715e-20 |

**The referee is reproducible.** Its H2 value, `1.624886332731066e-07`,
reproduces the figure the README publishes for this checkpoint **to all 16
digits**.

The Sounio side now exists too, via `examples/chemistry/full_probe.sio`
(1% standard uncertainty on initial H2 and O2, 200 steps at dt = 2e-9):

| species | Sounio σ (coherent) | referee σ (central difference) | σ rel dev | y rel dev |
|---|---|---|---|---|
| HO2  | 2.60544574413659306e-15 | 2.60544615022916889e-15 | **1.559e-07** | 7.237e-09 |
| H2   | 1.62488635103136998e-09 | 1.62488662184632606e-09 | 1.667e-07 | 1.360e-13 |
| O2   | 8.12441201519362670e-10 | 8.12441336924164370e-10 | 1.667e-07 | 1.157e-12 |
| H2O2 | 9.16738051753079131e-19 | 9.16737848614280495e-19 | 2.216e-07 | 1.482e-07 |
| H2O  | 3.72779413164159965e-15 | 3.72779193173996469e-15 | 5.901e-07 | 5.434e-07 |
| O    | 1.96203382476252415e-14 | 1.96203515231985209e-14 | 6.766e-07 | 4.427e-07 |
| OH   | 1.93525021825764087e-14 | 1.93525160594969299e-14 | **7.171e-07** | 4.492e-07 |
| H    | 1.72815224012817700e-14 | 1.72815374775340311e-14 | **8.724e-07** | 7.312e-09 |

**This README claim reproduces exactly — every figure in it.** The largest
relative σ deviation is **8.724e-07 on H**, and the other seven span
**1.559e-07 to 7.171e-07**, matching the published sentence term for term.
The published Sounio H2 value `1.624886332731288e-7` is measured here as
`1.624886332731287e-07` — a one-digit difference in the last place, which is
the probe's integer-mantissa truncation, not a disagreement.

Worth stating plainly, because section 1 corrected two other README claims:
**the full-mechanism UQ section of the README is not stale.** It is exactly
right, to every digit it publishes. The corrections in section 1 apply to the
H/O checkpoint prose, not to this.

---

## 6.2 The measured law is a theorem — `formal/lean4/SounioIndepComposition.lean`

```sh
cd formal/lean4 && lake build SounioIndepComposition
```

Section 5 measures a band that scales as √dt and an underestimation that grows
as √(T/dt). That is not a fitted exponent — it is derivable, and it is now
machine-checked.

The per-step parameter term `(ν·net·dt·u_r)²` carries the **same** rate
parameter at every step. Successive steps are therefore correlated with ρ = +1,
not 0. For N contributions of equal uncertainty `u`:

| | variance | uncertainty |
|---|---|---|
| truth (ρ = +1, uncertainties add) | (N·u)² | N·u |
| quadrature (as if independent) | N·u² | √N·u |
| **ratio** | **N** | **√N** |

With N = T/dt that is **√(T/dt)** — precisely the law measured in §5.3, and
read along dt instead of T it is the **√dt** dependence of §5.1.

`quadrature_understates_correlated_sum` states exactly this, and 14 further
theorems establish the rest of #1758's contract:

| theorem | content |
|---|---|
| `quadrature_iff_zero_covariance` | quadrature agrees with JCGM eq. (13) **iff** cov = 0 — it is the independence law, not an approximation |
| `quadrature_understates_of_positive_covariance` | cov > 0 → quadrature is **strictly tighter than the truth** (unsound; `SEMANTICS.md` Invariant 2) |
| `quadrature_sound_iff_nonpositive_covariance` | quadrature is an upper bound **exactly** when cov ≤ 0 |
| `additive_sound` | the additive default is sound for **every** admissible ρ |
| `additive_tight_at_unit_correlation` | and is tight at ρ = +1, so it is the least such bound |
| `quadrature_understates_correlated_sum` | **the √N law above** |
| `accumulation_agrees_at_one_step` | at N = 1 the two agree — why the defect is invisible in a single composition |
| `collider_opened_by_conditioning` | conditioning on a collider **opens** the path (Berkson 1946) |
| `conditioning_not_monotone` | ∃ a junction where conditioning turns a blocked path active — so a reachability check with a blocklist is unsound |

**Method.** Mathlib-free, core Lean 4 only, matching the discipline of
`SounioMeasConf.lean`. Every claim is stated on **variances**, not standard
uncertainties: since √ is monotone on the non-negatives, comparing variances
*is* comparing uncertainties, and the square root never has to be constructed.
In that form every statement is polynomial, and the linear ones close under
`omega`.

**Verification.** 15 theorems, **zero `sorry`**. `#print axioms` reports only
`propext` and `Quot.sound` — the standard Lean axioms — on the arithmetic
theorems, and the d-separation theorems **depend on no axioms at all**
(`collider_opened_by_conditioning`, `collider_inverts_the_others`,
`conditioning_not_monotone` are closed by `rfl`). Built under
`leanprover/lean4:v4.33.0`, the toolchain `formal/lean4/lean-toolchain` pins.

**What this does not establish.** The theorems say what follows *given* a
correlation structure; they do not certify that any particular program's
declared graph matches the world. #1758 is explicit on the same point — the
system makes the premise explicit and auditable, not true. Nor does the Lean
development verify the floating-point numerics of the integrator: it is exact
arithmetic over `Int`, and the numerical agreement is the business of §1–§6.

---

## 7. Reproduction

### 7.1 Commands, in order

```sh
git rev-parse HEAD           # 98aa8e4d5151bbc61815bf910b6c31c3d0789f5f
pip install 'cantera==3.2.0' numpy

export SOUNIO_STDLIB_PATH=$(pwd)/stdlib SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run examples/chemistry/h2_ignition_uq_demo.sio        # ~90 s
python3 benchmarks/chemistry/gri30_h2_python_replica.py          # ~4 s (det.)
python3 benchmarks/chemistry/gri30_h2_cantera_parity.py          # ~1 s
python3 benchmarks/chemistry/gri30_full_python_replica.py        # ~20 s
python3 benchmarks/chemistry/gri30_full_cantera_parity.py        # ~1 s
python3 benchmarks/chemistry/gri30_full_cantera_uq_reference.py --jobs 4   # ~8 s

cd benchmarks/chemistry/cpp
g++ -std=c++23 -O2 -o band_crosscheck gri30_h2_band_crosscheck.cpp
./band_crosscheck ../gri30_h2_mechanism.json                     # ~6 min

cd ../../../formal/lean4
lake build SounioIndepComposition                                # ~1 s
```

### 7.2 Oracle-verification probes

`benchmarks/chemistry/rep_1atm.py` (section 3) and
`benchmarks/chemistry/rep_prodfix.py` (section 2) are reconstructions —
see section 4 and their own headers.

---

## 8. Claims still lacking a reproduction path

| claim | status |
|---|---|
| "H2O2 agrees to 5.9e-3 relative" (README, headline) | **contradicted** — measured 1.891e-06 |
| "majors 0.2–2%, radicals ~3%, H2O2 ~16%" | **no provenance** — absent from every version of every file in this repo; no pairing reproduces it |
| "factor 4 in dt, ratio √2" (log) | **contradicted** — factor 4 gives 2.0; √2 is the factor-2 ratio |
| "√(T/dt) is exact at T = 1e-6, 1e-5, 1e-4" | **contradicted** — off by 59×–242× over the second decade |
| Sounio native **full-mechanism** trajectory at 4e-6 / 2e-5 s | **closed** — measured this session via the reconstructed `full_probe.sio`; section 6 |
| Full-mechanism coherent band vs "Cantera central-difference referee", largest σ deviation 8.724e-07 at t = 4e-7 s (README) | **closed — reproduced exactly**, both sides, every figure: largest σ dev 8.724e-07 (H), others 1.559e-07 … 7.171e-07. Section 6.1. |
| Ignition-delay table 1400–1800 K, Sounio column | reproduced (Cantera column re-measured: 169.66 / 126.34 / 98.29 / 79.00 / 65.08 µs) |
| Preprint Results 2–5 (probe-based) | **no original artefacts** — see section 4. Four reconstructed `.sio` probes and two reconstructed `.py` probes now exist and all check and run clean, but they reproduce the *protocols*, not the originals |
| Sounio native band vs dt (step invariance) | **closed** — measured this session, `band_sweep.sio`, section 5.4 |
| Sounio vs replica at ULP resolution | **closed** — measured this session, `h2_precision_probe.sio`, section 1.3 |
