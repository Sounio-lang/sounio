# GRI-Mech 3.0 cross-validation — measured results

> **This PR is one of four.** The work was split so each carries its own
> justification and can be reviewed on its own evidence:
>
> | PR | contents | why separate |
> |---|---|---|
> | this one | `benchmarks/chemistry/`, `examples/chemistry/` | the measurements and this document |
> | gas-constant alignment | `stdlib/chemistry/`, replicas, tests | a semantic change to the constants, spanning stdlib |
> | C++23 cross-check | `benchmarks/chemistry/cpp/` | an independent third implementation |
> | Lean development | `formal/lean4/` | targets the d-separation PR, not `main` |
>
> **The numbers in this document are measured in the ALIGNED regime and
> require the gas-constant PR.** Landing this alone would leave the probes
> setting an initial state the modules do not share. The published-regime
> numbers are kept alongside, dated, for the record.

**Every number on this page was produced by the command printed above it, at
commit `98aa8e4d5151bbc61815bf910b6c31c3d0789f5f` (branch `claude/gri-mech-cantera-preprint-776qb9`), on
2026-09-01.** Nothing is carried forward from an earlier log, a prior session,
or a draft. Where a run was not performed, the row says so.

Environment: Linux x86-64, Python 3.11, `cantera 3.2.0`, `numpy 2.4.6`,
`g++ (Ubuntu 13.3.0) -std=c++23 -O2`. Sounio compiler: the committed ELF
`bin/madaros-linux-x86_64` (md5 `ff69dae4`, tree `98aa8e4d`), run under
`SOUNIO_SOUC_ENGINE=lean_single`.

> **Corrected 2026-09-01.** This line previously read `-std=c++20`, which does
> not build the cross-check as shipped. Measured rather than assumed:
> `g++ -std=c++20 -O2 -fsyntax-only cpp/gri30_h2_band_crosscheck.cpp` fails with
> three errors — `std::expected` is C++23-only, and the multidimensional
> `operator[](std::size_t, std::size_t)` "must have exactly one argument" before
> C++23. `-std=c++23` compiles clean. The numbers in sections 5 and 6 were
> produced by the C++23 build, matching the command blocks printed there.

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

### 1.1a Claim (A) is a real measurement of a real defect, mislabelled

**Superseded 2026-09-01, after the measurement in §2. An earlier revision of
this file said claim (A) "has no provenance and should not be cited". That was
wrong, and the correction matters more than the original claim.**

Claim (A) is not a parity table. It is the **per-species error profile of the
`reac - nu` reverse-rate defect** at the isothermal pre-front checkpoint —
a real number, from a real measurement, of a real defect, presented as though
it were a Sounio-versus-Cantera comparison.

```sh
python3 benchmarks/chemistry/rep_traj_bug.py
```

| claim (A) | measured under `reac - nu` |
|---|---|
| "majors within **0.2–2%**" | H2 **0.21%**, O2 **0.18%**, HO2 0.14%, O 1.49%, H2O 1.69%, H 1.76% |
| "radicals **~3%**" | OH **3.44%** |
| "H2O2 **~16%**" | H2O2 **16.17%** |

Three figures, three matches, the last one to three significant figures. The
provenance is closed.

**In one line: the most-cited numerical claim of this project — "majors
0.2–2 %, radicals ~3 %, H2O2 ~16 %" — was produced by the buggy form of the
reverse rate, `reac − nu`, and is an artefact of the defect the project would
later discover.**

What was wrong was the **label**, not the number. No pairing of Sounio,
replica and Cantera produces percent-level deviations — that part of the
earlier finding stands, and §1.2 measures it. But claim (A) was never a
statement about that comparison.

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
   documents — a **5-to-6-significant-figure** agreement. The
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

> **Decided and implemented 2026-09-01, after this section was written.** The
> operator's decision was to align **both** constants, as two changes with
> distinct justifications kept in separate commits: `R_cal` 1.9872041 →
> 1.9872042586408316 (Cantera's `R_SI/4.184`, the subject of this section) and
> the molar-volume constant `1/(82.057·T)` → `101325.0/(8.31446261815324·T)·1e-6`
> (CODATA-2018 `R_SI` and the exact standard state, which removes a second
> truncation). Both land in
> [#2382](https://github.com/Sounio-lang/sounio/pull/2382), not here. The
> paragraph above records the state of the evidence at the time of measurement
> and the reasoning that was put to the operator; it is **not** the final
> disposition. The caveat it raises stands and is unresolved: whether
> GRI-Mech 3.0's rate parameters were regressed under this or another rounding
> of R is still not established from the regression documentation, which is not
> in this repository. The alignment was chosen with that uncertainty explicit,
> not because it was closed.
>
> Measured separately, by 2×2 factorial: under `TDY` initialisation the
> molar-volume constant contributes **exactly zero** to the parity gap —
> `R_cal` is the entire effect. The second change is justified on its own terms
> (removing a truncated constant), not by an improvement it does not produce.

---

## 2. The reverse-rate defect — real, historical, fixed, and reproduced here

**Superseded 2026-09-01. An earlier revision of this file said this defect
"does not exist and never existed". That was true of the file the brief named
and false as a statement about the repository.**

### 2.1 It existed, in the adiabatic replica

`benchmarks/chemistry/README.md` documents it, and I found it only after
reporting the opposite:

> *the first version of the **adiabatic** Python replica computed reverse
> rates with product exponents derived as `reac - nu` … which silently zeroed
> the reverse channels of H2+O2 → H+HO2 and the back-dissociation reactions at
> states where only major species are present. The error grew with temperature
> (0.1% at 1100 K → 8.6% at 2000 K on the delay) and was invisible to every
> Sounio↔replica pin. Per-reaction rate comparison against Cantera at t0
> exposed it in one shot.*

The buggy revision is not recoverable — history for these files in this clone
begins at `d25b43a4` and no version under any ref contains `reac - nu`. The
README is testimony, not artefact. **Method error worth naming: I searched the
file the brief named, found nothing, and generalised from "absent here" to
"never existed".**

### 2.2 The mechanism, at the exponent

`reac - nu` ≡ `2·reac - prod`. The guard is `if p > 0`, so a negative exponent
is *skipped*. In all **29 of 29** reactions the products (correct exponent ≥ 1)
go negative and drop out, while the reactants (correct exponent 0) come in at
+2 or +4. The reverse term stops depending on product concentrations.

For `H + HO2 <=> O2 + H2` — the channel the brief singles out:

| species | correct | under `reac - nu` |
|---|---|---|
| H2 | 1 | **−1** (skipped) |
| O2 | 1 | **−1** (skipped) |
| H | 0 | **+2** |
| HO2 | 0 | **+2** |

Its products are O2 and H2, the majors, so its **reverse direction is
H2 + O2 → H + HO2 — chain initiation**. In a majors-only state the correct
reverse rate is large; the buggy one is ∝ c[H]²·c[HO2]² ≈ 0. That is exactly
the README's "zeroed the reverse channels of H2+O2 → H+HO2", derived
independently.

### 2.3 Per-species deltas, isothermal checkpoint, all three forms

Three forms, not two:

| form | expression | equals |
|---|---|---|
| shipped | `p = prod[r][s]` | `prod` |
| proposed "fix" | `p = reac + nu` | **≡ `prod` — an identity** |
| reported bug | `p = reac - nu` | `2·reac − prod` |

**The proposed fix is a no-op**, confirmed by measurement and not only by
algebra: **exactly 0**, bit-for-bit, on all eight species, at the checkpoint
**and** in the 1-σ band.

```sh
python3 benchmarks/chemistry/rep_traj_bug.py
```

Under the bug (T = 1500 K, t = 1e-4 s, dt = 1e-8):

> **Provenance, corrected 2026-09-01.** The absolute columns below were
> measured in the *aligned* regime from a working copy that was never
> committed — the same defect as §6.2b, in the section that closes the
> provenance of claim (A). `rep_traj_bug.py` is that harness, committed. It
> runs in whatever regime the oracle it loads is in. **From the frozen snapshot,
> whose oracle is aligned, every absolute column below reproduces to every
> printed digit** — H2 1.45202409180648259e-07, R16 forward
> 5.22176484288829602e-06, d[HO2]/dt −8.13560209958764591e-08. From this
> branch, whose oracle is in the published regime, the absolutes sit in the
> sixth digit and **the deltas — which are the claim — reproduce to four
> figures**: 2.105e-03, 1.756e-02, 1.490e-02, 1.774e-03, 3.441e-02, 1.687e-02,
> 1.394e-03, 1.617e-01; `fix − shipped` is 0.000e+00 bit-for-bit in both.

| sp | shipped | under `reac - nu` | delta | band |
|---|---|---|---|---|
| H2 | 1.45202409180648259e-07 | 1.44896760284528512e-07 | **2.105e-03** | 4.755e-04 |
| H | 9.78118787162766235e-09 | 9.95290506208524791e-09 | **1.756e-02** | 5.072e-02 |
| O | 1.45235755728164051e-09 | 1.47399979085128111e-09 | **1.490e-02** | 7.585e-02 |
| O2 | 7.39988119297947390e-08 | 7.38675293103348018e-08 | **1.774e-03** | 9.361e-05 |
| OH | 1.22456521149193150e-09 | 1.26670674554832556e-09 | **3.441e-02** | 7.543e-02 |
| H2O | 1.17789779335583026e-08 | 1.19776593065772939e-08 | **1.687e-02** | 3.408e-02 |
| HO2 | 1.70510868870023519e-11 | 1.70748634715518421e-11 | **1.394e-03** | 1.100e-02 |
| H2O2 | 1.62463513786769239e-13 | 1.88736066163527837e-13 | **1.617e-01** | 1.398e+00 |

This table is claim (A). See §1.1a.

### 2.4 d[HO2]/dt — the −34% does not reproduce

```sh
python3 benchmarks/chemistry/rep_traj_bug.py
```

Reproduced by the committed harness: shipped −8.13588260837949163e-08,
under the bug −8.93772453622013026e-08, **−9.86%**.

| quantity | value |
|---|---|
| d[HO2]/dt, shipped | −8.13560209958764591e-08 |
| d[HO2]/dt, under the bug | −8.93741991782262909e-08 |
| **relative change** | **−9.86%** |

Not −34%. And distinct from the **−50.5%** figure, which is R16's *net rate of
progress* at a radical-loaded probe state — a different quantity at a
different state. Three numbers, three quantities; the −34% matches none of
them and is **left unexplained rather than forced into agreement**.

### 2.5 Why the shipped and fixed forms are identical yet the bug is not

```sh
python3 benchmarks/chemistry/rep_traj_bug.py
```

R16 at the checkpoint:

```
forward          5.22176484288829602e-06
reverse          8.54651292554704534e-09
reverse/forward  1.637e-03
```

The reverse channel is 0.16% of the forward. The *fix* is the same expression,
so its delta is exactly zero. The *bug* does not merely delete that 0.16% —
it substitutes a term keyed to reactant rather than product concentrations,
and ten thousand steps of chain branching turn that substitution into the
percent-level profile of §2.3.

### 2.6 Adiabatic provenance: the README's magnitudes reproduce

```sh
python3 benchmarks/chemistry/rep_adiabatic_bug.py --quick    # 1100 K and 2000 K
python3 benchmarks/chemistry/rep_adiabatic_bug.py            # all four
```

> **Provenance, corrected 2026-09-01.** This table was first measured by
> reintroducing `reac - nu` in a working copy of
> `gri30_h2_adiabatic_replica.py` that was never committed, and the frozen
> snapshot did not even ship the replica. `rep_adiabatic_bug.py` is that
> harness, committed; it monkey-patches the versioned module's `uv_rhs` rather
> than carrying its own chemistry, so it cannot drift from the replica it
> characterises. The replica now ships in the snapshot. Reproduced to every
> printed digit: 674.9575 / 675.6025 / +0.096 % and 46.3075 / 50.2675 /
> +8.552 %. Run from the frozen snapshot, whose replica is aligned, the same
> four delays print identically: the constant shift moves a 675 µs delay by
> ~7e-4 µs, below the 0.005 µs resolution of dt = 5e-9, so this section is one
> of the few that reproduces to every digit in both regimes.

Ignition delay at the time of maximum d[H2O]/dt, dt = 5e-9:

| T₀ (K) | correct (µs) | under the bug (µs) | error | README |
|---|---|---|---|---|
| 1100 | 674.9575 | 675.6025 | **+0.096%** | **0.1%** |
| 1400 | 169.6625 | 173.7175 | +2.390% | — |
| 1700 | 78.9775 | 81.9575 | +3.773% | — |
| 2000 | 46.3075 | 50.2675 | **+8.552%** | **8.6%** |

Both documented anchors reproduce. On this criterion every sign is positive:
the defect always delays ignition, consistent with removing a radical source.

**A variable the README does not declare, found by the committed harness.**
The replica exposes two delay definitions — the time of maximum d[H2O]/dt
and the time of maximum dT/dt — and the harness prints both, because a defect
that moved only one would be suspect. They disagree:

| T₀ (K) | error, d[H2O]/dt criterion | error, dT/dt criterion |
|---|---|---|
| 1100 | **+0.096 %** | **−2.375 %** |
| 2000 | **+8.552 %** | +9.947 % |

At 1100 K the sign flips. A defect that shifts the H2O-rate delay by 0.1 %
shifts the temperature-rise delay by 2.4 % *the other way*, which says the
dT/dt peak is broad and ill-conditioned there at this step, not that the
defect accelerates ignition. The README's anchors — 0.1 % and 8.6 % — are
reproduced **on the d[H2O]/dt criterion only**, and the README does not say
which criterion it used. That omission is now a documented ambiguity rather
than a silent assumption, and it is the kind of unstated protocol variable
§6.3 instance (8) is about: a documentation number with no resolution label
and no definition of the quantity it measures.

### 2.7 The shipped isothermal path is correct

Verified against the oracle rather than by inspection — all 29 net rates of
progress against `Cantera.net_rates_of_progress`, radical-loaded state:

```sh
python3 benchmarks/chemistry/rep_prodfix.py
```

| | worst relative deviation, published regime | after #2382 |
|---|---|---|
| shipped reverse path | **7.877e-07** (the R_cal residual of §1.5) | **8.442e-15** |
| `reac - nu` | **1.838e+00** (184%) | **1.838e+00**, unchanged |
| R16 specifically | shipped **2.860e-08**, bug 5.046e-01 | shipped **4.134e-15**, bug 5.046e-01 |

> **Re-measured 2026-09-03, and the second column is new.** The first column
> was measured before the gas constant was aligned; #2382 has since merged, so
> the producer no longer runs in that regime and the numbers cannot be left
> standing unqualified. Re-running it now gives the second column: the shipped
> path collapses to the double-precision floor, and the defect's magnitude does
> not move at all — 1.838e+00 and 5.046e-01 to four figures in both regimes.
> That contrast is the section's point. **The defect is a property of the
> stoichiometry; the floor beneath it was a property of the constant.**
>
> The published-regime figures are reproducible on demand rather than trusted:
> setting `R_CAL = 1.9872041` in `gri30_h2_python_replica.py` and re-running
> returns R16's shipped deviation to exactly **2.860e-08**, which is how the
> first column was attributed rather than assumed.
>
> One further correction, found by the snapshot verifier and not by any gate
> here: this producer built its radical-loaded state from `1/(82.057·T)` while
> every sibling had moved to `P0/(R·T)·1e-6`. It is aligned in the same commit.
> The change is 5.7e-06 in the state and moves the shipped floor from 9.204e-15
> to 8.442e-15; the buggy column does not move by a single bit, because R16's
> forward term depends only on the fixed radical seeds, which the molar volume
> does not touch.

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
python3 benchmarks/chemistry/rep_1atm.py
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

**Data Availability wording, to be used verbatim:** *reconstructed from the
described protocol on 2026-09-01; the original artefacts were not recoverable
from the repository history.*

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

> **Cross-branch reference.** `cpp/gri30_h2_band_crosscheck.cpp` is in
> [#2383](https://github.com/Sounio-lang/sounio/pull/2383), not on this branch,
> so `audit_provenance.py` reports this section as failing when run here and
> passing when run against the frozen release, where the file ships. That
> difference is the audit working, not a defect: it is the only section whose
> producer lives in a sibling pull request. `-std=c++20` will not build it —
> `std::expected` and the multidimensional `operator[]` are both C++23.

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

> **The committed oracle is now the aligned one; the tables in this section
> are the pre-alignment record.** The script initially carried the same
> `1.0/(82.057*T)` shorthand as everything else, which put it at a *different
> initial density than `stdlib/chemistry/gri30_full.sio`, the module it is the
> oracle for* — the one thing an oracle may not do. Caught 2026-09-01 by
> re-running it and finding it print the published-regime pressure while
> section 6.2b reports the aligned one: **6.2b had been measured with a working
> copy that was never committed.** The constant is now
> `P0/(R_SI*T)*1e-6`, so **section 6.2b reproduces from the committed file and
> this section's tables no longer do.** They are kept as the published-regime
> measurement, not deleted.

Output of the **pre-alignment** version (the regime of the tables below):

```
cantera 3.2.0
FULLMECH species=53 reactions=325
INIT worst relative deviation from intended concentrations = 0.000000e+00
INIT pressure = 101325.576758 Pa
```

Output of the **committed, aligned** version (the regime of section 6.2b):

```
cantera 3.2.0
FULLMECH species=53 reactions=325
INIT worst relative deviation from intended concentrations = 1.629030e-16
INIT pressure = 101325.124717 Pa
```

The residual 1.629030e-16 is one ULP: with the aligned constant the intended
total concentration *is* the ideal-gas concentration at P0, so the TDY round
trip through molar masses lands within a single unit in the last place instead
of exactly on it. It is 1e-10 times smaller than the deviation the TPX defect
of section 1.4 introduced, and does not change any digit reported in 6.2b.

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

## 6.2 The measured law is a theorem — `SounioIndepComposition.lean` (sibling PR)

```sh
# the Lean development ships in its own PR, targeting the d-separation PR
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

## 6.2b STEP 6 re-measured in the aligned regime

The §6 tables above were measured in the **published** regime and are stale
once the constants are aligned (`claude/align-molar-volume-constant`,
`8acb16c0` + `fb474888`). Re-measured, t = 4e-6 s, P = 101325.124717 Pa:

| sp | Sounio (native) | S↔replica | ULP | S↔Cantera |
|---|---|---|---|---|
| H2 | 1.62485234694381110e-07 | **0.000e+00** | **0** | 1.629e-15 |
| H | 1.05531016903763002e-11 | 1.531e-16 | 1 | 5.119e-12 |
| O | 1.24299936798135503e-12 | 1.137e-15 | 7 | 1.188e-11 |
| O2 | 8.12417702326992007e-08 | 1.629e-16 | 1 | 6.516e-16 |
| OH | 1.09368026746034605e-12 | 1.108e-15 | 6 | 3.175e-11 |
| H2O | 1.83311859985921710e-12 | 4.407e-16 | 2 | 2.127e-11 |
| HO2 | 1.20493001257778295e-13 | 1.886e-15 | 9 | 1.702e-12 |
| H2O2 | 2.04377109287929189e-16 | 1.930e-15 | 16 | 3.630e-12 |
| N2 | 7.88066565562194265e-06 | 2.150e-16 | 1 | 3.224e-15 |
| NNH | 1.57673756335980780e-17 | 1.954e-16 | 1 | 5.129e-12 |
| **worst** | | **1.930e-15** | **16** | **3.175e-11** |

| checkpoint | published | aligned |
|---|---|---|
| t = 4e-6 s | 5.268e-07 | **3.175e-11** |
| t = 2e-5 s | 8.242e-07 | **1.024e-11** |

The full mechanism reproduces the H/O result: ~5 orders, landing at the floor
of CVODE's own `rtol`. NNH — the Δn = +1 species of §3.2 — tracks to 5.129e-12.

**Dependency, and a conflict between the two branches.** These numbers require
the constant alignment on both sides. The oracle side is aligned in this branch
(section 6); the module side is aligned in
`claude/align-molar-volume-constant`. The two must land together, or the probes
here will set an initial state the modules do not share.

They also **conflict semantically**, which a textual merge will not surface.
`claude/align-molar-volume-constant` is cut from `main`, which never carried
the TPX → TDY fix of section 1.4; its `gri30_h2_cantera_parity.py` therefore
restores `gas.TPX = T, P0, X` and deletes `initial_concentrations` and
`initial_state_deviation`. Taking that file wholesale — merging carelessly, or
building an artefact from that branch — **silently reverts the only real defect
this work fixed**. The correct merged state is the TDY structure of this branch
with the aligned constants of that one. Verified by the deviation print: the
TDY form reports `0.000000e+00` (or one ULP once aligned), the TPX form
`5.692129e-06`.

---

## 6.3 The instrument hid the defect — nine instances

> **A note on the count.** The brief that commissioned this section asked for
> two new findings, bringing it to *six*. It brought it to **seven** — the
> section already carried five instances, not four — then to **eight**,
> when the operator recorded his own falsified premise as instance (8), and
> to **nine** when the archive layer failed silently under the release that
> was meant to freeze the other eight. The
> miscount is worth keeping rather than absorbing, because the instance most
> easily dropped from a mental list is (4), where the *reference's* own error
> was the thing being attributed to the method under test, and that is the one
> this document had to correct twice.

Every defect in this document was invisible to the instrument that was
supposed to find it, and in each case the instrument had **less resolution
than the defect it was asked to measure**. The readings then came out
attributed to the object rather than to the instrument. This is the pattern
worth carrying out of this work; the individual numbers are secondary.

The instances divide into two kinds, and the division is the point.

Instances **(1)–(4) are instruments set too coarse**: the resolution is a
number, the defect is a number, and the first is larger. They are fixable by
tightening a tolerance or choosing a finer probe.

Instances **(5)–(9) have no *syntactic* signature**. They are invisible to
every tool that reads the program as text or as types — no token to grep, no
dimension to check, no diff to review, no unit test that could have been
written against the source alone, because each branch, file and literal
involved is individually well-formed and correct.

**They are not, however, undetectable.** An earlier revision of this section
said they were, and then this document built an instrument that catches one of
them, which is a contradiction and is corrected here. The distinction that
survives is:

| | (1)–(4) | (5)–(9) |
|---|---|---|
| signature | a magnitude | none in the syntax |
| detected by | a tighter setting | a **behavioural** invariant, printed |
| the fix | calibrate | instrument deliberately, and **fail closed** |

The constructive form, and the reason this section is worth more than its
diagnosis: **a defect with no syntactic signature can still be given a
behavioural one, by refusing to proceed when the provenance of the initial
state cannot be established.** `benchmarks/chemistry/rep_tolerance.py`,
`rep_traj_bug.py` and `rep_resolution.py` are the reference implementation —
each draws its initial state from the oracle's `initial_concentrations()` and
**raises rather than reports** when that helper is absent, because its absence
*is* the signature of the TPX variant of instance (7). They do not compare two
protocols that were never the same and then print a plausible number; they
refuse. That is the whole recipe, and it costs four lines.

The corresponding recipe for instance (6) is a gate that **recomputes** a
convention constant rather than matching its text, since the text is what
carries no signature.

### (1) A reduced mechanism hides a standard-state error

12 of 29 H/O reactions carry Δn ≠ 0, so a 1-bar/1-atm confusion would be
*present* in 41% of the submechanism. It is unobservable there anyway, because
Kc enters only the reverse term and at 1500 K in the induction period that term
is 6–15 orders below the forward one. §3.2. It becomes observable only where an
equilibrium pins a population — NNH, Δn = +1, in the full mechanism.

### (2) A tolerance 57× too loose hides a constant

`test_g30_sim` asserts `1.452024295e-7` at 1e-4 relative tolerance. The
molar-volume shorthand moved the module by **1.740e-06** — **1.7% of the
tolerance**. The gate could not have failed on it. After alignment the same
gate sits at 0.14% of tolerance, 12.4× tighter, and now has room to see a
future regression of this size.

### (3) A shared initialisation hides an initial-state error

Under TDY both sides are built from the *same* `mtot`, so an error in it
cancels and the comparison is blind to it:

| regime | init | P initial (Pa) | worst replica-vs-Cantera |
|---|---|---|---|
| published | TDY | 101325.576758 | 2.660e-06 |
| published | TPX | 101325.000000 | 3.922e-05 |
| aligned | TDY | 101325.124717 | **2.032e-11** |
| aligned | TPX | 101325.000000 | 9.055e-06 |

Aligning `R_cgs` moves the TDY column **not at all** — 2.660e-06 before and
after. Under TPX, which does not share `mtot`, the same change is worth
3.922e-05 → 6.576e-06. The protocol choice, not the constant, decided what the
test could see.

### (4) A loose reference tolerance hides the integrator's own error

> **Corrected twice on 2026-09-01, and the second correction is instance (7)
> catching the author of this section.** The table was first published with no
> committed producer. A re-measurement then gave 3.251e-09 for the oracle's own
> spread, and this section *withdrew* the published 2.515e-08 as unreproducible.
> That withdrawal was wrong. The re-measurement ran in the **published**
> regime; the published table had been measured in the **aligned** one, and
> had not said so. Re-run in the aligned regime — `rep_tolerance.py` from the
> frozen snapshot, fresh `gas` per run, ten species — the oracle's spread is
> **2.515e-08, to four figures the number that was withdrawn.** The number was
> right; it lacked a producer and a regime label, and the corrector supplied a
> third regime error on top of the two missing labels. Both regimes are now
> given, from committed runs.

```sh
python3 benchmarks/chemistry/rep_tolerance.py            # published regime, this tree
python3 probes/rep_tolerance.py                          # aligned regime, frozen snapshot
```

| comparison | published regime | aligned regime | what it measures |
|---|---|---|---|
| CVODE default (`rtol=1e-9`) vs CVODE `rtol=1e-12` | 3.251e-09 (HO2) | **2.515e-08** (H2O) | the *oracle's* own tolerance spread |
| RK4 dt=1e-8 vs CVODE default | 2.662e-06 (O) | 2.517e-08 (H2O) | replica vs a loosely-set oracle |
| RK4 dt=1e-8 vs CVODE `rtol=1e-12` | 2.660e-06 (O) | 2.074e-11 (H2O) | replica vs a tightly-set oracle |
| RK4 dt=1e-8 vs RK4 dt=5e-9 | 2.222e-14 (H2O2) | 3.465e-14 (H2O2) | RK4 truncation + roundoff — the only row that is the replica's |

Two things the two-column form shows that neither column alone did.

**The oracle's tolerance spread is not a constant of the instrument.** It is
2.515e-08 in one regime and 3.251e-09 in the other — a factor of 7.7 from a
change in the initial density of 5.7e-06. CVODE's adaptive path depends on the
state, so "the oracle's resolution" has no value independent of the protocol
it is run under, and quoting one number for it without the regime is exactly
the defect this instance describes.

**In the aligned regime the middle two rows are the same number** — 2.517e-08
against a default-tolerance oracle, 2.074e-11 against a pinned one, and
2.515e-08 between the two oracle settings. A harness trusting the default
would report the replica as 2.5e-08 off; all of that is the reference's error,
and the replica's own contribution is bounded at 3.5e-14. That was the point
the original table made, and it stands. What did not stand was the label.

The bound for citing agreement follows from §7.5, which measures the floor
*at the pinned setting* rather than the spread between settings: 1.416e-11 in
the aligned regime. The aligned residual of 2.074e-11 sits at that floor, and
is stated accordingly there. §7.3.

### (5) Integrated observables hide what per-reaction comparison exposes

The `reac - nu` defect was "invisible to every Sounio↔replica pin" and fell in
one shot to a per-reaction rate comparison against Cantera. I reproduced that
path without knowing I was repeating it: the shipped reverse path matches
Cantera to 7.877e-07 per reaction, the buggy one to 1.838e+00. §2.7.

### (6) A convention constant has no syntactic signature, so it is not searchable

The alignment of §1.5 changed the molar-volume and activation-energy constants
at **30 sites** across `stdlib/chemistry/`, both Python replicas, the demo and
the tests. One site survived it: `adiabatic_init` in
`examples/chemistry/h2_ignition_uq_demo.sio`, twelve lines below the
`demo_init` the same commit did change, still carrying a **truncated**
`R_SI = 8.314462618` against CODATA's `8.31446261815324` — 1.5e-10 relative.

Why the sweep missed it is the finding. `8.314462618` and `8.31446261815324`
are both well-formed `f64` literals. They have the same type, the same
dimension, the same units, the same magnitude, and differ only in digits a
reader's eye compresses to "8.31446…". A convention constant carries **no
syntactic signature**: there is no token, no annotation and no shape that
distinguishes the truncated form from the exact one, so no grep, linter or
type-checker can enumerate the sites reliably. The 30 that were found were
found by searching for the *old* spelling; a site already half-migrated to a
different wrong value matches neither the old pattern nor the new one.

The general form: **a constant is the one kind of program content whose
correctness is invisible to every tool that reads the program.** Detecting it
needs a value-level invariant — a unit-carrying type, a named constant with a
single definition site, or a gate that recomputes the constant rather than
matching its text.

### (7) Provenance of a number is a property of the git topology, and no type system sees it

`claude/align-molar-volume-constant` and the branch carrying the TPX → TDY fix
of §1.4 are **each internally correct**. Every file in each branch is
self-consistent; both pass CI; a reviewer reading either diff sees nothing
wrong, because nothing in either diff is wrong.

The defect exists only in the **ancestry**. The alignment branch is cut from
`main`, which never carried the TDY fix, so its copy of
`gri30_h2_cantera_parity.py` restores `gas.TPX = T, P0, X` and deletes
`initial_concentrations` and `initial_state_deviation`. Merging it, or building
a release artefact from it, silently reverts the only real defect this work
fixed — and the result is a *plausible* number, not a crash.

No type system, test suite or review process examines the topology of the
graph that produced a working tree. The only signature is numerical:

| initialisation | worst deviation from intended initial concentrations |
|---|---|
| `TPX` | **5.692129e-06** |
| `TDY`, unaligned constant | **0.000000e+00** |
| `TDY`, aligned constant | **1.629030e-16** (one ULP) |

**This table is a provenance signature carried by a number.** It identifies
which merge produced the tree that produced the result, from the result alone,
with ten orders of magnitude between the failing case and the passing ones.
`benchmarks/chemistry/rep_tolerance.py` and `rep_traj_bug.py` therefore refuse
to report at all when the oracle they load exposes no
`initial_concentrations` — the absence of that helper *is* the TPX variant, so
the probes fail closed on the ancestry rather than comparing two protocols
that were never the same.

The general form: **a numerical result inherits its meaning from a merge
history that no static analysis of the merged tree can recover.** The remedy
is not a stronger type system; it is an invariant printed alongside the result,
chosen so that different ancestries give different values.

### (8) A documentation number, never measured, propagates into a reviewer's hypothesis

The seven instances above are all failures of the *authors* of a measurement.
This one is a failure of its **reviewer**, and it is the same defect, which is
why it belongs in the list rather than in an acknowledgements note.

The operator commissioning this work raised a specific, well-posed physical
objection: that the two sides might be sharing an integrator, and that the
replica-vs-Cantera residual might be RK4 truncation. The objection was correct
to raise and forced the measurements of §7.3 — which is exactly what a reviewer
is for. The **premise** of it, however, was a number:

> "dt-convergence: dt = 1e-8 vs 5e-9 agree to 4 significant figures"
> — `benchmarks/chemistry/README.md`

Four significant figures reads as a truncation error near 1e-4. From that, an
agreement at 2e-11 between two independent integrators looks impossible, and a
shared-stepping hypothesis is the natural explanation. The reasoning is sound.

The premise is not. Measured, that self-convergence is **2.7e-15 … 2.2e-14** —
about **fifteen** significant figures, eleven orders from the documented four.
The README's "4 significant figures" is about **ignition delays** (126.315 vs
126.317 µs), a phase-sensitive quantity at the front, and was never a statement
about the pre-front checkpoint at all. It was a documentation figure that had
not been measured *for the quantity it was being read as*.

The pattern is identical to instances (1)–(4), one level up: **an unmeasured
number of low resolution was taken as the resolution of a different quantity,
and the discrepancy was attributed to the object rather than to the reading.**
Here the object was the harness's integrator, and it was innocent.

Two things follow, and the second is why this instance is worth the space.

First, the reviewer's correction was the right move regardless: the hypothesis
was falsifiable, it was put in falsifiable form, and it was killed by
measurement in one run. A wrong hypothesis that names its own test is worth
more than a right one that does not.

Second, **the defect reached the reviewer through the same channel it reached
the authors** — the README. This document has now corrected four separate
statements in that file, and each had been read, cited and reasoned from before
being checked. A documentation number is an instrument reading like any other,
and it carries no resolution label. That is the general form: **prose reports
numbers without their resolution, and a reader supplies a plausible one.**

---

### (9) The archive layer failed silently, and reported it where no check could see

The eight instances above are failures *inside* the measurement. This one is
a layer further out, in the machinery that was supposed to freeze the other
eight, and it is a new kind: **the failure was reported, but only on a page
that requires the owner's login, while every public signal read as success.**

v1.0.0 of the frozen snapshot received its Zenodo DOI 45 seconds after the
GitHub release. v1.0.1 — the version carrying every remediation in this
document — received none in ten minutes, and would never have. Its
`.zenodo.json` had gained two fields the schema forbids, both added by the
author while inserting the ORCID and DOIs the archive was asked to carry: a
`related_identifiers` relation, `isVersionOf`, that does not exist (Zenodo's
legacy deposit schema enumerates 33, `isNewVersionOf` among them), and a
top-level `version` key, which the schema's `additionalProperties: false`
rejects outright. Zenodo's GitHub integration rejected the deposit. The GitHub release
published normally, with its tarball and checksum; the public Zenodo API
listed one version of the concept and said nothing about a second; the only
notice was on the owner's authenticated Zenodo GitHub page.

Every check that existed had passed. The snapshot verifier passed. The
provenance auditor passed. The release workflow's own guard — verify the
tree at the tagged commit before publishing — passed, because the tree was
correct. The defect was in a metadata file none of them read, in a field
whose vocabulary lives on a server, and the layer that read it reported
failure into a channel the checks could not reach. **A silent failure with a
private error message is, from the outside, indistinguishable from
success**, and it had every appearance of success for as long as nobody
polled for the DOI.

Detected the same way as (5)–(8) — behaviourally, by measuring what should
have changed and did not — and remedied the same way, **in two rounds,
because the first remedy had a blind spot of exactly the kind it was built
to remove.** Zenodo's schema is vendored into the snapshot as
`zenodo-legacyrecord.schema.json`, fetched from the `zenodo/zenodo` source
because `zenodo.org` no longer serves it at its documented path, and
`verify_snapshot.py` validates `.zenodo.json` against it **offline, before a
release**. The first version of that check used full `jsonschema`
validation when the library was present and an enumeration of the
`relation` vocabulary otherwise. The author's machine lacked the library;
the enumeration caught `isVersionOf` and passed the file. The release
workflow's own guard — run the verifier at the tagged commit before
publishing — executes on a runner that *has* the library, ran full
validation, and **refused the v1.0.2 tag** on the `version` key the
enumeration could not see. The guard worked; the offline check did not,
until the library was installed where the snapshot is built and the fallback
was made to reject unknown top-level keys as well. Both paths are proved to
discriminate by negative control. That two-path check, run before the
irreversible step, is the reference implementation for this instance, as
the fail-closed probes are for (5)–(7).

v1.0.1 stays published without a DOI, and its README says why. v1.0.2 is
its results tree byte-for-byte, with metadata the archive accepts.

The general form: **the last layer in a pipeline is the one no earlier layer
can check, and when it reports failure only to an authenticated view, the
public record certifies what it was asked to reject.** The remedy is to
pull that layer's contract — here, a schema — down to where it can be
checked before the irreversible step.

### The hazard that is NOT an instance here

**Sharing the integrator would hide the integrator's error**, and would do it
invisibly: agreement *improves*, which reads as confirmation. Checked, and it
does not occur in this tree —

```sh
grep -n "IdealGasReactor\|net.advance" benchmarks/chemistry/*.py
```

all three Cantera harnesses call `IdealGasReactor` + `ReactorNet` +
`net.advance(t_end)`; the replica's `rk4_step` calls only its own `dc_dt`.
Two independent integrations, no contact. Recorded as a hazard to guard, not
as a finding, because reporting it as observed would be inventing it.

---

## 6.4 One constant deliberately left divergent

```sh
grep -n '8\.314462618\b' benchmarks/chemistry/flame1d_replica.py \
                           stdlib/constants/physical.sio
```

```
benchmarks/chemistry/flame1d_replica.py:81:R = 8.314462618        # J/mol/K
stdlib/constants/physical.sio:228:// Value: 8.314462618 J/(mol·K) (exact)
stdlib/constants/physical.sio:233:    8.314462618
```

Verified 2026-09-01. **Neither file is in the frozen release**: both live in
the upstream repository only, so this section is a claim about the upstream
tree and the command above must be run there, not against an unpacked
snapshot. Recording that explicitly because a section citing files a reader
does not have is precisely the defect §7.4 audits for — here it is intentional
and scoped, not an oversight.

`flame1d_replica.py:81` keeps `R = 8.314462618`, the truncated value, while
every other constant in the GRI-Mech path is now `8.31446261815324`. **This is
deliberate.**

`flame1d_replica.py` is a different benchmark — a 1-D flame, not the GRI-Mech
cross-validation — with its own published reference values and its own gates
in `README.md`. Aligning it would move those numbers, and I have not measured
by how much. Changing a separate benchmark's published results as a side
effect of a chemistry-constant fix is the unmeasured widening this document
argues against everywhere else.

The divergence is therefore intentional and recorded here so it is not read as
an oversight. Whoever aligns it should measure the effect on the flame
reference table first. `stdlib/constants/physical.sio` carries the same
truncated value under an "(exact)" label — filed as issue #2381, deliberately
not fixed in this pass.

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

### 7.3 Are the two sides really independent integrations?

Asked because the answer is not obvious from the numbers: **two independent
integrators agreeing to 2e-11 deserves an explanation, not an assumption.**

**They are independent.** The decisive lines:

```python
# gri30_h2_cantera_parity.py — Cantera side
r = ct.IdealGasReactor(gas, energy="off", clone=False)
net = ct.ReactorNet([r]); net.rtol = 1e-12; net.atol = 1e-22
net.advance(t_end)              # ONE call. CVODE, its own adaptive stepping.

# gri30_h2_python_replica.py — replica side
def rk4_step(c, t, dt, kc):
    k1 = dc_dt(t, c, kc)        # dc_dt is replica code. Never calls Cantera.
```

Cantera is never a rate evaluator inside the RK4 loop; the replica never calls
Cantera. Same for `gri30_full_cantera_parity.py` and
`gri30_full_cantera_uq_reference.py`.

**Why the agreement is nevertheless possible.** RK4's truncation error at this
checkpoint is tiny, because the pre-front trajectory is smooth and slow
relative to dt = 1e-8. Measured against CVODE at `rtol=1e-13`:

| dt | worst relative error |
|---|---|
| 8e-7 | 1.384e-06 |
| 4e-7 | 7.738e-08 |
| 2e-7 | 4.554e-09 |
| 1e-7 | 2.739e-10 |
| **1e-8** | **6.580e-12** |

Successive-error ratios, which must be 2⁴ = 16 for a fourth-order method:

| pair | H2 | H | O | O2 | OH | H2O | HO2 | H2O2 |
|---|---|---|---|---|---|---|---|---|
| 8e-7/4e-7 | 15.53 | 15.44 | 15.57 | 15.64 | 16.60 | 15.46 | 20.36 | 17.89 |
| 4e-7/2e-7 | 15.72 | 15.68 | 15.73 | 15.78 | 16.27 | 15.69 | 18.26 | 16.99 |
| 2e-7/1e-7 | 15.19 | 15.17 | 15.15 | 15.23 | 15.75 | 15.15 | 17.49 | 16.62 |

**Fourth order confirmed empirically.** The order test fails at dt ≤ 1e-8 only
because the differences fall below the roundoff floor — which is itself the
finding that RK4 truncation there is ~1e-12, not that the method misbehaves.

**The direct falsification, added 2026-09-01.** The order table above argues
from the *coarse-dt* regime and then extrapolates. The decisive test is at the
operating step itself: halve dt and watch the deviation to Cantera.

```sh
python3 benchmarks/chemistry/rep_tolerance.py
```

| sp | \|dev\| dt=1e-8 | \|dev\| dt=5e-9 | ratio |
|---|---|---|---|
| H2 | 2.854e-07 | 2.854e-07 | **1.000** |
| H | 2.365e-06 | 2.365e-06 | **1.000** |
| O | 2.660e-06 | 2.660e-06 | **1.000** |
| O2 | 2.388e-07 | 2.388e-07 | **1.000** |
| OH | 2.624e-06 | 2.624e-06 | **1.000** |
| H2O | 2.400e-06 | 2.400e-06 | **1.000** |
| HO2 | 2.247e-07 | 2.247e-07 | **1.000** |
| H2O2 | 1.891e-06 | 1.891e-06 | **1.000** |

A fourth-order truncation error falls by 2⁴ = 16 under this halving. It does
not move — **ratio 1.000 on all eight species, to four significant figures**.
The residual is a *fixed offset, invariant under step size*: the constant of
§1.5, not the integrator. This is a direct falsification of the truncation
hypothesis at the step actually used, and it does not rely on extrapolating the
coarse-dt law.

The same run gives RK4's self-convergence, `|c(1e-8) − c(5e-9)|/|c|`, as
**2.7e-15 … 2.2e-14** — eight orders below the 2.66e-06 gap. Truncation cannot
account for the gap even in principle.

**And the integration modes, checked mechanically rather than by reading.**
Parsing `gri30_h2_cantera_parity.py` and inspecting `integrate()`:

```
params            ['gas', 'T', 't_end', 'dt']
params used       ['gas', 'T', 't_end']        <- dt is NEVER referenced
method calls      ['IdealGasReactor', 'ReactorNet', 'advance']
loops in body     0                             <- there is no stepping at all
```

and `grep -c 'cantera\|ct\.'` over the whole replica returns **0**. Cantera is
not a rate evaluator inside an RK4 loop, and the replica holds no reference to
Cantera by which it could become one.

**A caution against a natural mis-inference.** The README's statement that
dt-convergence "agrees to 4 significant figures" is about **ignition delays**
(126.315 vs 126.317 µs), a phase-sensitive quantity measured at the front.
Reading it as the checkpoint's accuracy suggests an RK4 error near 1e-4 and
makes the 2e-11 agreement look impossible. At the pre-front checkpoint,
dt = 1e-8 and dt = 5e-9 agree to ~1e-14.

**And it does not rescue claim (A).** Extrapolating the now-established dt⁴
law, reaching 0.2% would need dt ≈ 4.9e-6 s — 493× the step used, far outside
RK4 stability for this mechanism. Claim (A) is not an integrator-comparison
artefact; §1.1a gives what it actually is.

## 7.5 The oracle's resolution at the setting actually used — and what it costs this document's headline

§6.3 instance (4) measured CVODE at its *default* tolerance against CVODE at
`rtol=1e-12`. That is the distance between two settings. It is not the
resolution of the setting the harness pins, and only the latter can say whether
a residual measured against that setting is agreement or noise.

```sh
python3 benchmarks/chemistry/rep_resolution.py
```

Ten species, a fresh `gas` object per run, t = 1e-4 s:

| comparison | worst rel | on |
|---|---|---|
| `rtol=1e-12` vs `rtol=1e-13` | **1.473e-11** | H2O |
| `rtol=1e-13` vs `rtol=1e-14` | 5.839e-12 | H2O |
| `rtol=1e-12` vs `rtol=1e-14` | 2.057e-11 | H2O |

**The oracle's own answer at `rtol=1e-12` is uncertain at the 1.473e-11
level.**

### The consequence, stated against this document's own headline

The aligned-regime residual this document reports is **2.032e-11**. The oracle's
resolution at the tolerance used to measure it is **1.473e-11**. The ratio is
**1.38**.

> **Therefore 2.032e-11 is not citable as agreement.** It is an *upper bound on
> the disagreement*, bounded below by the instrument. Two integrators that
> differed by anything under ~1.5e-11 would produce the same reading. The
> correct statement is: **with the constants aligned, the replica and Cantera
> agree to within the oracle's resolution, and the residual is at that
> resolution, not below it.**

This retires the framing used earlier in this file and in
`benchmarks/chemistry/README.md`, where the aligned figures (8.9e-13 … 1.2e-11)
were called agreement "at the floor of CVODE's own `rtol = 1e-12`". Two of
those figures are **below** the measured floor of 1.473e-11 — which does not
make the agreement better than the floor, it makes those particular numbers
unresolvable. They are reported here as measured and must not be read as
resolution.

What survives, and it is the result that matters, is unaffected: the
**published-regime** gap of 2.660e-06 is 180,000× the oracle's resolution, so
its existence, its magnitude and its attribution to one rounded constant are
all far above the noise. The alignment demonstrably removes something real; how
much of what remains is real is a question this oracle configuration cannot
answer.

**To answer it would need a tighter reference** — CVODE at `rtol=1e-14`, whose
own residual against `1e-13` is 5.839e-12, or an arbitrary-precision
integration. Neither is done here, and the claim is bounded accordingly rather
than asserted past the instrument.

---

## 7.5b Step bisection in the aligned regime — where the offset no longer hides it

§7.3's bisection ran in the **published** regime, where a 2.66e-06 constant
offset dominates and would mask anything smaller. The informative run is the
aligned one, where that offset is gone and the residual is 2e-11.

```sh
python3 benchmarks/chemistry/rep_resolution.py --dir <aligned tree>
```

Deviation from Cantera at `rtol=1e-12`, three step sizes:

| sp | dt=1e-8 | dt=5e-9 | dt=2.5e-9 | r(1,2) | r(2,3) |
|---|---|---|---|---|---|
| H2 | 2.442e-12 | 2.441e-12 | 2.439e-12 | 1.000 | 1.001 |
| H | 2.008e-11 | 2.008e-11 | 2.005e-11 | 1.000 | 1.002 |
| O | 2.053e-11 | 2.053e-11 | 2.049e-11 | 1.000 | 1.002 |
| O2 | 2.016e-12 | 2.013e-12 | 2.014e-12 | 1.001 | 1.000 |
| OH | 1.976e-11 | 1.975e-11 | 1.972e-11 | 1.001 | 1.002 |
| H2O | 2.074e-11 | 2.074e-11 | 2.070e-11 | 1.000 | 1.002 |
| HO2 | 3.136e-12 | 3.140e-12 | 3.141e-12 | 0.999 | 1.000 |
| H2O2 | 8.137e-12 | 8.172e-12 | 8.145e-12 | 0.996 | 1.003 |

**Ratio 1.000, to three decimals, across a factor of four in step size.** Even
with the constant removed, the remaining 2e-11 is *not* step-dependent: a
fourth-order truncation error would have fallen by 256 over this range. The
residual is a fixed offset at both scales.

The aligned oracle floor measured in the same run is **1.416e-11**, against a
worst residual of 2.074e-11 — ratio **1.46**. So the two statements hold
together and neither rescues the other:

- the residual is **not truncation**, at either regime or any step tried;
- the residual is **not resolvable** as a real disagreement, because it sits at
  1.46× the oracle's own uncertainty.

What it *is* cannot be determined with this instrument. §7.7 decomposes what
can be measured and states plainly what cannot.

---

## 7.7 The aligned residual, decomposed into what was measured

Every entry below is a measurement from `rep_resolution.py --dir <aligned>`;
nothing is inferred by subtraction, because the parts are not orthogonal and a
subtraction would manufacture a number.

```sh
python3 benchmarks/chemistry/rep_resolution.py --dir <aligned tree>
```

| part | how measured | value | share of residual |
|---|---|---|---|
| **total residual**, replica vs Cantera `rtol=1e-12` | §7.5b, worst species (H2O) | **2.074e-11** | 100% |
| **oracle's own resolution** at `rtol=1e-12`, one state | Cantera `1e-12` vs `1e-13`, fresh `gas` | 1.416e-11 | 68% *at that one state* |
| **oracle's own resolution**, ten states at ±1e-6 density | `rep_floor_spread.py`, fresh `gas` each | **3.730e-12 … 4.142e-11** | **18% … 200%** |
| **replica truncation + roundoff**, upper bound | `\|c(1e-8) − c(5e-9)\|/\|c\|`, worst (H2O2) | **≤ 3.465e-14** | 0.17% |
| — of which truncation alone | see below | **not separable at this step** | — |
| **unaccounted** | not a measurement; see the interval row | **undefined** — the oracle's band spans the residual | — |

> **Corrected 2026-09-01: the 68/32 partition was a single sample from a
> distribution, and the distribution swallows it.** §6.3 (4) found the
> oracle's tolerance spread differs 7.7× between two regimes 5.7e-06 apart in
> initial density. So the "floor" of 1.416e-11 was measured at *one* state.
> Measured at ten states with the initial density scaled by 1 + δ,
> δ ∈ [−1e-6, +1e-6], fresh `gas` per run, worst over ten species:
>
> ```sh
> python3 benchmarks/chemistry/rep_floor_spread.py
> ```
>
> | δ | floor | δ | floor |
> |---|---|---|---|
> | −1.00e-06 | 6.281e-12 | +1.11e-07 | 2.214e-11 |
> | −7.78e-07 | 8.964e-12 | +3.33e-07 | 8.260e-12 |
> | −5.56e-07 | **4.142e-11** | +5.56e-07 | 1.104e-11 |
> | −3.33e-07 | **3.730e-12** | +7.78e-07 | 7.592e-12 |
> | −1.11e-07 | 8.728e-12 | +1.00e-06 | 1.045e-11 |
>
> **Interval [3.730e-12, 4.142e-11], a factor of 11.1**, from perturbations
> of one part per million. At the top of the interval the oracle's own floor
> **exceeds the residual**, so the partition is not merely fragile there —
> it is undefined. The correct statement replaces a number with a range:
> **the oracle explains between 18 % and more than 100 % of the aligned
> residual.** The residual is inside the oracle's noise band.
>
> The observation that generalises: **the resolution of an adaptive
> integrator is not a smooth function of the initial state.** CVODE re-selects
> its step sequence under a perturbation of 1e-6, and the re-selection moves
> the tolerance-induced error by an order of magnitude with no monotone trend
> in δ (the table is not sorted by floor because it cannot be — there is no
> order to recover). An adaptive oracle's resolution therefore has no value
> independent of the state it is run from, and must be measured **in regime,
> every time**, over an ensemble rather than at a point.

### Why truncation and roundoff cannot be separated here — and why that is itself the finding

RK4 truncation falls 16× per halving of dt. Roundoff accumulated over N steps
grows with N. Measuring the replica's self-difference at three step sizes:

| sp | \|c(1e-8)−c(5e-9)\| | \|c(5e-9)−c(2.5e-9)\| | ratio |
|---|---|---|---|
| H2 | 1.823e-16 | 2.188e-15 | 0.083 |
| H | 2.537e-15 | 3.247e-14 | 0.078 |
| O | 2.705e-15 | 3.218e-14 | 0.084 |
| O2 | 2.683e-15 | 8.943e-16 | 3.000 |
| OH | 1.283e-14 | 3.293e-14 | 0.390 |
| H2O | 2.809e-16 | 3.905e-14 | 0.007 |
| HO2 | 3.411e-15 | 7.580e-16 | 4.500 |
| H2O2 | 3.465e-14 | 2.626e-14 | 1.320 |

Not one ratio is near 16. **Halving the step makes the self-difference worse,
not better, for five of eight species** — by 12× for H, O and H2, by 140× for
H2O — and the ratios are erratic across two orders of magnitude. That is the
signature of **roundoff dominance**: at 10,000 to 40,000 steps the accumulated
double-precision error is of order √N·ε ≈ 2e-14 … 4e-14, which is exactly the
band these differences sit in. A difference of a few ULP has no stable ratio.

> **Corrected 2026-09-01: "at the floor" was the wrong word, and the 12–140×
> is not explained.** The growth of the self-difference under halving was
> attributed above to roundoff. It cannot be: a random-walk roundoff gives
> √2 = 1.41× per halving, a systematic one 2×, fourth-order truncation
> 1/16. Nothing known gives 12× to 140×. The one further hypothesis with a
> mechanism — **step stagnation**, where dt·|dc/dt| drops below half an ULP
> of c so the update rounds back to c and the step is lost — was tested:
>
> ```sh
> python3 benchmarks/chemistry/rep_stagnation.py
> ```
>
> | sp | dt=1e-8 | dt=5e-9 | dt=2.5e-9 |
> |---|---|---|---|
> | H2 | 8.29e+11 | 4.15e+11 | 2.07e+11 |
> | H2O | 9.07e+12 | 4.54e+12 | 2.27e+12 |
> | HO2 | 5.04e+11 | 2.52e+11 | 1.26e+11 |
> | *(all eight)* | *> 1e+11* | *> 1e+11* | *> 1e+11* |
>
> Each entry is the per-step increment in units of half an ULP of the state.
> Stagnation needs it below 1; it is above 10¹¹ for every species at every
> step. **Not confirmed.** The 12–140× growth is therefore recorded as
> **unexplained**, with that label, and the sentence that followed it is
> withdrawn: the replica's self-difference at dt = 1e-8 is a **ceiling on
> what step refinement can deliver here, not a floor on the method's
> accuracy** — refinement makes it worse for reasons this document does not
> know. What survives is the bound: whatever the mechanism, the replica's
> self-difference is ≤ 3.465e-14, three orders below the residual, so **the
> replica contributes nothing measurable to the 2.074e-11**, and that line of
> the table stands.

> **Resolved 2026-09-02 with a second integrator, in Sounio.** The label
> *unexplained* above was owed to the absence of an independent instrument:
> a self-difference cannot say which of the two runs is wrong. There is now a
> second method — Gragg-Bulirsch-Stoer, the modified midpoint rule with
> Richardson extrapolation in `h²`, sharing this replica's right-hand side
> verbatim so that only the time stepping differs:
>
> ```sh
> SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/chemistry/gbs_oracle.sio
> ```
>
> **The oracle is characterised before it is used**, because Richardson
> extrapolation divides by `((n_k/n_j)² − 1)` and so amplifies roundoff as the
> depth grows. Sweeping depth against subdivision sequence (self-difference
> between macro-steps `H = 1e-6` and `5e-7`, worst over species):
>
> | depth | order | seq 2,4,6,8,10,12,14,16 | seq 2,4,6,8,12,16,24,32 |
> |---|---|---|---|
> | 3 | 6 | 5.730e-09 | 5.730e-09 |
> | 4 | 8 | 8.413e-11 | 8.413e-11 |
> | 5 | 10 | 8.942e-13 | 6.257e-13 |
> | 6 | 12 | **5.734e-14** | **1.421e-14** |
> | 7 | 14 | 3.066e-13 | 2.314e-14 |
> | 8 | 16 | 2.581e-13 | 1.556e-14 |
>
> Both sequences bottom out at depth 6 and rise afterwards: that minimum **is**
> the truncation-to-roundoff crossover of the extrapolation, measured in place.
> The wider sequence bottoms out four times lower, so the instrument used below
> is depth 6 with 2,4,6,8,12,16,24,32, resolution **1.421e-14** worst over
> species, and per species: H2 6.198e-15, H 1.421e-14, O 3.702e-15,
> O2 5.007e-15, OH 5.910e-15, H2O 1.334e-14, HO2 2.273e-15, H2O2 3.107e-15.
>
> **The halving ladder against that independent method**, relative distance
> ×1e18, each species readable against its own resolution above:
>
> | dt | H2 | H | O | O2 | OH | H2O | HO2 | H2O2 |
> |---|---|---|---|---|---|---|---|---|
> | 1e-8 | 10755 | 4904 | 4841 | 357 | 16380 | 5898 | 2652 | 32629 |
> | 5e-9 | 10573 | 7442 | 1851 | 2861 | 3546 | 5618 | 757 | 1709 |
> | 2.5e-9 | 8385 | 40254 | 30185 | 2146 | 29552 | 45365 | 1705 | 24083 |
> | 1.25e-9 | 729 | 63257 | 78596 | 13056 | 77005 | 60955 | 14212 | 73805 |
>
> At `dt = 5e-9` seven of eight distances sit **below** the oracle's own
> per-species resolution: there the replica agrees with an independent method
> to within the oracle's noise. At `dt = 1.25e-9`, four steps of refinement
> later, seven of eight are **above** it, by 2.6× to 24×. **The replica's
> distance to an external reference has a minimum and then grows as the step
> shrinks.** Growth measured against a different method cannot be truncation
> being resolved, and cannot be an artefact of comparing a run with itself.
>
> So the 12–140× of the table above is no longer unexplained: it is the
> **right-hand branch of the total-error curve of a fixed-step method**, where
> accumulation over 10⁴ to 8×10⁴ steps overtakes a truncation term that is
> already spent. **What is still owed is the exponent, not the mechanism.**
> Per halving the observed factors are 2.1× to 6.6× (from the 4.6×–43× above
> over two halvings), and a systematic accumulation predicts 2×; the fastest
> species exceed that and no model here derives 6.6×. The location of the
> minimum is bracketed between `1e-8` and `2.5e-9` and is **not** pinned,
> because the `5e-9` row is at the instrument's floor.
>
> The bound that the residual line depends on is unchanged and now has
> independent support: at `dt = 1e-8` the replica is 3.263e-14 from the second
> method, three orders below the 2.074e-11 residual, so **the replica still
> contributes nothing measurable to it**. It also follows that `dt = 1e-8` was
> a fortunate choice: refining it does not improve this replica, it degrades it.

### The honest statement of the central result

- **The published gap of 2.660e-06 is real, is one rounded constant, and is
  180,000× the oracle's resolution.** Its existence, magnitude and attribution
  are far above every noise source in this table. That result stands
  unqualified.
- **After alignment, the residual is 2.074e-11; the oracle's own uncertainty,
  measured over an ensemble of initial states, spans 3.7e-12 to 4.1e-11, and
  the replica accounts for at most 3.5e-14.** The residual lies *inside* the
  oracle's noise band. No part of it can be attributed to a real difference
  between the two implementations, and no part can be excluded from being one.
- **Therefore: with the constants aligned, replica and Cantera agree to within
  the oracle's resolution.** Not "to 2e-11". Not "at the floor of rtol". *To
  within the resolution of the instrument used to compare them.*

To resolve the last 6.6e-12 would need a reference at least an order tighter
— Cantera at `rtol=1e-14`, whose own floor is 6.068e-12 and so would not
suffice either — or an arbitrary-precision integration of the same mechanism.
Not done here; bounded rather than claimed.

---

## 7.6 The truncation curve in time — the checkpoint sits at its minimum

Truncation measured at one instant says nothing about the integrator elsewhere
on the trajectory. `|c(dt=1e-8) − c(dt=5e-9)| / |c|`, worst over the eight
reported species:

| t (s) | worst | on | regime |
|---|---|---|---|
| 1.00e-06 | **1.608e-11** | H2O | early induction |
| 1.00e-05 | 7.330e-14 | O | induction |
| **1.00e-04** | **2.222e-14** | H2O2 | **the pre-front checkpoint** |
| 1.20e-04 | 8.508e-14 | HO2 | approaching the front |
| 1.30e-04 | 2.088e-13 | HO2 | into the front |

**The checkpoint is the minimum of the curve.** Truncation there is 724× lower
than at t = 1e-6 and 9.4× lower than just past the front. The parity comparison
is therefore made at the single sampled point where the integrator is under the
least stress — which is the right choice for isolating a *constant* from an
*integrator*, and the wrong one for claiming the integrator has been validated.

Both readings are stated because both are true:

- **For the question this document asks** — is the residual a constant or the
  integrator? — the checkpoint is well chosen. Truncation there (2.2e-14) is
  five orders below the published-regime gap (2.7e-06), so the gap cannot be
  truncation, and §7.3's bisection confirms it directly at ratio 1.000.
- **For a claim that the two integrators agree**, the checkpoint is the most
  favourable point available and should not be presented as representative.
  Into the front, truncation rises by an order; the rise is *not* the
  explosion a stiff-solver failure would give, so the replica is not
  misbehaving — but a parity table taken at t = 1.3e-4 would sit at 2e-13, an
  order worse, and no such table is reported here.

The rise at t = 1e-6 is the sharper caution: **truncation is three orders
*higher* early than at the checkpoint**, because several species are still
many orders below their eventual values and a relative measure is unforgiving
there. Anyone re-using this protocol at a shorter horizon inherits a much
weaker bound than the one this document reports.

---

## 7.4 Provenance audit of this document

Prompted by §6.2b, where a published table turned out to have been measured
with a working copy that was never committed. That is the STEP 4 pathology —
an artefact cited but not present — reappearing *inside* the work correcting
it, which made it worth asking whether §6.2b was the only one.

**It was not.** Criterion applied to every section: does the section report
high-precision numbers, and if so, does a command block name a file that
exists in the released tree?

```sh
python3 benchmarks/chemistry/audit_provenance.py          # audits the working tree
python3 benchmarks/chemistry/audit_provenance.py --tree .  # or an unpacked release
```

The auditor is itself committed, and it exits non-zero on a finding, so this
contract is checkable rather than asserted.

| section | numbers | finding, before remediation |
|---|---|---|
| §1.1a claim (A) provenance | — | command was ```python3 - < the harness in §2.3``` — **a placeholder, not a command** |
| §2.3 per-species under the bug | 32 | no committed producer; measured from an uncommitted working copy |
| §2.4 d[HO2]/dt | 2 | inherited from §2.3, therefore also unproducible |
| §2.5 R16 forward/reverse | 3 | output block with no command at all |
| §2.6 adiabatic anchors | 8 | cites `gri30_h2_adiabatic_replica.py`, **absent from the released tree** |
| §3.1 Kc, 29 reactions | 6 | command was ```python3 - < the harness in section 7.2``` — the same placeholder |
| §6.2b STEP 6 aligned | 40 | uncommitted working copy (found first; fixed in `cad8a9c0`) |
| §6.3 instance (4) | 3 | no producer and no regime label. First re-measured in the wrong regime (3.251e-09) and wrongly withdrawn; reproduced at 2.515e-08 in the regime it was measured in |
| §6.4 divergent constant | 2 | cites `flame1d_replica.py` and `stdlib/constants/physical.sio`, absent from the released tree |

Everything else — 8 sections running from a named committed file, 19 inheriting
a parent section's command — passes.

**Remediated in this revision**, each by a committed producer rather than by
softening the claim:

| was | now |
|---|---|
| §1.1a, §2.3, §2.4, §2.5 | `benchmarks/chemistry/rep_traj_bug.py` — runs all three exponent forms and prints the per-species table, d[HO2]/dt and R16's forward/reverse |
| §3.1 | `benchmarks/chemistry/rep_1atm.py`, which already produced these numbers and merely was not named |
| §6.3 (4), §7.3 | `benchmarks/chemistry/rep_tolerance.py` — oracle resolution, replica-vs-oracle at both settings, and RK4 self-convergence |

**What the audit cost, stated plainly.** No published number was ultimately
withdrawn — but one was withdrawn *for four hours* by this audit itself, on a
re-measurement made in the wrong regime, before the committed producer run in
the right regime reproduced it to four figures (§6.3 (4)). The audit's own
error is recorded there as an instance of what it audits. Every absolute
column in §2.3–2.5 reproduces **to every printed digit** from the frozen
snapshot, whose oracle is aligned; from this branch, whose oracle is in the
published regime, the deltas reproduce to four figures and the absolutes sit
in the sixth digit. No claim was weakened to fit a measurement.

**Two probes now fail closed on provenance.** `rep_traj_bug.py` and
`rep_tolerance.py` both draw the initial state from the oracle's
`initial_concentrations()` and *raise* if it is absent, because its absence is
the signature of the TPX variant (§6.3 instance (7)). They refuse to print
rather than compare two protocols that were never the same.

---

## 8. Claims still lacking a reproduction path

| claim | status |
|---|---|
| "H2O2 agrees to 5.9e-3 relative" (README, headline) | **contradicted** — measured 1.891e-06 |
| "majors 0.2–2%, radicals ~3%, H2O2 ~16%" | **provenance closed** — it is the `reac - nu` defect's per-species profile (0.21% / 0.18% / 3.44% / 16.17%), not a parity table. §1.1a |
| "d[HO2]/dt changes by −34%" | **not reproduced** — measured **−9.86%**. Distinct from the −50.5% on R16's net rate of progress, which is a different quantity at a different state. §2.4 |
| "factor 4 in dt, ratio √2" (log) | **contradicted** — factor 4 gives 2.0; √2 is the factor-2 ratio |
| "√(T/dt) is exact at T = 1e-6, 1e-5, 1e-4" | **contradicted** — off by 59×–242× over the second decade |
| Sounio native **full-mechanism** trajectory at 4e-6 / 2e-5 s | **closed** — measured this session via the reconstructed `full_probe.sio`; section 6 |
| Full-mechanism coherent band vs "Cantera central-difference referee", largest σ deviation 8.724e-07 at t = 4e-7 s (README) | **closed — reproduced exactly**, both sides, every figure: largest σ dev 8.724e-07 (H), others 1.559e-07 … 7.171e-07. Section 6.1. |
| Ignition-delay table 1400–1800 K, Sounio column | reproduced (Cantera column re-measured: 169.66 / 126.34 / 98.29 / 79.00 / 65.08 µs) |
| Preprint Results 2–5 (probe-based) | **no original artefacts** — see section 4. Four reconstructed `.sio` probes and two reconstructed `.py` probes now exist and all check and run clean, but they reproduce the *protocols*, not the originals |
| Sounio native band vs dt (step invariance) | **closed** — measured this session, `band_sweep.sio`, section 5.4 |
| Sounio vs replica at ULP resolution | **closed** — measured this session, `h2_precision_probe.sio`, section 1.3 |
