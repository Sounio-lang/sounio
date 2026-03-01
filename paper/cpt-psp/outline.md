# Type-Safe Pharmacokinetic Modeling with Certified Uncertainty Propagation

**Target:** CPT: Pharmacometrics & Systems Pharmacology (Wiley, IF ~3.5)
**Authors:** Marli Gerenutti (first), Demetrios Chiuratto Agourakis
**Format:** Original Research Article, ~4000 words + figures + tables

---

## Why This Paper Matters

No pharmacometrics paper has ever cited mechanized type theory proofs.
No PL paper has ever been written by a pharmacologist as first author.
This paper lives at a crossroads that doesn't exist yet — and that's the point.

The audience is pharmacometricians, not type theorists. They care about:
- Does it predict drug concentrations correctly?
- Does it handle uncertainty in a way regulators accept?
- Is it better than what they already use (NONMEM, Monolix, R/mrgsolve)?

---

## Structured Outline

### Abstract (~250 words)

Physiologically-based pharmacokinetic (PBPK) modeling requires propagating
measurement uncertainty through complex ODE systems, yet current tools
(NONMEM, Monolix, R/mrgsolve) treat parameters as point estimates and
rely on post-hoc Monte Carlo for uncertainty quantification. We present
a PBPK modeling framework implemented in Sounio, a systems programming
language with native uncertainty types. The framework's `Knowledge<T>`
type attaches GUM-compliant uncertainty to every parameter, and the
compiler statically verifies that uncertainty is never silently discarded.
We evaluate a 14-compartment whole-body PBPK model with Rodgers-Rowland
tissue partitioning, mechanistic hepatic/renal clearance, and
blood-brain barrier transport. Numerical validation against the
analytical Bateman equation achieves <0.001 mg/L error across all time
points. We demonstrate that compile-time dimensional analysis prevents
a class of unit-mismatch errors that are common in pharmacokinetic
practice and invisible to current tools. The framework is open-source
(Apache 2.0) with 6,111 lines of domain-specific code and FDA/EMA
validation metrics built in.

### 1. Introduction (~600 words)

- PBPK modeling is critical for drug development and regulatory submission
- FDA/EMA increasingly accept PBPK for dose adjustment, DDI prediction,
  pediatric extrapolation
- Current tools (NONMEM, Monolix, Simcyp, GastroPlus, PK-Sim) handle
  uncertainty via Monte Carlo or bootstrap — computationally expensive,
  no static guarantees
- The GUM (ISO/IEC 98-3) defines a standard for uncertainty propagation,
  but no PK tool implements it at the language level
- We present a PBPK framework where uncertainty is a TYPE, not a
  post-processing step
- Key claims:
  1. Compile-time dimensional analysis prevents unit errors
  2. GUM-compliant uncertainty propagation through ODE integration
  3. Validated against analytical solutions
  4. FDA/EMA regulatory metrics built into the framework

### 2. Methods (~1500 words)

#### 2.1 The Knowledge<T> Abstraction

- Explain `Knowledge<T>` at a level pharmacometricians understand
- No type theory jargon — focus on "every number knows its uncertainty"
- Show simple example: `let cl: Knowledge<L_per_h> = measure(5.46, 0.5)`
- Explain automatic GUM propagation through arithmetic
- Compare to: manually tracking SD in Excel, or wrapping in MC

#### 2.2 14-Compartment PBPK Model

- Standard whole-body PBPK architecture
- Reference ICRP 110 for physiological parameters
- Rodgers-Rowland partition coefficients (cite original 2006-2007 papers)
- Organ modules: liver (well-stirred + CYP panel), kidney (GFR + tubular),
  brain (BBB + P-gp efflux)

#### 2.3 Dimensional Analysis

- Show how the compiler catches unit errors:
  ```
  let dose: mg = 500.0
  let volume: L = 42.0
  let conc: mg_per_L = dose / volume    // OK — units check
  let wrong = dose + volume             // COMPILE ERROR
  ```
- Discuss prevalence of unit errors in PK practice (cite Mars Climate Orbiter
  and more relevant pharma examples)

#### 2.4 ODE Integration with Uncertainty

- RK4 and Tsit5 solvers
- How uncertainty propagates through each integration step
- The Bateman equation as validation target
- Caffeine as test compound (cite Grzegorzewski 2022)

#### 2.5 Validation Metrics

- GMFE (Geometric Mean Fold Error)
- Percentage within 2-fold
- FDA/EMA acceptance criteria
- Cross-validation: numerical vs analytical Bateman

### 3. Results (~800 words)

#### 3.1 Numerical Accuracy

Table: Caffeine PK — RK4 numerical vs. analytical Bateman equation
- t=1h, 6h, 12h, 24h checkpoints
- Error < 0.001 mg/L at all time points
- Mass balance: 95.5% eliminated at 24h (consistent with 5h half-life)

#### 3.2 Uncertainty Propagation

Table/Figure: Uncertainty bounds through the ODE integration
- Input uncertainties: ka ± 10%, ke ± 15%, V ± 5%
- Output: concentration ± propagated uncertainty at each time point
- Compare GUM analytical vs Monte Carlo (10,000 samples) — agreement

#### 3.3 Compile-Time Error Prevention

Table: Classes of errors caught at compile time vs. runtime
- Unit mismatches (mg vs g, L vs mL)
- Missing uncertainty annotation
- Temporal validity expiration
- Effect leakage (I/O in pure computation)

#### 3.4 Comparison with Existing Tools

Table: Feature comparison
| Feature | NONMEM | Monolix | R/mrgsolve | Sounio |
|---------|--------|---------|------------|--------|
| Static uncertainty | No | No | No | Yes |
| Unit checking | No | No | Partial | Yes |
| GUM compliance | No | No | No | Yes |
| Provenance tracking | No | No | No | Yes |
| ODE solvers | DVERK/LSODA | LSODA | ODE45 | RK4/Tsit5/BDF |
| Open source | No | No | Yes | Yes |

### 4. Discussion (~800 words)

- What this approach gains: static guarantees, audit trails, reproducibility
- What it gives up (for now): PopPK, covariate modeling, MCMC
- Regulatory implications: GUM compliance could streamline review
- Limitations: single-subject (no PopPK yet), no infusion, synthetic validation only
- Future work: Bayesian PopPK integration, real clinical data validation,
  Simcyp/GastroPlus comparison with matched compounds

### 5. Conclusion (~250 words)

- First PBPK framework with compile-time uncertainty verification
- Language-level GUM compliance is a new paradigm for pharmacometrics
- The approach is orthogonal to Monte Carlo — they compose, not compete
- Open-source, reproducible, formally verified

---

## What Needs To Be Generated/Run

1. **Caffeine PK simulation with uncertainty** — run the PBPK model with
   Knowledge<T> parameters, capture output at 1h/6h/12h/24h
2. **Monte Carlo comparison** — run 10,000 MC samples with same parameter
   uncertainties, compare confidence intervals to GUM analytical
3. **Compile-time error examples** — screenshot/listing of actual compiler
   errors when unit mismatches occur
4. **GMFE calculation** — run validation metrics on caffeine + metformin

## Key References To Add

- Rodgers & Rowland (2006, 2007) J Pharm Sci — Kp prediction
- ICRP 110 (2002) — Reference adult physiology
- FDA (2018) PBPK Analyses Guidance for Industry
- EMA (2018) Qualification and Reporting of PBPK
- Jones et al. (2015) CPT:PSP — Best practices for PBPK
- Maharaj et al. (2013) — Simcyp validation methodology
- Jamei et al. (2009) — Simcyp platform paper
- JCGM 100:2008 (GUM)
