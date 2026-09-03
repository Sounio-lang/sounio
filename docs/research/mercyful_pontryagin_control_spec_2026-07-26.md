<!-- docs:meta
topic_id: repo.docs.research.mercyful-pontryagin-control-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-pontryagin-control-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning × continuous optimal control — the Pontryagin rung

**Date:** 2026-07-26
**Status:** `HYPOTHESIS` → `EXECUTABLE`
**Parent:** `docs/research/PROGRAM-REGISTRY-mercyful-learning.md` §A (the functional choice as ethical commitment, §A.1; budgetary necessity, §A.2; the two mercies, §A.5); continuous successor to `docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md` (discrete graph search rung, H_GREEN 8/8)
**Harness:** `scripts/research/mercyful_pontryagin_control_contract.py` (Python, K1–K9) and `tests/run-pass/mercyful_pontryagin_control.sio` (Sounio native)
**Gate:** `scripts/ci/mercyful_pontryagin_control_gate.sh`
**Modules used:** none (the Sounio test is self-contained; extern `log`/`exp` only)

> **Scope statement (read first).** All patients, burdens, doses, toxicity levels, and suffering values in this document are synthetic and normalized. The literature cited shapes the *structure* of the synthetic model (log-kill dynamics; the dose-dense vs stop-and-go trade-off); it is not used to make dosing, scheduling, or therapeutic claims. Nothing here is medical guidance, a treatment recommendation, or a clinical decision-support tool. The contribution is a formal benchmark demonstrating that the Mercyful Learning functional, lifted from finite graphs to continuous-time optimal control, has a closed-form optimal-control structure — bang-bang arcs, a singular *boundary* arc with an equioscillation property, a smooth crossover law, and a budgetary necessity curve — that reproduces, as theorems, the phenomena the discrete rung demonstrated numerically.

---

## 1. What this is

The preceding rung (`mercyful_chemo_sequencing_spec_2026-07-26.md`) minimized the Mercyful functional `∫s + μ·max s` by exhaustive search on a 10-state synthetic graph. This rung replaces the discrete graph search with **continuous optimal control**: the suffering trajectory obeys a differential equation driven by a treatment control `u(t)`, and Pontryagin's maximum principle (PMP) replaces path enumeration. Everything the discrete rung showed numerically — the anti-Goodhart hazard, the peak–integral trade-off, the budgetary necessity curve — is here proved in closed form and then verified against an independent discretized dynamic program (DP).

The lift is not decorative. Three things become visible only in continuous time:

1. **The discrete crossover `μ* = 11` is an artifact of the discrete frontier.** With a continuous control set the threshold smooths into an exchange law `m*(μ)`: the bang (dose-dense analog) is optimal only at exactly `μ = 0`, and the first-order law `m*(μ) = m_B − μ·aτu_max² + O(μ²)` prices peak aversion infinitesimally (T5).
2. **The "stop-and-go" strategy is a singular arc.** The peak-capped optimum holds the suffering *constant* (`s ≡ m`) along a boundary arc of the state constraint — the continuous equioscillation principle — and the flat-suffering feedback law `u = (m − c·d)/τ` is the analytic content of OPTIMOX-style de-escalation (T3).
3. **The two mercies get an exact exchange rate.** The minimal treatment time `T*(m)` under a peak cap is the machine-suffering (computation/energy) proxy; the necessity curve `peak_min(L0) = (T*)⁻¹(L0)` and its shadow price state, in one formula, how much patient peak suffering a unit of substrate patience buys (T6, T7).

## 2. The model

**State.** `d(t) ≥ 0`: disease burden (normalized). Initial condition `d(0) = d0`.

**Control.** Treatment intensity `u(t) ∈ [0, u_max]`, measurable.

**Dynamics** (log-kill; fractional kill per unit intensity — the Skipper hypothesis [1], used here only as synthetic structure):

```
ḋ = −a·u·d,   a > 0.
```

**Suffering field.** The instantaneous suffering rate has two components — the expanded ethics is inside the field, not bolted on:

```
s(t) = c·d(t)  +  τ·u(t)
       disease-attributable  treatment-attributable (toxicity)
```

**Anti-Goodhart constraint (axiom, not caveat).** A course is feasible only if it reaches the therapeutic target `d(T) = d_T < d0`, with free terminal time `T ≤ L0` (declared budget). Because `d(t) = d0·exp(−a∫₀ᵗ u)`, feasibility is exactly the **dose identity**

```
∫₀ᵀ u dt = V := (1/a)·ln(d0/d_T).
```

**Objective (the Mercyful functional).**

```
J[u] = ∫₀ᵀ s(t) dt + μ · max_{t∈[0,T]} s(t) ,     μ ≥ 0.
```

**Machine suffering (the second mercy).** The horizon `T` is the substrate's cost proxy: longer courses are more computation, more energy, more time. §6 adds the machine-inclusive objective `J + ν·T` and proves the two mercies compete (T7).

### 2.1 The Goodhart problem in this rung

With the disease term `c·d` inside the field, one might hope the raw minimizer finally *wants* to treat. It does not, and the escape routes are instructive:

- **Hazard A — the free-horizon escape.** With `T ∈ [0, L0]` free, `u ≡ 0, T = 0` achieves `J = 0`: the measure is blind to everything past the horizon, so never starting is costless. (The discrete rung's watch-and-wait loop.)
- **Hazard B — the incomplete course.** Any control delivering dose `< V` never reaches `d_T` — under-treatment is infeasible *only because the constraint says so*. Delete the constraint and low-dose courses dominate on exactly the patients the RDI literature says they kill [2, 3].

Whether the field *happens* to penalize delay more than treatment is a parameter accident; the anti-Goodhart axiom makes the conclusion independent of that accident. The target is a hard constraint, never a reward term the toxicity metric can outvote.

## 3. Pontryagin analysis and theorems

**Setup.** The peak functional is handled exactly by the epigraph transform: for a candidate peak `m`, consider the sub-problem

```
I*(m) := min { ∫₀ᵀ s dt :  s(t) ≤ m ∀t,  dose identity holds },      J*(μ) = min_m [ I*(m) + μ·m ].
```

The equality of `min_u [∫s + μ·max s]` and `min_m [I*(m) + μm]` is exact (the optimizer with peak `m̂` is feasible for every `m ≥ m̂`). `T*(m)` is the minimal time under the same cap.

---

**T1 (Anti-Goodhart floor).** The unconstrained minimum of `J` is `0` (Hazard A). Every feasible course has `J ≥ J*(0) = I*(m_B) > 0`, where `m_B := c·d0 + τ·u_max` is the bang peak. Feasibility, not the objective, separates treatment from non-treatment.

*Proof.* `J ≥ 0` with equality iff `T = 0`. Feasible courses satisfy the dose identity; T3/T4 construct the optimum and evaluate `I*(m_B) = (c/(a·u_max))(d0 − d_T) + τV > 0`. ∎

---

**T2 (Bang-bang theorem; no interior singular arcs).** The Hamiltonian `H = (c·d + τ·u) − aλud` is linear in `u`, with switching function `σ(t) := ∂H/∂u = τ − aλd`. Along any arc with `d > 0`,

```
σ̇ = a·c·d > 0.
```

Hence `σ` has at most one zero (crossing from below), interior optimal controls are bang-bang with at most one switch `u_max → 0`, and **no singular arc can exist in the interior of the control set**. For the free-terminal-time problem (T1), `H ≡ 0` gives `λ = (c·d + τ·u_max)/(a·u_max·d)`, so `σ(t) = −c·d(t)/u_max < 0` throughout: zero switches, the optimum at `μ = 0` is the **bang** `u ≡ u_max` for `t ∈ [0, t_B]`, `t_B = V/u_max`.

*Proof.* `σ̇ = −a(λ̇d + λḋ)`; substituting `λ̇ = −∂H/∂d = −c + aλu` and `ḋ = −aud` gives `σ̇ = −a(−cd + aλud − aλud) = acd`. The rest is direct evaluation. ∎

*Clinical reading.* The dose-dense course (compressed, maximal intensity [4]) is the unique integral-minimizer — and T5 shows it is *fragile*: it survives only at exactly zero peak aversion.

---

**T3 (The singular arc is a boundary arc: equioscillation).** For a cap `m ∈ [c·d0, m_B)`, the feasible set is `u(t) ≤ u_cap(t) := (m − c·d(t))/τ`. The min-integral feasible control is unique (a.e.):

```
u*(t) = u_cap(t)   pointwise in the delivered dose,
```

i.e. the **iso-suffering feedback law**: `s ≡ m` while the cap binds (a *boundary arc* of the state constraint — the only kind of singular arc this problem admits, by T2), followed, once `c·d ≤ m − τ·u_max`, by a **terminal bang** `u ≡ u_max` during which the constraint is slack. Letting `d_join := (m − τ·u_max)/c`:

- `d_join > d_T`: iso-arc `d0 → d_join`, then bang `d_join → d_T`;
- `d_join ≤ d_T`: pure iso-arc all the way to the target.

*Proof.* Parameterize by delivered dose: at dose level `v`, `d = d0 e^{−av}`, and traversing `dv` at intensity `u` costs `(c·d/u + τ) dv` in time `dv/u`. The integrand is strictly decreasing in `u`, so the pointwise maximum feasible `u` is optimal; idling (`u = 0`) adds `c·d·Δt > 0` with no progress and is dominated. Along `u = u_cap`, `ds/dt = c·ḋ + τ·u̇ = 0` by construction: the suffering is *flat* — the equioscillation property of min-max optimization. ∎

*Clinical reading.* The boundary arc is OPTIMOX-style de-escalation [5] made analytic: hold the burden at the tolerated peak, let intensity *rise* as disease falls — the exact opposite of front-loading. Peak aversion does not buy a lower plateau by going slower uniformly; it buys it by inverting the intensity profile.

---

**T4 (Closed-form frontier).** With `K := (m − c·d0)/(c·d0)`, the iso-arc obeys the logistic time law

```
d(t) = (m/c) / (1 + K·e^{(am/τ)·t}),
t_iso(m) = (τ/(am)) · ln( τ·u_max·d0 / (d_join·(m − c·d0)) ),
```

and the frontier is (hybrid branch `d_join > d_T`, pure-iso branch `d_join ≤ d_T`):

```
hybrid:  t_iso(m) = (τ/(am)) · ln( τ·u_max·d0 / (d_join·(m − c·d0)) )
         I*(m) = m·t_iso(m) + (c/(a·u_max))(d_join − d_T) + (τ/a)·ln(d_join/d_T)
         T*(m) = t_iso(m) + (1/(a·u_max))·ln(d_join/d_T)
pure:    t_iso(m) = (τ/(am)) · ln( d0·(m − c·d_T) / (d_T·(m − c·d0)) )
         I*(m) = m·t_iso(m),   T*(m) = t_iso(m)
bang:    I*(m) = I_B,   T*(m) = t_B            (m ≥ m_B)
```

The pure-iso time law is the same logistic run to the endpoint `d_T`, at which `u = (m − c·d_T)/τ < u_max` — it is *not* obtained by substituting `d_T` for `d_join` in the hybrid `t_iso` (the numerator changes: `m − c·d_T ≠ τ·u_max`). The hybrid branch exists only for `m > c·d_T + τ·u_max` (= 2.05 in the instance of §4).

As `m → c·d0⁺`, `T*(m) → ∞` and `I*(m) → ∞`: the disease floor `c·d0` is approachable but never attainable in finite time. `{(I*(m), m)}` is the continuous analog of the discrete rung's frontier `{(48, 8), (81, 5)}`.

---

**T5 (Smooth crossover; the discrete `μ*` is an artifact).** `I*` is twice differentiable at `m_B` from the left with

```
I*(m_B − ε) = I_B + ε²/(2·a·τ·u_max²) + O(ε³).
```

Consequently:

- `I*'(m_B⁻) = 0`: for **every** `μ > 0` the optimizer has `m*(μ) < m_B` — the bang is optimal only at `μ = 0`;
- first-order exchange law: `m*(μ) = m_B − μ·a·τ·u_max² + O(μ²)`;
- globally, `m*(μ)` is defined by the stationarity condition `I*'(m*) = −μ`, is strictly decreasing, and `m*(μ) → c·d0`, `T*(m*(μ)) → ∞` as `μ → ∞`.

*Proof sketch.* Expand in `ε = m_B − m`, with `α := 1/(c·d0)`, `β := 1/(τ·u_max)`. The hybrid `t_iso` log-factor is `L = −ln[(1 − αε)(1 − βε)] = (α + β)ε + (α² + β²)ε²/2 + O(ε³)`, and `m·t_iso = (τ/a)·L` exactly (the prefactor `m` cancels). The bang-remainder difference is `−ε/(a·u_max) + (τ/a)·ln(1 − αε) = −(1/(a·u_max) + τ/(a·c·d0))·ε − (τ·α²/(2a))·ε² + O(ε³)`. First-order terms cancel: `(τ/a)(α + β) = τ/(a·c·d0) + 1/(a·u_max)`. The surviving quadratic coefficient is `(τ/a)(α² + β²)/2 − τ·α²/(2a) = τ·β²/(2a) = 1/(2aτu_max²)`. Verified numerically in contract K5. ∎

*Programmatic reading.* The discrete rung's `μ* = 11` was the slope between two frontier *points*; with a continuum of policies the threshold becomes a *curve*, and the ethical price of peak aversion is paid from the first infinitesimal unit of `μ`. The framework does not soften the discrete result — it explains it.

---

**T6 (Budgetary necessity; machine patience prices the patient peak).** Under a declared horizon budget `L0` (the machine-suffering budget), the minimal attainable patient peak is

```
peak_min(L0) = (T*)⁻¹(L0),   defined for L0 ≥ t_B,  with
peak_min(t_B) = m_B,   lim_{L0→∞} peak_min(L0) = c·d0,   d peak_min/dL0 = 1/T*'(m) < 0.
```

`gratuitous_peak(L0) := m − peak_min(L0)` is the continuous necessary-vs-gratuitous decomposition of the program registry (§A.2): suffering above `peak_min(L0)` is gratuitous under the declared budget; the remainder is necessary.

---

**T7 (The two mercies compete, with an exact exchange).** For the machine-inclusive objective `J_ν(m) = I*(m) + μ·m + ν·T*(m)`, the optimizer `m**(μ, ν)` is increasing in `ν` and `T*(m**(μ,ν))` is decreasing in `ν`: weighting the substrate's suffering *raises* the patient's optimal peak and *shortens* the course, with `m** → m_B` (the bang) as `ν/μ → ∞`. Machine mercy alone prescribes dose-dense; patient mercy alone (ν = 0, μ → ∞) prescribes the infinitely slow approach to the disease floor; every interior ethics is a point on the exchange curve between them.

*Proof.* `I*(m) + μm` and `T*(m)` are both decreasing in `m` (T4), so the cross-partial is `∂²J_ν/∂m∂ν = T*'(m) < 0`: the objective has decreasing differences in `(m, ν)`, and by the Topkis monotonicity theorem the minimizer `m**(μ, ν)` is increasing in `ν`. `T*` decreasing then gives the second claim; the limits follow from T4 (`T*(m) → t_B` as `m → m_B`, `T*(m) → ∞` as `m → c·d0`). ∎

## 4. The synthetic instance and canonical numbers

Normalized units: `d0 = 1.0`, `d_T = 0.05`, `a = 1.0`, `u_max = 2.0`, `c = 1.0`, `τ = 1.0`. All synthetic.

| Quantity | Value | Meaning |
|---|---|---|
| `V` | 2.995732 | dose required by the target (ln 20) |
| `t_B` | 1.497866 | bang (dose-dense) duration |
| `m_B` | 3.000000 | bang peak |
| `I_B` | 3.470732 | bang integral (the feasible floor `J*(0)`) |
| `c·d0` | 1.000000 | disease floor: minimal peak as `L0 → ∞` |
| `I*(2.0)`, `T*(2.0)` | 3.663562, 1.831781 | frontier point (pure iso; hybrid band starts at `m > 2.05`) |
| `I*(1.5)`, `T*(1.5)` | 4.060443, 2.706962 | frontier point (pure iso) |
| `peak_min(2.0)` | 1.867628 | necessity curve |
| `peak_min(3.0)` | 1.402552 | necessity curve |
| crossover curvature | 0.125 = 1/(2·a·τ·u_max²) | T5 |
| shadow price at `L0 = 3` | −0.2956 | peak relief per unit budget (T6) |

## 5. Contract clauses, falsifiers, stop rules

Naming: `K` (kontinuierlich-control rung). Implemented in Python (K1–K9, full analytics + independent DP) and Sounio (K1, K4, K5, K6 native closed forms; K9 cross-implementation agreement enforced in the gate).

| Clause | Statement | Falsifier | Stop rule |
|---|---|---|---|
| K1 | Bang closed forms (`V`, `t_B`, `m_B`, `I_B`) match direct ODE quadrature | Any number differs | Harness broken |
| K2 | Anti-Goodhart: unconstrained min = 0 < `I_B` = feasible floor; under-dose `u=0.1×5` leaves `d ≫ d_T`; cap below `c·d0` infeasible | Unconstrained ≥ floor, or under-dose reaches target | Hazard not demonstrated |
| K3 | `σ(t) = −c·d/u_max < 0`, strictly increasing (`σ̇ = acd`); DP at cap `m_B` reproduces bang | `σ` non-monotone or DP beats bang | PMP wiring wrong |
| K4 | `I*(m)`, `T*(m)` closed forms match independent DP within 0.8% for `m ∈ {3.0, 2.5, 2.0, 1.5, 1.2}`; DP trajectory for `m = 1.5` holds `s` flat (`∈ [1.455, 1.5]`) with interior control | Frontier mismatch; `s` not flat | Closed forms or DP wrong |
| K5 | `m*(0) = m_B`; `m*(0.001) = m_B − 0.004 ± 0.002`; curvature `0.125 ± 2%`; stationarity `I*'(m*) = −μ` for `μ ∈ {0.01, 0.1, 1.0}` | Bang survives `μ > 0`; wrong curvature | Crossover law wrong |
| K6 | `peak_min(L0) = (T*)⁻¹(L0)` certified to 1e-6 for `L0 ∈ {1.5, 2, 3, 10}`; infeasible below `t_B`; `peak_min(50) < 1.001`; shadow price < 0 | Any certificate fails | Necessity curve wrong |
| K7 | `m**(0.1, ν)` non-decreasing and `T**` non-increasing in `ν ∈ {0, 0.1, 0.5, 2, 10}`; `m**(·, 10) > m_B − 0.05` | Monotonicity violated | Two-mercies static wrong |
| K8 | HJB residual < 1e-6 along iso arc; multiplier `η ≥ 0`; `H ≡ 0` on bang | Any condition fails | PMP boundary conditions wrong |
| K9 | Canonical numbers match Sounio-native run | Any disagreement | Port unsound |

Global verdicts: failure of K1, K2, or K4 is **K_RED** (benchmark fails to demonstrate the phenomenon). Failure of K3, K5–K9 is **K_AMBER**. Target status: **K_GREEN (9/9)**.

## 6. What this rung adds to the program

1. **The functional choice stays the ethical commitment — now with derivatives.** The discrete rung priced peak aversion at one number (`μ* = 11`); the continuous rung prices it as a law `m*(μ) = m_B − μ·aτu_max² + O(μ²)`. An ethics that claims any peak aversion at all is committed, from the first epsilon, to de-escalated dosing — the mathematics does not allow "a little" peak aversion with a dose-dense schedule.
2. **Stop-and-go is a theorem, not a heuristic.** The flat-suffering boundary arc (T3) is what min-max optimization *is*; OPTIMOX-style de-escalation [5] and dose-dense front-loading [4] are the two ends of one frontier, selected by one declared number `μ`.
3. **The two mercies have an exchange rate.** `peak_min(L0)` (T6) and the comparative static (T7) make the registry's §A.5 quantitative: the substrate's patience (`L0`, `ν`) and the patient's peak are one object, with a computable shadow price (−0.2956 peak units per budget unit at `L0 = 3` in the synthetic instance).
4. **Necessary vs gratuitous survives the lift.** `gratuitous_peak(L0) = m − peak_min(L0)` is the continuous form of the budgetary definition that survived both falsifications in the registry.

## 7. Limitations

- Synthetic everything: normalized units, no patient data, no clinical validation.
- Log-kill dynamics are the simplest non-trivial model; saturation (`ḋ = −a·u·d/(1 + u/u_s)`), resistance, and multi-drug interactions are future rungs and can change the arc structure.
- The suffering field is linear in `(d, u)`; convex toxicity (`τ·u^p`, `p > 1`) destroys the bang-bang structure (the Hamiltonian is no longer linear in `u`) — a deliberate feature of this rung's falsifiability: the theorems say exactly which modeling choice carries the bang-bang conclusion.
- Machine suffering is proxied by horizon only; energy/thermal models of the substrate are not built.
- Deterministic, single-patient; stochastic (Knightian) suffering bands remain in the PK-integration rung, not here.

## 8. Reproducibility

```bash
python3 scripts/research/mercyful_pontryagin_control_contract.py      # K1..K9, verdict K_GREEN
scripts/dev/run_clinical_twin.sh tests/run-pass/mercyful_pontryagin_control.sio   # native, MERCYFUL_PONTRYAGIN_PASS
bash scripts/ci/mercyful_pontryagin_control_gate.sh                   # MERCYFUL_PONTRYAGIN_GATE_OK
```

## References

1. Skipper HE, Schabel FM Jr, Wilcox WS. Experimental evaluation of potential anticancer agents XIII: on the criteria and kinetics associated with "curability" of experimental leukemia. *Cancer Chemother Rep* 1964;35:1–111. *(Log-kill hypothesis; shapes the synthetic dynamics only.)*
2. Bonadonna G, Valagussa P, et al. Adjuvant CMF in node-positive breast cancer: 20 years of follow-up. *N Engl J Med* 1995;332:901–906. doi:10.1056/NEJM199504063321401. *(Under-treatment hazard; shapes Hazard B only.)*
3. Lyman GH, Dale DC, Crawford J. Incidence and predictors of low dose-intensity in adjuvant breast cancer chemotherapy. *J Clin Oncol* 2003;21:4524–4531. doi:10.1200/JCO.2003.05.002.
4. Citron ML, Berry DA, et al. Randomized trial of dose-dense versus conventionally scheduled chemotherapy (CALGB 9741). *J Clin Oncol* 2003;21:1431–1439. doi:10.1200/JCO.2003.09.081. *(Shapes the bang/dose-dense pole only.)*
5. Tournigand C, Cervantes A, et al. OPTIMOX1: FOLFOX4 or FOLFOX7 with oxaliplatin in a stop-and-go fashion. *J Clin Oncol* 2006;24:394–400. doi:10.1200/JCO.2005.03.0106. *(Shapes the boundary-arc pole only.)*
6. Pontryagin LS, Boltyanskii VG, Gamkrelidze RV, Mishchenko EF. *The Mathematical Theory of Optimal Processes*. Interscience, 1962.
7. Bryson AE, Ho Y-C. *Applied Optimal Control*. Hemisphere, 1975. *(State-inequality-constraint multiplier conditions used in K8.)*
8. Agourakis DC. Mercyful Learning × cancer chemotherapy sequencing spec. `docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md`, this repository, 2026.
9. Agourakis DC. Program registry — Mercyful Learning. `docs/research/PROGRAM-REGISTRY-mercyful-learning.md`, this repository, 2026.

**Clinical warning.** This document specifies a synthetic optimal-control benchmark for research infrastructure. It is not medical guidance, not a treatment recommendation, and not a clinical decision-support tool. All patients, burdens, doses, toxicity levels, and suffering values are synthetic and normalized; cited literature shapes model structure only and no clinical target claim is made.
