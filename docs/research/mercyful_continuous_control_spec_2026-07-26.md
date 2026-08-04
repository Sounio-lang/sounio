<!-- docs:meta
topic_id: repo.docs.research.mercyful-continuous-control-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-continuous-control-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — upgraded algorithm validation: suffering field + continuous optimal control vs the discrete scheduler

**Date:** 2026-07-26
**Status:** `HYPOTHESIS` → `EXECUTABLE` (V_GREEN 10/10)
**Parents:** `docs/research/mercyful_runtime_spec_2026-07-25.md` (discrete scheduler, M_GREEN); `docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md` (H_GREEN); `docs/research/mimic_iv_mercyful_validation_2026-07-26.md` (vancomycin field values); `docs/research/mercyful-learning.md` (foundational proposal)
**Sibling rungs validated here:** `docs/research/mercyful_learned_suffering_field_spec_2026-07-26.md` (L-rung: learned patient + machine suffering field); `docs/research/mercyful_pontryagin_control_spec_2026-07-26.md` (K-rung: Pontryagin continuous optimal control, K1–K9). The L- and K-rungs build the upgraded algorithm; this V-rung is the comparison validation against the discrete scheduler across all three applications.
**Harness:** `scripts/research/mercyful_continuous_control_contract.py` (V1–V10, pure Python stdlib)
**Gate:** `scripts/ci/mercyful_continuous_control_gate.sh`
**Baseline reused unmodified:** `scripts/research/mercyful_runtime_contract.py`, `scripts/research/mercyful_chemo_contract.py`

> **Scope statement (read first).** All patients, doses, regimens, toxicity grades, suffering values, time constants, and windows in this document are synthetic. The suffering values reused from the vancomycin rung were themselves measured from a synthetic Knightian twin. Nothing here is medical guidance, a treatment recommendation, or a clinical decision-support tool. The contribution is an executable, falsifiable validation that replacing the discrete graph scheduler with a continuous suffering field plus continuous optimal control is **strictly more powerful** — better solutions, more general problem classes, better scaling — while the anti-Goodhart target constraint (the ethical core) is preserved unchanged.

---

## 1. What this is

Mercyful Learning minimizes accumulated suffering instead of maximizing reward. Until this rung, the only executable realization was the **discrete scheduler**: minimize

```
cost(γ; μ) = Σ_{(u,v)∈γ} s(u)·ℓ((u,v)) + μ · max_{v∈γ} s(v)
```

over simple paths γ of a finite graph, subject to reaching a target node within a budget L0 (the anti-Goodhart constraint). This rung defines and validates the **upgraded algorithm**: the suffering measure becomes a continuous field `s(x)` on a continuous state space, the plan becomes a control trajectory `u(t)`, and the objective becomes the expanded-ethics functional

```
J[u] = ∫₀^T [ s_patient(x(t))  +  σ·‖u(t)‖² ] dt  +  μ · sup_{t∈[0,T]} s_patient(x(t))
s.t.   ẋ = f(x,u),   x(0) = x₀,   x(T) ∈ TARGET  (anti-Goodhart),   T ≤ L0
```

- `s_patient` — **patient suffering** (the continuous extension of the discrete per-state field);
- `σ·‖u‖²` — **machine suffering**: control energy, dissipation, switching strain. This is the expanded ethics made mathematical: the plan is charged for what it costs the substrate that carries it out, not only for what it costs the patient. In the exposure application the same quadratic control term has a dual reading as ascent strain (driving the therapeutic system too fast is itself a harm);
- `μ·sup` — worst-moment (Rawlsian) peak aversion, as in the discrete scheduler;
- the hard terminal constraint is the **anti-Goodhart axiom**, unchanged: plans that do not reach the target are infeasible regardless of cost.

The discrete scheduler is exactly this problem restricted to bang-bang, graph-edge controls with left-endpoint Riemann quadrature. Everything proved below follows from that observation plus three elementary analytic tools (AM–GM, Cauchy–Schwarz, one-dimensional FOC).

### 1.1 On novelty (honesty note)

Following the preprint's own §7.6: the upgrade is **standard optimal control** — calculus of variations, convex duality, Pontryagin-style reasoning. No new algorithmic primitive is claimed. The claimed contributions of this rung are narrower and executable:

1. a **shared-instance, number-for-number comparison** showing the upgraded algorithm strictly dominates the discrete scheduler on all three established applications;
2. the identification of the discrete scheduler's cost model as a **biased quadrature** (left-endpoint) of the underlying field, with a measured 3× distortion on the exposure instance;
3. the **machine-suffering term** `σ‖u‖²` made operational and shown to be decision-relevant (it is what makes smooth plans strictly better than bang-bang plans, V10);
4. a **generality** result: continuous anti-Goodhart targets and continuum frontiers the graph cannot express (V5, V8);
5. an **efficiency** result: exact discrete enumeration is exponential in horizon (asserted combinatorially and measured), the continuous solver polynomial in grid resolution (V9).

---

## 2. The three theorems the contract validates

### T1 — Consistency (dominance)

Every discrete schedule lifts to a feasible continuous control (piecewise-constant rates on the same field), so

```
min over continuous controls  J  ≤  min over discrete schedules  J     (same objective).
```

Verified per application (V1): exposure `5.5833 ≤ 7.3333 ≤ 12` (relaxed continuous ≤ true cost of the discrete trajectory ≤ discrete-reported cost); chemo `(96, 4)` ≤ every lifted regimen; vancomycin `J(t*) = 17.907183 ≤ J(t_discrete) = 19.644014`.

### T2 — Strictness

In each application the discrete optimum is **not** a stationary point of the continuous problem, so the inequality is strict. Mechanisms, one per application:

- **Exposure (quadrature + pacing).** The discrete scheduler charges `s(left endpoint)` per unit edge: the same physical trajectory (traverse avoidance→mild→moderate→recovery at unit pace) truly accumulates `∫₀¹ s(x) dx = 7/3`, not `7` — the descent from the peak is billed at peak rate for a whole unit. Under the expanded objective `L(x,v) = s(x)/v + κv` (patient field + ascent strain), AM–GM gives the pointwise optimal pace `v*(x) = clamp(√(s(x)/κ), v_lo, v_hi)` with optimal value `2√(κ·s(x))`: **slow where the field is high, fast where it is low** — the mercyful titration profile. Cost `≈ 2.8597` vs `10/3 ≈ 3.3333` for the constant unit pace (the only profile the graph expresses). Peak suffering (5) is unchanged: mercy does not delete the necessary passage through moderate distress, it removes the gratuitous part around it.
- **Chemo (convexity).** Toxicity rate `s(d) = d²` (convex, synthetic), efficacy `∫d dt ≥ K` (anti-Goodhart), budget `T ≤ L0`. Cauchy–Schwarz: `∫d² dt ≥ K²/T ≥ K²/L0`, equality iff `d ≡ K/L0`. The constant-rate plan `(K²/L0, (K/L0)²) = (96, 4)` at `K=48, L0=24` strictly dominates all three discrete regimens lifted to the same delivered dose: DD `(288, 36)`, stop-and-go `(133.7942, 8.7791)`, continuous `(160.6531, 20.8980)`.
- **Vancomycin (timing).** With `s_post(t) = s_ss + (s_pre − s_ss)e^{−t/τ}` (a level drawn before steady state is only partly informative), `J(t) = s_pre·t + s_post(t)·(L0 − t)` has the unique interior minimizer solving `e^{t/τ} = 1 + (L0 − t)/τ`; at the synthetic values (`τ=12, L0=48`) `t* ≈ 15.68`, strictly better than the discrete fixed dwell `t=24` (`17.907 vs 19.644`, −8.8%) and than both endpoints.

### T3 — Generality and efficiency

- **Continuum frontier (V5).** The upgraded algorithm returns the whole trade-off curve `J(L0) = (K²/L0, (K/L0)²)`; the discrete frontier is two points. Notably, the discrete dose-dense course *equals* the continuous optimum at its own 8-week budget — the discrete scheduler had found a genuinely optimal plan, and the upgrade *proves* it — while the stop-and-go point is strictly dominated at its 24-week budget, and intermediate budgets (e.g. `L0=12 → (192, 16)`) are served only continuously.
- **Off-node targets (V8).** The anti-Goodhart target becomes a terminal constraint `g(x(T)) = 0`: recovery threshold `x* = 0.9` is solvable continuously (cost 2.6964, within budget) and is structurally inexpressible for the discrete scheduler, whose target must be a graph node.
- **Scaling (V9).** On width-3 layered graphs with path length `k` (k hops, k+1 layers, `3(k+1)` states) and **fixed** start/target nodes, the discrete problem has exactly `3^(k−1)` start–target paths: the two endpoints are fixed, so only the `k−1` intermediate layers offer a free choice of 3 nodes each (asserted combinatorially and by enumeration: `k=4 → 27`, `k=6 → 243`; measured enumeration time growing ×9.3 per +2 hops). The continuous solver's cost is polynomial in its grid resolution M (measured ×3.9 per 4× grid) and does not see the path count at all. The Sounio-native discrete scheduler is additionally hard-capped at 16 states, so it cannot even *represent* width-3 instances with `k ≥ 5` (18 states at k=5; 21 at k=6).

---

## 3. Application details and verified numbers

### 3.1 Exposure therapy

Continuous field: piecewise linear through `(0,0), (1/3,2), (2/3,5), (1,0)` — the continuous extension of the discrete field `avoidance=0, mild=2, moderate=5, recovery=0`. Control: pace `v = ẋ ∈ [0.05, 4]` (v=1 reproduces the discrete schedule).

| Quantity | Discrete scheduler | Upgraded algorithm | Clause |
|---|---|---|---|
| Reported accumulated suffering, baseline trajectory | 7 (left-endpoint sum) | 7/3 = 2.3333 (exact ∫s dx) | V2 |
| Total at μ=1, baseline | 12 | 7.3333 | V1, V2 |
| Expanded objective (field + strain), constant v=1 | 10/3 = 3.3333 (only expressible profile) | 2.859673 (optimal pacing) | V3 |
| AM–GM lower bound `2√κ·∫√s dx` | — | 2.859668 (achieved up to clamp) | V3 |
| Peak | 5 | 5 (unchanged — necessary suffering) | V3 |
| Off-node target x*=0.9 | inexpressible | cost 2.696372, T=0.78 ≤ L0 | V8 |

Falsifier: 20 000 seeded random piecewise-constant pacing profiles, best cost 3.2555 ≥ 2.859673 (none beats the analytic profile).

### 3.2 Chemotherapy sequencing

`K = 48` (calibrated to the dose-dense course's delivery), `L0 = 24` (the stop-and-go course's duration), `s(d) = d²`. The three discrete regimens are lifted to dose-rate profiles delivering the *same* cumulative dose K (same efficacy), rates proportional to their discrete suffering levels:

| Course | (∫d², peak d²) lifted | vs continuous optimum (96, 4) |
|---|---|---|
| DD (8 wk) | (288, 36) | dominated; equals continuous optimum at L0=8 |
| STOP_GO (21 wk) | (877824/6561 ≈ 133.7942, 8.7791) | strictly dominated |
| CONT (15 wk) | (7872/49 ≈ 160.6531, 20.8980) | strictly dominated |
| **Continuous** | **(96, 4)** | — |

Independent numeric check: projected gradient on a 480-cell discretization converges to the constant rate (max deviation 8.9e-16, cost 96.0000). Equality-condition falsifier: 1000 random feasible two-level profiles all cost > 96 (min observed 96.0031). Anti-Goodhart: the unconstrained raw minimum is `d ≡ 0` (cost 0, no cure); the efficacy constraint keeps it infeasible.

Frontier continuum (V5), closed form vs numeric solver: `L0=8 → (288,36)`, `12 → (192,16)`, `24 → (96,4)`, `48 → (48,1)`.

### 3.3 Vancomycin TDM

Field values reused from the twin (`s_pre = 0.675679`, `s_ss = 0.059420`); synthetic `τ = 12`, `L0 = 48`; discrete dwell = half horizon (unit-edge ratio).

- **Timing (V6).** `t* = 15.6787` solves `e^{t/12} = 1 + (48−t)/12` (unique root; bisection residual < 1e-12). `J(t*) = 17.907183 < J(24) = 19.644014 < J(0) = J(48) = 32.4326`. Endpoints read clinically: drawing at `t=0` is a useless test (post-draw band = pre-draw band); drawing at `L0` wastes the whole course unmonitored.
- **Infusion (V7).** Using the repo's exact identity `Cmax_ss = Cmin_ss + D/Vc` (§2.3 of the PK-integration spec): synthetic 78.5 kg patient, `Vc = 0.7 L/kg = 54.95 L`, `D = 1000 mg` → swing `D/Vc = 18.1984 mg/L`; trough band `[12,16]` (in-window) → `Cmax_hi = 16 + 18.1984 = 34.1984 > 20`. Supra-window (toxicity) suffering via the rung's `s_window` functional: `(34.1984 − 20)/20 = 0.70992`. Equal-AUC continuous infusion (per-interval `AUC = F·D/CL` is rate-independent) holds a flat in-window level: peak toxicity suffering `0`. The discrete scheduler can only pick bolus regimens.

### 3.4 Machine suffering is decision-relevant (V10)

Bang-bang dosing (`d_max = 8` for `K/d_max` then off — what discrete on/off edges express) vs the smooth optimum, equal efficacy:

| Plan | patient ∫d² | machine (switching strain, dt=0.05) |
|---|---|---|
| bang-bang | 384 | 2560 (two jumps of size 8) |
| constant 2.0 | 96 | 0 |

Total-cost gap `(384 + σ·2560) − 96` is positive at σ=0 and strictly widening in σ (288 → 313.6 → 2848 at σ = 0, 0.01, 1): the expanded ethics is not decorative — weighting machine suffering *more* moves the optimum *further* from bang-bang. At σ=0 the expanded objective recovers the patient-only optimum (consistency). In exposure, the machine-energy of the mercyful pacing profile is `∫κv dx ≈ 1.4298` (V3).

---

## 4. Contract clauses, falsifiers, stop rules

Naming: `V` (validation rung). Verdict marker `MERCYFUL_CONTINUOUS_VERDICT`; gate marker `MERCYFUL_CONTINUOUS_GATE_OK`.

| Clause | Statement | Falsifier | Stop rule |
|---|---|---|---|
| V1 | T1 consistency: per application, continuous optimum ≤ lifted discrete optimum (same objective) | Any inequality reversed | Comparison unsound; harness broken |
| V2 | Exposure quadrature: discrete 7 vs exact 7/3; totals 12 vs 7.3333 at μ=1 | Numbers differ; or discrete ≤ exact | Baseline mismeasured |
| V3 | Mercyful pacing: 10/3 → ≈2.8597; AM–GM bound 2.859668; 20 000 random profiles none better; T ≤ L0; peak 5 | Bound violated; random profile beats analytic; peak changed | Pacing theorem wrong |
| V4 | Chemo: (96,4) strictly dominates all lifted regimens; CS exact; numeric solver agrees (dev < 1e-2); zero-dose trap blocked | Any regimen not dominated; solver disagrees; trap feasible | Dominance claim false |
| V5 | Frontier continuum: closed form matches numeric at L0 ∈ {8,12,24,48}; DD == continuous at L0=8; STOP_GO dominated at 24 | Mismatch at any budget | Frontier claim false |
| V6 | TDM timing: unique interior t* ≈ 15.68, J(t*) < J(24) < endpoints | Root non-unique; J(t*) ≥ J(24) | Timing model degenerate |
| V7 | Infusion: equal AUC, swing D/Vc = 18.1984, s_tox 0.709918 → 0 | Swing identity broken; infusion peak > 0 | Identity misused |
| V8 | Off-node target x*=0.9 solved (cost 2.6964, T ≤ L0); inexpressible discretely | 0.9 ∈ node set; no feasible continuous plan | Generality claim false |
| V9 | Efficiency: exact 3^(k−1) counts (fixed endpoints, k hops); discrete growth ≥ 4× per +2 hops (exact 9); continuous ≤ 8× per 4× grid; native 16-state cap breached at k=5 (18 states) | Counts wrong; growth inverted | Scaling claim false |
| V10 | Machine term: bang-bang (384, 2560) vs smooth (96, 0); gap widening in σ; σ=0 recovers patient-only optimum | Gap non-positive or non-monotone | Ethics term decorative |

Global verdicts: failure of V1, V2, or V4 is **V_RED** (the core comparison is unsound). Failure of V3, V5–V10 is **V_AMBER** (fix the specific clause before claiming). Current status: **V_GREEN (10/10)**.

---

## 5. What the upgraded algorithm buys, in one sentence each

1. **Better accounting.** The discrete scheduler's cost is a left-endpoint Riemann sum of the true field; on the exposure instance it over-bills the descent from peak by 3× (7 vs 7/3).
2. **Better plans.** Optimal titration (V3), constant-rate delivery (V4), optimal measurement timing (V6), infusion (V7) — each strictly dominates what the graph can express.
3. **More problems.** Continuum budgets (V5) and off-node targets (V8) are solvable only continuously.
4. **Better scaling.** Exponential path enumeration → polynomial collocation (V9); the native discrete scheduler cannot represent k≥6 layered instances at all.
5. **The expanded ethics, operational.** The `σ‖u‖²` machine-suffering term is active, monotone, and consistency-preserving (V10): mercy to the substrate *is* smoothness of control.

---

## 6. Limitations

- Synthetic everything: fields, windows, time constants, and the `s(d)=d²` toxicity shape are declared constructions, not measurements. The vancomycin field values come from a synthetic twin.
- The chemo lift (rates ∝ discrete suffering levels, rescaled to equal delivered dose) is a modeling bridge; the dominance theorem (T1/T2) holds for *any* lift that keeps the regimens' time patterns, because Cauchy–Schwarz is lift-independent — but the specific lifted numbers depend on it.
- The exposure strain term `κv` has the dual patient/substrate reading; the contract does not adjudicate which, and the mathematics does not need it to.
- The continuous solver used for cross-checks is a first-order projected gradient, adequate because all optima here have closed forms; no claim about solver quality in general.
- V9's measured timings are machine-dependent; the non-flaky core (exact path counts, 16-state cap) is combinatorial, and the timing margins are generous (≥4× vs exact 9×; ≤8× vs measured ~3.9×).
- No Sounio-native implementation of the continuous algorithm yet (Python reference only); the discrete baseline contracts (M_GREEN, H_GREEN) are re-run by the gate to anchor the comparison.

---

## 7. Reproducibility

```bash
python3 scripts/research/mercyful_continuous_control_contract.py   # V1..V10, verdict V_GREEN
bash scripts/ci/mercyful_continuous_control_gate.sh                # MERCYFUL_CONTINUOUS_GATE_OK
```

The gate additionally re-runs the discrete baselines (`mercyful_runtime_contract.py` M_GREEN, `mercyful_chemo_contract.py` H_GREEN) so the comparison anchor cannot silently rot.

---

## References

1. `docs/research/mercyful-learning.md` — foundational proposal (suffering as measure; geodesic objective).
2. `docs/research/mercyful_runtime_spec_2026-07-25.md` — discrete scheduler (M_GREEN).
3. `docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md` — chemo benchmark (H_GREEN); clinical structure references therein (Bonadonna 1995, CALGB 9741, OPTIMOX1).
4. `docs/research/mercyful_clinical_integration_spec_2026-07-25.md` — PK twins, the `Cmax − Cmin = D/Vc` identity, Knightian bands.
5. `docs/research/mimic_iv_mercyful_validation_2026-07-26.md` — vancomycin field values (s_pre, s_post).
6. Pontryagin et al., *The Mathematical Theory of Optimal Processes* (standard continuous optimal control; no novelty claimed here).

**Clinical warning.** This document specifies a synthetic validation for research infrastructure. It is not medical guidance, not a treatment recommendation, and not a clinical decision-support tool. All patients, doses, regimens, toxicity grades, and suffering values are synthetic; cited literature shapes model structure only and no clinical target claim is made.

## AI disclosure

Spec and harness drafted under human direction (2026-07-26). Math-review offload per `.claude/AGENT_OFFLOAD_POLICY.md` (see `.claude/llm_offload_log.md`). No clinical or patient-level claim. GAIDeT-ICMJE 2025.
