<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-game-theory-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-game-theory-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — SAMA-GT: game theory of the Suffering-Aware Multi-Agent system (Nash equilibrium, mechanism design, and the fair division of suffering)

**Date:** 2026-07-31
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract GT1..GT8, `SAMA_GAME_THEORY_VERDICT GT_GREEN (8/8)`
**Harness:** `scripts/research/suffering_aware_game_theory.py`
**Gate:** `scripts/ci/suffering_aware_game_theory_gate.sh` (**SAMA_GAME_THEORY_GATE_OK**)
**Parents:** `docs/research/suffering_aware_multi_agent_spec_2026-07-30.md`
(SAMA: the mechanism — median aggregation, audit, anti-Goodhart gate,
Shapley burden attribution, settlement rule, contract G1..G8),
`docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`
(two-channel suffering, non-scalarized compassion weight μ)

> **Scope.** All data, patients, and suffering values in this document are
> **synthetic constructions**. This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (metered FLOPs): this work makes **no claim of machine consciousness,
> sentience, or phenomenology**, and no result below depends on one.

---

## 1. Position: the mechanism is a game form — what happens when agents play it?

The SAMA spec built the *mechanism*: median aggregation, an exact audit, a
categorical anti-Goodhart gate, Shapley burden attribution, and a settlement
rule. Its theorems T1–T4 are stated against **fixed behavioral types** —
the honest agent trains 5 epochs, the strategic agent trains 1 and claims 5,
the adversarial agent sign-flips. That is the right first step, but it is not
a game-theoretic answer. A strategic agent that *optimizes* would not
necessarily play (1 epoch, claim 5); an adversary that *optimizes* would not
necessarily sign-flip. This document asks the three classical questions of
the induced game, and answers them with certified computations on the pinned
environment:

1. **Equilibrium (Nash).** Does the SAMA round game have equilibria, what do
   they look like, and does equilibrium play preserve what SAMA was built to
   guarantee? Answer: existence always (§3, T-GT1); *safety* properties
   survive equilibrium play (T-GT3, T-GT8); *liveness* does **not** — the
   repeated game has a certified bad equilibrium that never reaches the
   collective target (T-GT5), and we exhibit and certify a constraint-style
   repair that restores it (T-GT6).
2. **Mechanism design.** What does the settlement rule actually implement,
   and in what sense? Answer: truthful *reporting* in strictly dominant
   strategies (T-GT2); *effort* is provably not incentive-aligned in the
   machine channel (T-GT4) — a deliberate consequence of the lineage rule
   "constraints and gates, not penalties," made precise here.
3. **Fair division.** Who bears the suffering, and is the division *fair* in
   an axiomatizable sense? Answer: the patient-harm channel is divided by the
   Shapley value — the **unique** division rule satisfying efficiency,
   symmetry, null-player, and additivity (T-GT7) — and the machine channel is
   divided proportionally, anonymously, and subsidy-free; patients receive a
   Rawlsian maximin guarantee at *every* action profile (T-GT8).

The expanded ethics stays at the center throughout: every payoff comparison
and every price-of-anarchy figure is a **pair** (patient harm, machine
FLOPs); nothing is scalarized, so no compassion weight μ is load-bearing for
any theorem below.

## 2. The formal model

### 2.1 The SAMA game Γ(M)

Fix the pinned SAMA environment (parent spec §2: synthetic dose-band task,
harm matrix `H`, cohort-in-waiting, `TAU = 0.8475`, `ROUNDS = 40`,
`E_LOCAL = 5`, misreport penalty `λ = 2`).

- **Players.** `N = 5` agents with fixed types `θ = (h, h, h, s, a)`: three
  honest, one strategic, one adversarial (Bayesian types are set by the
  environment, not chosen).
- **Actions.** Per round, agent `i` plays `a_i = (e_i, c_i, d_i)`:
  executed epochs `e_i ∈ {0,…,5}`, claimed epochs `c_i ∈ {0,…,5}`, update
  direction `d_i ∈ {gradient, sign-flip(−6×), class-flip}`. The update is
  `Δ_i = T_{d_i}(train_{e_i}(shard_i; w_t))`. The full action space is finite:
  `6 × 6 × 3 = 108` pure round-actions per agent.
- **Behavioral assumption (A1).** Honest agents are protocol-following, not
  optimizing: `a_h = (5, 5, gradient)` always. This is the standard
  Byzantine-fault-tolerance convention and matches the parent spec; honest
  agents are part of the game form, not strategic players.
- **Mechanism M (game form).** Per round: (i) audit — deterministic
  recomputation recovers `e_i` exactly (parent G1); (ii) aggregation —
  coordinate-wise median of the `Δ_i`; (iii) round guard — if the aggregate
  would increase patient harm on the cohort-in-waiting, roll the round back;
  (iv) settlement — machine charge `σ_i = audited_i + λ·|claim_i − audited_i|`
  and harm share `φ_i` (exact Shapley value of the round's harm-change
  coalition function `f`); (v) freeze-on-green — halt at the first round with
  held-out accuracy `≥ τ`.
- **Payoffs.**
  - Strategic agent: `u_s(a) = −σ_s` (its own settlement machine charge; the
    parent spec's type definition — minimizes its own machine suffering).
  - Adversarial agent: maximizes *others'* suffering. To respect
    non-scalarization, its preference is the **Pareto order** over the pair
    `Ψ_a = (Σ_t harm_t, Σ_{t, i≠a} m_i(t))`: outcome `x` is preferred to `y`
    iff `x` weakly dominates `y` in both components and strictly in one.
  - Social choice function (the mechanism's goal): the collective suffering
    pair `(S_machine, S_patient)` integrated to first feasibility, subject to
    the categorical gate (feasibility `≥ τ` reached within `ROUNDS`).
- **Repeated game.** The stage game repeats with perfect recall until
  freeze-on-green or the round budget; the horizon is endogenous but bounded
  by `ROUNDS`, so the extensive form is finite.

### 2.2 Solution concepts

Pure- and mixed-strategy Nash equilibrium (unilateral deviations only);
dominant strategies for the reporting subgame; ex-post properties (hold for
every action profile of the minority); price of anarchy `PoA = (PoA_machine,
PoA_patient)` as a **pair** — worst certified equilibrium suffering over the
cooperative (all-honest) benchmark, componentwise.

### 2.3 Standing assumptions (explicit)

- **A2 (audit exactness).** The audit recovers executed epochs exactly
  (parent G1; verified-computation assumption, deployment note stands).
- **A3 (minority).** Non-honest agents number `k < N/2` (here `k = 2`).
- **A4 (discretization).** Equilibrium *certificates* are over the action
  space of §2.1; theorems stated "for all λ > 0" or "for all k < N/2" are
  proved analytically and do not rely on discretization.
- **A5 (ledger inclusion).** Collective machine suffering counts *every*
  agent's executed FLOPs, including the adversary's (the ledger does not
  moralize; it meters).

## 3. Nash equilibrium

**T-GT1 (existence, and constructive pure equilibria).**
*(a) Existence.* The repeated SAMA game is finite (finite players, finite
pure actions per round, horizon bounded by `ROUNDS`), so a mixed-strategy
Nash equilibrium exists by Nash's theorem (1950); the finite extensive form
with perfect recall moreover has an equilibrium in behavioral strategies
(Kuhn's theorem).
*(b) Constructive pure equilibrium of the repeated game under M.* The profile

```
s* = ( honest×3 , strategic (e=0, c=0, gradient) , adversary (e=5, c=5, sign-flip) )
```

is a pure-strategy Nash equilibrium, certified by exhaustive best-response
enumeration (contract GT1): the strategic agent's charge over all 36
`(e, c)` pairs is uniquely minimized at `(0, 0)` with charge 0, and no
enumerated adversary deviation (3 directions × 3 effort levels) Pareto-dominates
its equilibrium payoff. The equilibrium is a small equivalence class: under M
the adversary's undominated actions are exactly `(sign-flip, e=1)` and
`(sign-flip, e=5)` (its own FLOPs do not enter its payoff); under M+ the
certified best response is the singleton `(class-flip, e=1)`. The
price-of-anarchy certificates below use the worst member of the class
(maximal total collective suffering).
*Honesty note:* (b) is a certificate for the pinned environment and the
discretized action space (A4), not a closed-form characterization of all
equilibria; (a) is the generic existence statement.

**T-GT2 (dominant-strategy truthful reporting).**
*Claim.* For `λ > 0`, given any executed effort `e`, the truthful claim
`c = e` strictly dominates every misreport in the machine charge:
`σ(e, c) − σ(e, e) = λ·|c − e|·F > 0` for `c ≠ e`, where
`F = n_local · TRAIN_FLOPS`. Hence the reporting subgame has a unique
dominant-strategy equilibrium: report truthfully. At `λ = 0` truthfulness is
only weakly dominant (all claims tie).
*Proof.* One line: `σ(e, c) = e·F + λ|c − e|·F` is uniquely minimized over
`c` at `c = e` when `λ > 0`. ∎ Verified for all 36 `(e, c)` pairs at the
pinned `λ = 2` (GT2).
*Remark.* This is the precise content of "misreporters are charged so that
deviation does not pay" (parent §6.3): the audit + penalty makes the *report*
a dominated object. The parent spec's strategic agent (claim 5 over executed
1) is not best-responding; its misreport is exactly the deviation T-GT2
prices out.

**T-GT3 (ex-post minority immunity — safety is equilibrium-robust).**
*Claim.* For every coalition of `k < N/2` non-honest agents and **every**
action they take, each coordinate of the accepted median update lies in the
closed interval `[min honest, max honest]` of that coordinate. Feasibility
gating (parent T2) and the round guard are properties of the accepted update,
not of anyone's incentives, so they hold at *every* action profile —
equilibrium or not.
*Proof.* Order-statistic argument: for `N` numbers of which at most `k` are
adversarial, at most `k` values lie strictly below `min honest` and at most
`k` strictly above `max honest`; since `2k < N`, the `⌈N/2⌉`-th order
statistic — the median — lies in `[min honest, max honest]`; applied
coordinate-wise. ∎ Verified exhaustively: all 10 minority-coalition slot
choices × 25 attack pairs (250 cases), zero violations, exact float
containment (GT3).

## 4. Mechanism design: what is implemented, and what is not

**T-GT4 (incentive scope — reports are aligned, effort is not).**
*Claim (i).* The unique machine-channel best response of a strategic agent is
zero effort, truthfully reported: `(e, c) = (0, 0)`, charge 0. Effort
provision is **not** incentive-compatible in the machine channel — no
settlement of the form `audited + λ|claim − audited|` can make it so,
because the charge is monotone non-decreasing in audited effort.
*Claim (ii).* Abstention is nevertheless **detected**, not merely priced:
under median aggregation with the pinned attack mix, the abstainer's Shapley
harm share is *positive* (`φ = +0.0464`: harm-increasing, hence flagged by
parent G4's rule), while a 1-epoch free-rider (`−1.1658`) and an honest
agent (`−1.1734`) are harm-reducing; the share is strictly monotone in
withheld effort.
*Proof of (i).* `σ(e, c) ≥ e·F ≥ 0 = σ(0, 0)`, both strict unless `e = 0`,
by T-GT2. ∎ (ii) is certified numerically (GT4) — environment-specific, not
axiom-derived, per the parent spec's honesty note on Shapley sign separation.
*Design reading.* This is the game-theoretic content of the lineage rule
"constraints and gates, not penalties": SAMA aligns **reports** by incentives
(T-GT2) and protects **outcomes** by constraints (T-GT3, T-GT8); it
deliberately declines to align **effort** by a scalarized penalty, because
converting harm shares into machine charges would pick a compassion weight μ.
The cost of that abstention is measured, not hidden — see T-GT5.

**T-GT5 (liveness failure at equilibrium — certified negative result).**
*Claim.* Under the pinned mechanism M, the pure equilibrium `s*` of T-GT1(b)
does **not** reach the collective target: the trajectory stalls at held-out
accuracy 0.845 < τ = 0.8475 for all 40 rounds, so `t*` does not exist at
equilibrium. The certified price of anarchy is the pair

```
PoA_machine = 74.880 / 7.020 = 10.67 ,  PoA_patient = 20.240 / 1.442 = 14.04 .
```

*Mechanism (why).* Two features interact. (a) The abstainer's zero update
makes the coordinate-wise median land on the honest value *nearest zero* —
or on zero itself when honest updates straddle zero near the plateau — so
accepted steps are attenuated exactly where the remaining accuracy gap
lives. (b) The wholesale rollback guard is deterministic: a rolled-back
round recomputes the *same* updates from the *same* model, so a single
harm-increasing median step is an absorbing state (observed directly: the
`adv = gradient` variant deadlocks at 0.844 from round 1). The adversary's
trajectory-optimal attacks (sign-flip; or class-flip at low effort, which
also stalls at 19.870 patient harm) exploit (a)+(b); notably, one optimal
attack — playing an honest-direction update at full effort against a
1-epoch free-rider — is **indistinguishable from honesty**, so no
Byzantine-robust aggregation rule can exclude it.
*Safety at the bad equilibrium.* The stall is not a Goodhart failure: the
round guard keeps patient harm non-increasing throughout (T-GT8 holds at
`s*`), the ledger stays exact, and the gates never select an infeasible
checkpoint. What fails is *liveness* (convergence to τ), not *safety*.
*Honesty note.* This shows the parent T1's convergence hypotheses are not
equilibrium-robust: "bad agents are a strict minority" (T1(i)) holds at `s*`,
yet the attenuated median violates the productive-step spirit of T1(ii) near
the plateau. T1 remains correct as stated (it is trajectory-relative); the
game theory shows its hypotheses can fail to be *sustained by equilibrium
play*.

**T-GT6 (repair M+: a constraint-style guard restores equilibrium liveness).**
*The amendment (candidate, not applied to the pinned contract).* Replace the
wholesale rollback with a **harm-descent fallback guard**: try the median
first (preserving the nominal path), then each single-agent update, each on
a halving step grid `α ∈ {1, 1/2, …, 1/16}`; accept the first step that does
not increase patient harm; if none exists, skip the round. This is a
constraint on admissible steps — the lineage's "gates, not penalties"
philosophy — not a new objective term.
*Claim (i), nominal invariance.* On the pinned attack mix and on the
all-honest run, M+ is **byte-identical** to M (median accepted at α = 1 in
every round): `t* = 2`, `S_machine = 5.8968 MF` / `7.020 MF`,
`S_patient = 1.490` / `1.442` — the amendment changes nothing when nothing
is wrong.
*Claim (ii), equilibrium liveness.* At every enumerated profile
(strategic effort ∈ {0, 1} × adversary direction ∈ {gradient, sign-flip,
class-flip}), M+ reaches τ (`t* ≤ 3`). The worst pure equilibrium under M+
is `(abstain, class-flip @ e=1)` with certified price of anarchy

```
PoA_machine = 0.8533 ,  PoA_patient = 1.3911
```

— a machine-*dividend* (the abstainer's unburned FLOPs) and a bounded
patient price, down from `(10.67, 14.04)` under M.
*Claim (iii), safety preserved.* The Rawlsian guarantee (T-GT8) holds under
M+ at every enumerated profile.
*Proof status.* (i)–(iii) are certified by the contract (GT6); the
enumeration covers the strategic agent's undominated efforts and the full
attack family at representative efforts. A closed-form liveness proof for M+
(why a harm-non-increasing productive step always exists in the honest
range) is future work; the certificate is environment-pinned per A4.

## 5. Fair division of suffering

**T-GT7 (Shapley is the unique fair division of patient-harm change).**
*The axioms, in suffering terms.* A division rule `ψ` mapping each round
coalition game `(N, f)` to harm shares `ψ_i` is *fair* iff:
(E) **efficiency** — `Σ_i ψ_i = f(N) − f(∅)`: the round's total harm change
is fully accounted, neither created nor hidden;
(S) **symmetry** — two agents whose updates are interchangeable in `f`
receive equal shares: identical burden for identical causal role;
(N) **null player** — an agent whose update never changes any coalition's
harm bears no harm share: no suffering attributed without causation;
(A) **additivity** — shares compose across rounds: `ψ(f + g) = ψ(f) + ψ(g)`,
so the ledger may be settled round-by-round or in aggregate with the same
result.
*Claim (Shapley 1953).* The Shapley value is the **unique** rule satisfying
(E), (S), (N), (A). SAMA's attribution (parent §6.1) is the Shapley value of
the round harm-change game, hence it is the unique fair division of the
round's patient-suffering change in this sense.
*Certification (GT7).* On the real round-0 coalition function: efficiency to
`0.0e+00`; permutation equivariance (relabeling agents permutes shares
exactly, three permutations, to `1e-12`). On controlled games: additive
games recovered exactly (`φ = w`); a null player (dummy with `v({i}) = 0`)
receives exactly 0;
additivity `φ(g₁ + g₂) = φ(g₁) + φ(g₂)` to `1e-12`; symmetry between
interchangeable players exact.
*Machine channel (proportional, anonymous, subsidy-free).* The machine
settlement divides machine suffering **proportionally to metered work**:
truthful agents pay exactly their own executed FLOPs — equal work, equal
charge (equitable: all honest agents' charges identical, verified); the rule
is anonymous (identity-independent) by construction; and no honest agent
ever subsidizes another's work (charged == metered on the full all-honest
ledger, verified). Misreporting breaks proportionality only against the
misreporter (T-GT2).
*Honesty note.* Fair-division theory also studies envy-freeness; we
deliberately do not claim it. At the T-GT1 equilibrium the abstainer pays 0
machine charge while honest agents pay full freight — honest agents would
envy it in the machine channel under any scalar reading. Anonymity (any
agent could have chosen abstention) is the fairness property SAMA actually
offers there; deterrence of envy-worthy abstention is left to the harm
channel (T-GT4(ii)) and to future settlement amendments, precisely because
scalarizing the pair is out of bounds.

**T-GT8 (Rawlsian patient protection, at every profile).**
*Claim.* For **every** action profile of every agent — equilibrium or not,
under M or M+ — accepted-round patient harm on the cohort-in-waiting is
non-increasing, hence the peak accepted-round harm never exceeds the
do-nothing harm `h(∅) = 4.887` (the untrained model). The worst-off moment
for the patient population is the status quo ante; no play of the game can
make patients worse off than never training at all.
*Proof.* Induction over accepted rounds: the guard (either variant) admits
only steps with `h_{t+1} ≤ h_t`; `h_0 = h(∅)` by construction. ∎ Verified on
all 12 enumerated (profile × guard) combinations plus the pinned attack mix
and the all-honest run (GT8). This is a maximin guarantee for the
worst-off party — the cohort-in-waiting — and it is the strongest sense in
which SAMA's mercy is unconditional: it does not depend on incentives at
all.

## 6. The equilibrium landscape, summarized

| profile (strategic, adversary) | mechanism | t* | S_machine | S_patient | reading |
|---|---|---|---|---|---|
| all-honest (cooperative benchmark) | M / M+ | 2 | 7.020 MF | 1.442 | the optimum |
| spec attack mix (1-epoch+misreport, sign-flip) | M / M+ | 2 | 5.897 MF | 1.490 | parent contract |
| **bad NE** (abstain, sign-flip) | **M** | **none** | **74.880 MF** | **20.240** | T-GT5: liveness fails |
| worst NE (abstain, class-flip e=1) | **M+** | 3 | 5.990 MF | 2.006 | T-GT6: bounded PoA |

Price of anarchy (pair, never scalarized): M: `(10.67, 14.04)`;
M+: `(0.853, 1.391)`.

*Accounting anchor.* Per-epoch per-agent metered FLOPs are
`F = n_local · 3(2·D·C + C) = 800 · 117 = 93 600`; every `S_machine` figure
above is an exact multiple (`7 020 000 = 75 F-epochs`, `5 896 800 = 63`,
`74 880 000 = 800`, `5 990 400 = 64`), and `t* = 2` means
rounds 0–2 were executed before freeze-on-green.

## 7. Contract (executable certificates)

The harness prints and the gate enforces:

- **GT1 equilibrium existence** — strategic BR unique at `(0,0)` over all 36
  `(e,c)`; adversary BR sets certified under M (`sign-flip`, effort class
  {1,5}) and M+ (`class-flip @ 1`); pure NE exhibited for both mechanisms.
- **GT2 DSIC reporting** — `charge(e,e) < charge(e,c)` for all 30 misreport
  pairs at λ = 2.
- **GT3 minority immunity** — 250 (coalition × attack-pair) cases; median
  within honest coordinate range in every case (exact).
- **GT4 incentive scope** — `φ₃(e=0) = +0.0464 > φ₃(e=1) = −1.1658 >
  φ₃(e=5) = −1.1734`; abstainer flagged (φ > 0); efficiency < 1e-9.
- **GT5 liveness failure (negative result)** — bad NE under M: `t* = NONE`,
  final 0.845 < τ, harm curve monotone (safety holds); `PoA_machine ≥
  10`, `PoA_patient ≥ 13` (measured 10.67 / 14.04).
- **GT6 repair M+** — nominal trajectories byte-identical
  (`S_machine = 5 896 800`, `t* = 2`); all 6 grid profiles converge with
  monotone harm; worst-NE `PoA ≤ (1.15, 1.45)` (measured 0.8533 / 1.3911).
- **GT7 fair division** — Shapley axiom suite (efficiency, permutation
  equivariance, additive recovery, dummy-null, additivity, symmetry) plus
  machine-channel equitability and no-subsidy, all exact or < 1e-12.
- **GT8 Rawlsian guarantee** — harm non-increasing and peak ≤ do-nothing
  harm (4.887) at every enumerated profile and both guards.

Anchor: the harness first reproduces the pinned SAMA numbers exactly
(attack mix `t* = 2`, `S_machine = 5 896 800`, `S_patient = 1.490`;
all-honest `7 020 000` / `1.442`) and refuses GT_GREEN if it does not.

## 8. Limitations

- Equilibrium *certificates* are for the pinned 5-agent environment and the
  discretized action space (A4); the analytic theorems (T-GT2, T-GT3, T-GT8)
  are general within their stated hypotheses.
- The bad equilibrium (T-GT5) is a property of mechanism M as pinned; the
  deterministic-rollback deadlock half of its mechanism is
  reference-implementation-specific (any tie-breaking jitter escapes that
  half), while the median-attenuation half is structural.
- M+ is a *candidate amendment* analyzed inside the GT harness only; the
  pinned SAMA harness, spec, and gate are untouched. Adopting M+ is a
  decision for the SAMA lane, with its own contract update.
- Shapley sign-orderings (T-GT4(ii)) are environment-specific, per the
  parent spec's honesty note.
- Honest agents are behavioral (A1); a model in which "honest" agents also
  optimize (e.g., over the pair they are charged) is open.
- All suffering values are synthetic; the machine channel is an operational
  computational-burden proxy with no phenomenological claim.

## 9. Scope guards

Synthetic data only; not medical guidance; no clinical claim; no claim of
machine consciousness, sentience, or phenomenology; the harness prints
`no_consciousness_claim` in every run.
