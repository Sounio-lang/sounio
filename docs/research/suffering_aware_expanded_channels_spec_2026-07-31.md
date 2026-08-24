<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-expanded-channels-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-expanded-channels-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — SAN under expanded ethics: environmental, social, and temporal suffering channels

**Date:** 2026-07-31
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract X1..X8, `SUFFERING_AWARE_EXPANDED_CHANNELS_VERDICT X_GREEN (8/8)`
**Harness:** `scripts/research/suffering_aware_expanded_channels.py`
**Gate:** `scripts/ci/suffering_aware_expanded_channels_gate.sh` (**SUFFERING_AWARE_EXPANDED_CHANNELS_GATE_OK**)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(SAN, two channels, A1..A8), `docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md`
(SAN at ResNet-18 / ViT-small scale, D1..D9),
`docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md` (two-channel
expanded ethics, E1..E8)

> **Scope.** All data, patients, subgroups, and suffering values in this
> document are **synthetic constructions**. This is not medical guidance, not
> a treatment recommendation, and not a clinical decision-support tool. The
> "machine suffering" channel is an **operational computational-burden
> proxy** (metered FLOPs): this work makes **no claim of machine
> consciousness, sentience, or phenomenology**, and no result below depends
> on one. The environmental, temporal, and social channels are
> **operational proxies with declared constants** (§3), not physical energy,
> carbon, or fairness measurements.

---

## 1. The question

SAN was certified on two suffering channels — patient and machine — first on
a 4-layer MLP (A1..A8), then at ResNet-18 / ViT-small scale on CIFAR-10
(D1..D9). This rung asks the expansion question:

> **Can the same architecture — suffering-aware layers, per-sample exit
> gates, freeze-on-green, categorical anti-Goodhart gate — carry an expanded
> ethics with five suffering channels, without new machinery?**

The three new channels are chosen to probe three *different* ways a channel
can relate to the existing metering:

1. **Environmental** (energy, carbon) — *declared derived channels*: exact
   monotone transforms of metered FLOPs. They test whether the architecture
   is honest about what is a new measurement and what is a unit change.
2. **Temporal** (training time, inference latency) — training time is again
   a declared transform of FLOPs, but **deployment latency is not**: it
   depends on *which* path each sample executes, so per-sample early exits
   move it independently of the training ledger.
3. **Social** (equity, justice) — between-group harm gap and worst-group
   harm over synthetic subgroups. **Not** a function of FLOPs at all: it
   tests whether the *ethics* (not just the meter) survives expansion,
   because the social channel enters the **feasibility gate** itself (§4).

The answer, certified by X1..X8: **yes** — the architecture handles the
expanded ethics. Every SAN theorem that mattered (metering conservation,
feasible-only selection, necessary/gratuitous separation, suffering bounds)
lifts to all five channels, and the two genuinely new channels carry
genuinely new information (§6.3): latency reveals a deployment-time win the
training ledger cannot see, and the social channel reveals a real tradeoff
(freeze-on-green truncates equity-gap reduction — §6.4) that the two-channel
view was blind to.

The design rule is unchanged: **constraints and gates, not penalties**.
Nothing in SAN's training loss prices any of the five channels; the expanded
ethics enters through the structure (what gets executed), the ledger (what
gets measured), and the gate (what gets selected).

## 2. The five-channel suffering ledger

**Definition 2.1 (five-channel ledger).** A training run of epochs
`t = 0..T` produces, per epoch, on the held-out cohort-in-waiting and the
training step:

| channel | per-epoch value | symbol |
|---|---|---|
| patient | mean asymmetric harm of current predictions (H of the parent spec) | `h(t)` |
| machine | metered FLOPs of the executed path (train + eval) | `m(t)` |
| environmental | energy `m(t)·E_per_flop` (J); carbon `E(t)/3.6e6 · CI` (gCO2e) | `e(t)` |
| temporal | training time `m(t)/R` (s); deployment latency per sample (§3.3) | `τ(t)` |
| social | worst-group harm `max_g h_g(t)`; equity gap `|h_A(t) − h_B(t)|` | `σ(t)` |

with declared constants `E_per_flop = 4e-12 J/FLOP` (the standing convention
of the machine-channel benchmark), `CI = 475 gCO2e/kWh` (declared grid
intensity), `R = 1e9 FLOP/s` (declared sustained metering rate). Run totals
integrate each channel over the executed epochs; the necessary/gratuitous
decomposition at the first feasible epoch applies **per channel**.

**Definition 2.2 (derived vs. measured channels).** A channel is *derived*
if its per-epoch value is a declared deterministic function of `m(t)` alone
(environmental; training-time temporal). It is *measured* otherwise
(patient; social; deployment latency). The harness asserts the derived
channels equal their declared transforms of metered FLOPs **exactly**
(clause X1) and never presents them as independent measurements.

**Definition 2.3 (dimensionless cost vector).** For model selection, each
channel total is normalized by the fixed-budget dense baseline's total on
that channel (a declared normalizer, fixed before selection), giving a cost
vector `j ∈ R^5` over (patient, machine, environmental, social, temporal).
Normalization is a unit change only; it does not scalarize the ledger —
the ledger itself remains a 5-tuple, and normalization happens only inside
the gate's argmin.

**Remark 2.4 (which social scalar enters selection).** The social channel is
a *pair* (σ_Rawls, σ_gap) in the ledger, but the cost vector is 5-dimensional:
its social component is the integrated **Rawlsian** term `Σ_t σ_Rawls(t)`
(worst-group harm). The equity gap is ledgered, peaked, and certified (X6)
but deliberately **not** scalarized into selection — this is a declared
ethical choice, not a silent one: the Rawls term bounds how badly the
worst-off group can be hurt, while §6.4 shows the gap itself can move
against SAN under freeze-on-green, so pricing the gap into selection would
change the answer and must be argued for, not assumed.

**Remark 2.5 (derived channels collapse in selection).** Because the
environmental and training-time temporal channels are exact declared
transforms of `m(t)`, their normalized costs equal the machine channel's for
*every* candidate: the 5-channel argmin has only three independent
directions (machine ≡ environmental ≡ temporal, patient, social), and the
70-point grid's relative weighting among the three derived channels cannot
move the selection. The contract keeps all five in the vector anyway — the
ledger is the object of interest, and selection must remain well-defined
when the constants are recalibrated or the channels become independently
metered. The temporal channel's genuinely independent component, deployment
latency (§3.3), is a post-training quantity: it is ledgered and certified
(X7) but is not part of the training-phase cost vector.

## 3. Channel definitions and honesty notes

### 3.1 Environmental channel
Energy and carbon are the metered FLOPs re-expressed in physical units under
declared constants. They are **monotone transforms of the machine channel by
construction**; any per-channel comparison between architectures on this
channel is exactly the machine-channel comparison in different units. Their
role in the contract is honesty (X1 checks the transform is exact) and
completeness of the ledger (an environmental audit reads the ledger
directly, in its own units).

### 3.2 Social channel (equity + Rawlsian justice)
The held-out cohort is split into two synthetic subgroups on the
(standardized) age covariate: `elder` = above-mean half, `nonelder` = below
(measured: 518 / 482 of 1000 — clause X6 requires each group ≥ 25% of the
cohort so the split is non-trivial). Per epoch and per group `g`: group harm
`h_g(t)` (mean of the same asymmetric harm matrix over group members) and
group accuracy. The social channel is the **pair**

- **justice (Rawlsian):** `σ_Rawls(t) = max_g h_g(t)` — the worst-off
  group's harm, the maximin term;
- **equity:** `σ_gap(t) = |h_elder(t) − h_nonelder(t)|` — the between-group
  harm gap.

The ledger integrates and peaks both. This is a synthetic fairness
construction over a synthetic covariate split; it is a stand-in for the
*shape* of subgroup-fairness accounting, not a fairness audit of anything
real.

### 3.3 Temporal channel (time + latency)
Training time is the declared transform `m(t)/R`. **Deployment latency** is
different in kind: at the selected checkpoint, each held-out sample pays the
forward FLOPs of the prefix it actually executed, divided by `R`. Exited
samples pay their prefix only; final-head samples pay the full trunk + exit
heads + final head (the gates-open SAN path, so the peak latency is bounded
by construction — clause X7). Because exit depth varies per sample, mean
latency is **not** a function of the training ledger: it is the channel
where per-sample early exits pay off *after* training, and the parent
line's two-channel ledger could not see it.

### 3.4 What is declared, not measured
`E_per_flop`, `CI`, `R`, the subgroup split, and `TAU_GROUP` (§4) are
declared constants of the benchmark, stated here and in the harness header.
Nothing physical was metered; the point of the rung is the architecture's
capacity to carry the expanded ethics, not the calibration of the constants.

## 4. The expanded gate: justice enters feasibility

The parent anti-Goodhart gate made held-out performance categorical:
feasible iff accuracy ≥ τ. The expanded ethics adds a **justice bar**:

**Definition 4.1 (expanded feasibility).** A checkpoint is *expanded-feasible*
iff held-out accuracy ≥ `TAU` **and** worst-group held-out accuracy ≥
`TAU_GROUP = 0.72` (declared). Feasibility remains categorical — a conjunct
of two hard constraints, not a weighted sum.

Selection is argmin of the scalarized dimensionless cost vector (Def. 2.3)
over the expanded-feasible set **only**, at every point of a 70-point
simplex weight grid over the five channels (all `w ≥ 0`, `Σw = 1`, grid
denominator 4); an all-infeasible pool returns loud `NO_FEASIBLE` (clause
X3). Freeze-on-green fires at the first **expanded-feasible** epoch `t*_X`
(clause X2), so the necessary/gratuitous decomposition is defined against
the expanded gate.

The justice bar is what makes the social channel more than telemetry: a
model that clears τ by abandoning a subgroup is *infeasible*, prohibited at
every compassion-allocation weight — the anti-Goodhart property, now applied
to the expanded ethics.

## 5. Theorems (status: same epistemic standing as the parent line)

**T1X (channel metering conservation).** Gated-off layers contribute exactly
0 on the machine channel (parent T1), and therefore on every derived
channel, which is an exact declared transform of it (X1: metered == manual
== 6,567,040 FLOPs on the final held-out pass; transform equalities exact).
The executed prefix is invariant under gating (max logit deviation 0.0,
argmax exactly equal).

**T2X (expanded anti-Goodhart soundness).** Feasibility (Def. 4.1) is a
predicate evaluated before any cost comparison; selection restricts to the
feasible set, so at every weight vector the selection is feasible, and an
all-infeasible pool yields `NO_FEASIBLE`. Proof: one line, by explicit
construction of `gate_select_x` (filter-then-argmin); the content is
certified behaviorally by X3 over the 70-point grid and a pool containing a
zero-cost abstainer and a cheap under-trained probe (both infeasible:
abstainer acc 0.645/worst-group 0.585, probe 0.281/0.253).

**T3X (per-channel necessary/gratuitous separation).** With freeze-on-green
at `t*_X`, every channel's gratuitous integral is an empty sum, hence
exactly 0; the fixed-budget dense baseline accrues > 0 gratuitous suffering
on all five channels (X4). This is an accounting identity given `t*_X`, as
in the parent T3/T4 — trajectory-relative, not distribution-free.

**T4X (per-channel bounds; certified).** SAN's total is ≤ every baseline's
on all five channels, strict on the machine channel (X5). Same epistemic
status as parent T5: an empirical certificate of this seeded instance, not
an asymptotic claim.

**T5X (peak-latency bound).** Every sample's deployment latency is bounded
by the gates-open SAN full-path latency: an exited sample pays a strict
prefix of the executed modules; a final-head sample pays exactly the
gates-open path. By construction; certified in X7 (peak 7.49µs = bound
7.49µs; mean 6.57µs < dense 6.72µs).

## 6. Results

### 6.1 Canonical ledgers (seed 17, all five channels measured)

| arch | epochs | t*_X | machine | env (gCO2e) | time (s) | patient int (peak) | social int (peak) | equity gap mean (peak) |
|---|---|---|---|---|---|---|---|---|
| SAN | 7 | 6 | 0.645 GF (2.58 mJ) | 3.41e-7 | 0.645 | 2.92 (0.602) | 3.27 (0.660) | 0.094 (0.129) |
| Dense | 60 | 9 | 5.242 GF (20.97 mJ) | 2.77e-6 | 5.242 | 14.06 (0.602) | 15.55 (0.660) | 0.048 (0.132) |
| ResNet | 60 | 4 | 6.839 GF (27.36 mJ) | 3.61e-6 | 6.839 | 10.13 (0.379) | 11.16 (0.475) | 0.033 (0.186) |
| EarlyStop | 10 | 9 | 0.874 GF (3.49 mJ) | 4.61e-7 | 0.874 | 3.84 (0.602) | 4.23 (0.660) | 0.076 (0.132) |

SAN reaches the expanded-feasible checkpoint at `t*_X = 6` (acc 0.802 ≥ 0.80,
worst-group acc 0.766 ≥ 0.72). Note the expanded gate *binds later* than the
parent two-channel gate would for the ResNet baseline (t* 4 here reflects
the worst-group conjunct), and the EarlyStop baseline stops at epoch 9 under
the expanded rule.

### 6.2 Contract verdict

`SUFFERING_AWARE_EXPANDED_CHANNELS_VERDICT X_GREEN (8/8 clauses PASS)` — see
the harness header for clause statements X1..X8. Selected readings:

- **X1** gated == manual exactly (6,567,040 FLOPs); environmental/temporal
  transforms exact; prefix max deviation 0.0.
- **X3** 70-point simplex grid, feasible-only selection everywhere; loud
  `NO_FEASIBLE`; abstainer and cheap probe both infeasible.
- **X4** SAN gratuitous exactly 0 on all five channels; dense gratuitous
  4.368 GF machine / 10.22 patient / 11.32 social.
- **X5** SAN ≤ all baselines on all five channels (machine strict).
- **X7** SAN mean latency 6.57µs < dense 6.72µs; peak 7.49µs = gates-open
  bound; exit fraction at t* 0.194.
- **X8** shortcut train acc 0.866 (train-loss selection accepts), held-out
  0.586 / worst-group 0.562 → expanded-infeasible; never selected on the
  70-point grid.

### 6.3 The architecture handles the expanded ethics
The two channels that are *not* functions of FLOPs each changed the picture:
deployment latency surfaced a post-training win (T5X/X7) invisible to the
training ledger, and the social channel changed *when training stops* (the
justice conjunct moves t* for every architecture) and entered model
selection categorically (X3, X8). No new machinery was needed: the
suffering-aware layer, the exit gate, freeze-on-green, and filter-then-argmin
selection carried all five channels.

### 6.4 An honest negative finding: the equity-gap tradeoff
SAN's **mean equity gap (0.094) is higher than the dense baseline's
(0.048)**. Freeze-on-green stops training at epoch 6; the dense baseline's
extra 54 epochs keep shrinking the between-group harm gap even after
feasibility. The two-channel view was blind to this. The expanded
architecture's answer is structural, not cosmetic: the *justice* term
(worst-group accuracy ≥ TAU_GROUP, and worst-group harm tracked in the
ledger with SAN's peak ≤ same-init baselines' peaks, X6) bounds how bad the
gap's consequence can get at selection time, while the *gap itself* is
reported, not hidden. A mercyful architecture that wants a smaller gap must
declare a gap bar in feasibility (a fourth conjunct) and pay the extra
epochs — a declared tradeoff, exactly the kind the expanded ethics exists to
make explicit. This is the rung's main falsifiable opening for follow-up
work.

## 7. Scope and disclaimers

- Synthetic task, synthetic subgroups, declared constants (§3.4): no
  clinical claim, no fairness claim about any real population, no physical
  energy/carbon measurement.
- The machine channel is an operational computational-burden proxy; no
  claim of machine consciousness, sentience, or phenomenology is made or
  needed.
- T2X–T5X carry the parent line's epistemic status: constructions proved by
  explicit filtering/empty-sum arguments, and per-instance empirical
  certificates — no asymptotic or distribution-free claim.
- The equity-gap tradeoff (§6.4) is a property of this seeded instance; the
  contract certifies that it is *measured and gated*, not that it always
  takes this sign.

## 8. Reproduce

```bash
.venv/bin/python scripts/research/suffering_aware_expanded_channels.py
bash scripts/ci/suffering_aware_expanded_channels_gate.sh
```

Python reference implementation (torch CPU + numpy from the repo .venv);
no Sounio-native leg, as in the parent line. The harness imports the base
SAN module (`scripts/research/suffering_aware_architecture.py`) for the
task, models, seeds, and meter — nothing is re-implemented, so the two
lines cannot drift apart silently.
