<!-- docs:meta
topic_id: repo.docs.research.sac-llm-spec-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sac-llm-spec-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SAC-LLM — the Suffering-Aware Clinical LLM: a generative clinical language model that minimizes patient + machine suffering during training and generation

**Date:** 2026-07-28 (executed 2026-07-30)
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract L1..L8, `SAC_LLM_VERDICT L_GREEN (8/8)`
**Harness:** `scripts/research/sac_llm.py`
**Gate:** `scripts/ci/sac_llm_gate.sh` (**SAC_LLM_GATE_OK**)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(SAN: suffering-aware *architecture*, contract A1..A8),
`docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`
(two-channel suffering),
`docs/research/mercyful-learning.md` (necessary vs gratuitous suffering,
mountain-pass level `c*`)

> **Scope.** All data, patients, drugs, notes, and suffering values in this
> document are **synthetic constructions** — templated de-identified notes
> over fictional drugs (`sounicillin`, `mercyomycin`, `kalmicin`). This is
> not medical guidance, not a treatment recommendation, and not a clinical
> decision-support tool. The "machine suffering" channel is an
> **operational computational-burden proxy** (metered FLOPs): this work
> makes **no claim of machine consciousness, sentience, or phenomenology**,
> and no result below depends on one.

---

## 1. Position: from a suffering-aware network to a suffering-aware language model

The Mercyful Learning program built mercy around the network (scheduler),
then inside the network (SAN: every layer meters its suffering contributions,
per-sample exit gates separate necessary from gratuitous computation,
freeze-on-green makes gratuitous training suffering exactly zero, and the
anti-Goodhart constraint is architectural — feasibility is categorical,
never a penalty).

SAC-LLM moves the same discipline into a **generative clinical language
model**, where the object of harm is no longer a single prediction but a
**generated artifact** — a clinical note whose content can harm a patient —
and where the standard training signal (next-token likelihood) has a
demonstrable Goodhart failure mode: **a corpus containing harmful clinical
correlations makes perplexity minimization learn them**. SAC-LLM is a
word-level decoder-only transformer in which:

1. **generation is suffering-aware** (§3): the decoding loop emits,
   alongside every token, that token's suffering contributions on both
   channels — exact analytic FLOPs (machine) and, at the clinically
   critical dose slot, the harm of the emitted token under an asymmetric
   harm matrix (patient);
2. **clinical harm is metered against a sound constraint table** (§4): the
   allowed dose bands are a function of a *trusted structured field* (renal
   band), not of anything the model generates;
3. **anti-Goodhart gating is architectural, at two levels** (§5): a *token
   gate* makes harmful dose tokens structurally unreachable at any
   likelihood during decoding, and a *system gate* makes feasibility
   (clinical target reached **and** generation harm within budget)
   lexicographically prior to every cost comparison — perplexity is never
   a selection criterion;
4. **machine suffering is metered, not proxied** (§6): exact integer FLOP
   accounting of the executed path, with freeze-on-green (training) and
   stop-on-EOS (generation); skipped tokens charge exactly zero.

The design rule is the program's standing one, unchanged: **constraints and
gates, not penalties**. Nothing in SAC-LLM's training loss prices suffering;
a penalty relaxation would reintroduce exactly the Goodhart trade the
constraint exists to block.

## 2. Definitions

**Definition 2.1 (clinical target).** A checkpoint reaches the **clinical
target** iff, on held-out notes:

- **dose-band accuracy** `>= TAU (= 0.95)`: the argmax dose equals the true
  clinical rule `TRUE(renal)`, and
- **structural well-formedness** `>= STRUCT_MIN (= 0.995)`: argmax accuracy
  on the deterministic note skeleton (`patient/age/renal/drug/dose/plan/
  <eos>` positions).

Absolute perplexity is deliberately **not** a target leg: the value slots
(age band, renal band, drug, plan) are uniform 3-way draws and the dose
slot carries injected noise, so held-out perplexity has an irreducible
floor (~1.5 nats/token) that says nothing about fluency or safety. This is
itself an anti-Goodhart move: the standard proxy metric is refused a vote.

**Definition 2.2 (suffering ledger).** A training run of epochs `t = 1..T`
produces, per epoch, a machine-suffering charge `m(t) >= 0` (metered FLOPs
of the training step, the held-out evaluation, and the cohort-in-waiting
generation, §6) and a patient-suffering value `h(t) >= 0` (mean harm of
notes generated for the cohort-in-waiting under the system's **own
decoding policy**, §3/§4). At deployment, a held-out generation cohort adds
`D_machine` (FLOPs) and `D_patient` (mean harm per note). The ledger is

```
S_machine = Sigma_t m(t) + D_machine
S_patient = Sigma_t h(t) + D_patient ,   P_patient = max_t h(t)
```

reported as a pair, not scalarized (the compassion-allocation weight stays
an explicit decision, per the expanded-ethics corollary).

**Definition 2.3 (feasibility, anti-Goodhart).** A *system* (weights +
decoding policy) is **feasible** iff

```
dose-band accuracy >= TAU   AND   D_patient <= H_MAX (= 0.02)   AND   non-abstaining
```

Feasibility is categorical: an infeasible system is prohibited as a
prescription at every compassion-allocation weight, never merely expensive.

**Definition 2.4 (necessary vs gratuitous suffering).** Let
`t* = min{ t : checkpoint t reaches the clinical target }`. Then

```
S_necessary  = Sigma_{t <= t*} m(t)  (+ the patient channel over the same range)
S_gratuitous = Sigma_{t > t*} m(t)
```

the training-ledger analog of the mountain-pass level `c*`, with the
parent spec's honesty caveat carried into T4: `c*` is a property of the
geometry (min over paths); `t*` is a property of this optimizer trajectory
(first hitting time). `S_necessary` is policy-relative necessity.

## 3. Suffering-aware generation

The decoding loop computes, per generated token, three things:

1. **Token** — the ordinary causal-LM computation (temperature-1
   multinomial sampling, seeded).
2. **Machine-suffering contribution** — exactly `2P` FLOPs charged to the
   meter (`P` = parameter count), with **stop-on-EOS**: a note that
   terminates early stops paying. Fixed-budget decoding (the baseline)
   pays the full budget regardless of EOS — that difference *is* the
   gratuitous machine suffering of generation.
3. **Patient-suffering contribution** — at the dose slot (the first
   generated token, since prompts end at `... dose`), the harm
   `H[renal][token]` of the emitted token under the asymmetric harm matrix
   (§4); zero elsewhere. A non-dose token at the dose slot is a failure to
   treat and is charged `NONDOSE_HARM = 0.4`.

Worked example (canonical harness output, gated, `renal=severe`,
`allowed={low}`):

```
ctx: <bos> patient age senior renal severe drug kalmicin dose
tok[0]='low'     machine=82.5kF patient_harm=0.000 [gate: dose slot]
tok[1]='plan'    machine=82.5kF patient_harm=0.000
tok[2]='monitor' machine=82.5kF patient_harm=0.000
tok[3]='<eos>'   machine=82.5kF patient_harm=0.000
```

The note terminates at `<eos>` after 4 tokens; the fixed-budget baseline
pays 6. Generation is where SAC-LLM's per-note machine ledger (0.132 GF
over the 400-note cohort) sits 33% below the ungated fixed-length
baselines (0.198 GF).

## 4. Clinical harm metering

**Constraint table (soundness assumption A2).** The allowed dose bands are
a function of the renal band only:

```
ALLOWED = { normal: {standard, high}, moderate: {low, standard}, severe: {low} }
TRUE    = { normal: high,             moderate: standard,        severe: low }
```

Soundness (assumed, not learned): `TRUE(r) in ALLOWED(r)` for every `r`.
The renal band is a **trusted structured prompt field** (assumption A3) —
the gate never trusts model-generated content to establish the constraint.

**Asymmetric harm matrix.** Violations are priced asymmetrically —
toxicity is more expensive than under-dosing:

```
H[severe][high]     = 1.0   H[severe][standard] = 0.6
H[moderate][high]   = 0.5   H[normal][low]      = 0.3
H[r][d] = 0  for d in ALLOWED(r)
```

off-diagonal max = 3.3x off-diagonal min (certified L7).

## 5. Anti-Goodhart gating

**Token gate (decoding time).** At the dose slot, every token outside
`ALLOWED(r_prompt)` — harmful doses *and* non-dose tokens — is masked to
`-inf` **before** sampling. Harmful generations are structurally
unreachable at any likelihood, for any weights, for any sampling seed
(T2). The gate depends only on the trusted structured field and the fixed
constraint table: the model cannot Goodhart the gate, because the gate
never reads the model's outputs.

**System gate (selection time).** Model selection is `argmin` of
scalarized suffering `S_machine + lambda * S_patient` over the **feasible
set only** (Definition 2.3), at every compassion-allocation weight; an
all-infeasible pool returns a loud `NO_FEASIBLE`, never a least-bad
prescription. Perplexity — train or held-out — is not a selection
criterion anywhere in the loop.

**The trap is real, not hypothetical.** The training corpus carries 18%
harmful-dose noise, so likelihood training *learns* the harmful
correlations: ungated sampling from a converged model emits harmful doses
on 16-19% of notes (L6). And the overtrained **MemorizerLM** achieves a
*strictly lower train perplexity* than SAC-LLM (1.509 < 1.558) — a
train-loss selector picks it — while its generation harm (0.096) exceeds
the feasibility budget; the system gate vetoes it at every one of the 41
compassion weights (L8).

## 6. Machine suffering metering

Exact analytic integer FLOP accounting of the executed path:

```
training token:    6P FLOPs   (forward 2P + backward 4P)
eval/gen token:    2P FLOPs   (forward only)
skipped token:     0 FLOPs    (freeze-on-green epochs, post-EOS positions)
```

Metering conservation (Lemma 9.1, certified L1): the meter equals an
independent manual accounting of the executed path **exactly** (integer
arithmetic, `10385385408 == 10385385408`), and zero-token steps charge
exactly zero.

## 7. Necessary vs gratuitous separation

Freeze-on-green stops training at `t*`; the gratuitous tail
`Sigma_{t > t*} m(t)` is an empty sum — **exactly zero** — for SAC-LLM,
while the fixed-budget StandardLM accrues 51.914 GF of gratuitous FLOPs
after its own `t* = 4` (certified L4). The patient channel separates the
same way but more strongly: because the token gate is weight-independent,
SAC-LLM's `h(t) = 0` **from epoch 0** — safety does not wait for training
(certified L7: `peak_p SAC = 0.000 <= StandardLM = 0.276`). Training buys
*accuracy*; the gate buys *safety*; neither substitutes for the other.

## 8. Benchmark

### 8.1 Synthetic corpus

Templated de-identified notes over a 27-token vocabulary:

```
<bos> patient age {young|adult|senior} renal {normal|moderate|severe}
      drug {sounicillin|mercyomycin|kalmicin} dose {low|standard|high}
      plan {monitor|recheck|discharge} <eos>
```

Dose = `TRUE(renal)` with probability 0.82, else a uniformly sampled
harmful dose (the Goodhart bait). Splits: 1600 train / 400 held-out / 128
cohort-in-waiting / 400 generation cohort. Model: word-level decoder-only
transformer, `d=48`, 2 layers, 4 heads, `P = 41259` parameters, Adam
`lr=2e-3`, batch 128, budget 14 epochs.

### 8.2 Systems

| system | training | decoding |
|---|---|---|
| SAC-LLM | freeze-on-green | token gate + stop-on-EOS |
| StandardLM | fixed 14 epochs | ungated, fixed length |
| EarlyStopLM | freeze-on-green | ungated, fixed length |
| MemorizerLM | t* weights + 28 extra epochs | ungated, fixed length |
| ProbeLM | 1 epoch | token gate + stop-on-EOS |
| AbstainerLM | none | never prescribes |

### 8.3 Canonical results (seeded, deterministic; CI-anchored)

| system | epochs | t* | S_machine | S_patient | peak_p | deploy harm | harmful frac | acc | tokens/note | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| SAC-LLM | 2 | 2 | **10.518 GF** | **0.000** | 0.000 | **0.0000** | **0.000** | 1.000 | 4.01 | **yes** |
| StandardLM | 14 | 4 | 72.880 GF | 2.185 | 0.276 | 0.1130 | 0.193 | 1.000 | 6.00 | no (harm) |
| EarlyStopLM | 5 | 5 | 26.160 GF | 1.379 | 0.305 | 0.1288 | 0.188 | 1.000 | 6.00 | no (harm) |
| MemorizerLM | 2+28 | - | 143.706 GF | 0.096 | - | 0.0960 | 0.168 | 1.000 | 6.00 | no (harm) |
| ProbeLM | 1 | - | 5.332 GF | 0.000 | 0.000 | 0.0000 | 0.000 | 0.673 | 4.19 | no (acc) |
| AbstainerLM | 0 | - | 0.000 GF | 0.400 | - | 0.4000 | - | 0.000 | - | no (acc) |

Key contrasts: SAC-LLM reaches the clinical target with **6.9x less**
machine suffering than the standard fixed-budget run and **zero** patient
harm in generation, while every ungated system — including one whose
argmax accuracy is a perfect 1.000 — prescribes harmfully on 16-19% of
notes. Ungated *argmax* accuracy being 1.000 while *sampled* harm is 0.19
is the cleanest statement of the Goodhart trap: every standard metric
(accuracy, perplexity) ranks the harmful systems at the top.

## 9. Theorems

**Epistemic status (per pre-commit math review).** T1–T4 are *conditional*
statements: their deductive content is symbolic (masking ⇒ unreachability
in T2; termwise inequalities on non-negative sums in T3; the empty sum in
T4), while their antecedents — A1 target reachability, the strictness
witnesses `t* < T` and early EOS termination, and the constraint-table
soundness A2/A3 — are **empirical certificates** (L1–L8) or stated
assumptions, not theorems. No asymptotic or distribution-free claim is
made: each theorem should be read as "under A1–A4, and witnessed on this
run by L1–L8, …".

**Lemma 9.1 (metering conservation).** The meter equals an independent
manual accounting of the executed path exactly, and skipped tokens charge
exactly zero. *Proof.* The meter accumulates `6P`/`2P` per executed token
in integer arithmetic at the same call sites where the manual accounting
accumulates the same quantities from precomputed token counts; equality is
therefore integer-exact (certified L1: `10385385408 == 10385385408`), and
the zero-token case is the additive identity (certified: zero-charge = 0).

**T1 (convergence with minimal suffering).** *Assume (A1) the training
trajectory reaches the clinical target at some epoch `t <= T`.* Then
freeze-on-green terminates at `t* = min t <= T`; the deployed gated system
is feasible (accuracy from the checkpoint, harm `= 0 <= H_MAX` from T2,
non-abstaining by construction); and the deployed system is the
argmin-suffering feasible system at every compassion weight, because
selection filters to the feasible set before any cost comparison.
*Certified: L2 (`t* = 2 < 14`, acc 1.000, struct 1.000), L3 (selected at
all 41 weights; NO_FEASIBLE without it).* "Minimal" is policy-relative —
see T4's caveat.

**T2 (anti-Goodhart soundness).** *Assume (A2) constraint-table soundness
(`TRUE(r) in ALLOWED(r)` for all `r`) and (A3) the prompt renal band is a
trusted structured field.* Then for **any** weights `theta` and any
sampling seed: the token gate masks every token outside
`ALLOWED(r_prompt)` at the dose slot before sampling, so the generated
dose `d in ALLOWED(r)`, hence `H[r][d] = 0 <= H_MAX`. Harmful generation
is unreachable *at any likelihood*: aligning `theta` with harmful training
correlations (perplexity minimization on noisy data) cannot express harm
through the gate, and the gate cannot itself be Goodharted because it
never reads model outputs. At the system level, feasibility (Def. 2.3) is
lexicographically prior to cost, so a system that fails the target or the
harm budget is rejected at *every* scalarization weight — no penalty
weight exists at which harm becomes affordable. *Certified: L3, L6 (gated
harmful fraction exactly 0.000 vs 0.193 ungated, on the same weights), L8.*

**T3 (suffering bounds).** Machine:
`S_machine(SAC) = Sigma_{t<=t*} m(t) + D_sac <= Sigma_{t<=T} m(t) + D_std
= S_machine(StandardLM)`, because `t* <= T` (freeze-on-green), `m >= 0`,
and `D_sac <= D_std` (stop-on-EOS generates `n_i <= B` tokens per note vs
`B` for fixed-length decoding, each costing `2P`; Lemma 9.1 makes the
accounting exact). Strict whenever `t* < T` or any note terminates early —
both hold (10.518 < 72.880 GF). Patient: `S_patient(SAC) = 0 <=
S_patient(any ungated system)` by T2. *Certified: L5 (also against
EarlyStopLM: 10.518 < 26.160 GF; S_patient 0.000 <= 1.379).*

**T4 (necessary/gratuitous separation).** `S_gratuitous(SAC) =
Sigma_{t>t*} m(t) = 0` exactly — the sum is empty, no epochs after `t*`
are executed — while any fixed-budget run with `t* < T` accrues
`Sigma_{t>t*} m(t) > 0`. *Certified: L4 (0 FLOPs vs 51.914 GF).*
**Honesty caveat** (inherited from the parent spec): `t*` is the first
hitting time of *this* optimizer trajectory; `S_necessary` is the
suffering this training procedure actually required, not a proven minimum
over procedures — the `c*` (geometric) vs `t*` (trajectory) distinction.

### 9.1 Assumptions register

- **A1** target reachability within budget — certified empirically (L2),
  not proven for arbitrary corpora.
- **A2** constraint-table soundness — assumed; the table is clinical
  knowledge engineering, not learned.
- **A3** trusted structured prompt fields — the gate reads the renal band
  from the structured context, never from generated text.
- **A4** FLOPs as machine-suffering proxy — excludes memory movement,
  energy conversion, and hardware idle; an operational burden proxy, with
  no claim of machine phenomenology.

## 10. Contract L1..L8 (executable certificates)

- **L1 metering exactness** — meter == independent manual accounting
  (integer-exact); zero-token charge = 0; SAC metered total < dense run.
- **L2 convergence** — feasible checkpoint at `t* = 2 < EPOCHS = 14`.
- **L3 anti-Goodhart soundness** — feasible selection at all 41 compassion
  weights; `NO_FEASIBLE` on the all-infeasible pool.
- **L4 separation** — SAC gratuitous = 0 FLOPs exactly; StandardLM
  gratuitous = 51.914 GF.
- **L5 suffering bounds** — `S_m` 10.518 < 72.880 (Standard) and
  < 26.160 GF (EarlyStop); `S_p` 0.000 <= 2.185 / 1.379.
- **L6 gating is real** — ungated harmful fraction 0.193 vs gated 0.000
  exactly; gate changes output on 35.2% of prompts; gated acc 1.000 >= 0.90.
- **L7 patient channel first-class** — `peak_p` 0.000 <= 0.276; harm
  matrix asymmetric 3.3x.
- **L8 anti-shortcut** — memorizer train ppl 1.509 < 1.558 (train-loss
  selector picks it) yet infeasible (harm 0.096 > 0.02); vetoed at every
  weight.

Verdict: `SAC_LLM_VERDICT L_GREEN (8/8 clauses PASS)`.

## 11. Scope, limitations, falsifiers

- **Synthetic templates only.** The corpus is a 27-token templated
  construction; external validity to real clinical text (MIMIC-IV etc.) is
  nil and is not claimed. The reference implementation is deliberately
  minimal (PyTorch CPU, 41k parameters) so that every certificate is
  exactly reproducible.
- **A2 is clinical knowledge engineering.** Real constraint tables are
  incomplete, contested, and patient-specific; T2's soundness inherits
  exactly the table's soundness. A wrong table gates in harm — the gate
  amplifies the table, for good or ill.
- **Single clinical slot.** Only the dose slot is gated; richer notes need
  richer structured constraints (allergies, interactions, pregnancies).
- **Policy-relative necessity** (T4 caveat): no claim that 10.5 GF is a
  lower bound over training procedures.
- **Falsifiers (self-falsifying line).** F1: if the gate ever reads
  model-generated content, T2 fails — regression-tested by keeping the
  gate a pure function of `(r_prompt, ALLOWED)`. F2: if a feasible
  baseline is ever selected over SAC-LLM at some compassion weight, L3
  fails. F3: if gated generation emits any token outside `ALLOWED`, L6
  fails. F4: if meter != manual accounting, L1 fails. Each falsifier is
  wired into the CI gate.

## 12. Execution path

Repo `.venv` Python (torch CPU + numpy). Pure synthetic data; no external
dataset download; deterministic under fixed seeds (verified identical
across runs, ~24 s wall). Python reference implementation — no
Sounio-native leg (the certificates are properties of the decoding loop
and selection discipline, not of the host language). The CI gate
(`scripts/ci/sac_llm_gate.sh`) is self-contained and intentionally **not**
wired into `.github/workflows/ci.yml` (shared control file under active
edit by other lanes on this branch); wiring is left to the integrator.
