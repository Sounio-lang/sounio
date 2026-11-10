<!-- docs:meta
topic_id: repo.docs.research.epistemic-reasoning-and-metacognition-2026-10-22
authority: repo_only
audience: researchers
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.epistemic-reasoning-and-metacognition-2026-10-22
-->

# Reasoning as `Knowledge`, functional metacognition, and the interoceptive organ

**Date:** 2026-10-22 · **Author:** Demetrios Chiuratto Agourakis

**Status: research record. Nothing here is implemented, and nothing here is a
claim of novelty.** It records a design line worked out on 2026-10-22, the
prior art found for each piece, what an independent adversarial review
(a separate reviewer told to refute the draft, given the draft but not its
author's reasoning) knocked down, and the kill test that decides each piece
that is still standing. A section is promoted only by running its kill test,
never by being re-argued.

Reading rule used throughout:

- **stands** — survived the review; still a hypothesis until its kill test runs
- **narrowed** — survived in a weaker form, stated as that form
- **dead** — prior art or an argument killed it; kept so nobody re-derives it

---

## 1. `Reasoning<T>`: confidence may not rise by re-reading itself

### 1.1 The observation it starts from

Feeding a model's own reasoning back to it, with no new external signal, rarely
improves the answer and often degrades it (Huang et al., ICLR 2024; Stechly,
Valmeekam & Kambhampati, ICLR 2025: on graph colouring, self-verification took
accuracy from 16% to 2% while a sound external verifier took it to 38%). The
second pass has no information about the truth that the first lacked; it can
only redistribute belief. That is a data-processing-inequality argument, and it
has been written down as one (Zenil et al., arXiv:2601.05280v1; an unreviewed preprint
shows that *k* consecutive self-acceptances carry no more information than one).

### 1.2 The rules (pseudo-signatures, not Sounio syntax)

| rule | content | status |
|---|---|---|
| R1 | `reflect : Reasoning<T> → Reasoning<T>` guarantees `Q' ≤ Q` unless a new provenance edge of class `Oracle` enters | **narrowed** (§1.3) |
| R2 | steps whose provenance is the same generator are `Correlated` with ρ = 1 by default; lowering ρ needs an explicit, recorded `decorrelate(ρ, evidence)` | stands |
| R3 | `verify(o: Oracle)` may raise Q: a Lean proof can reach ESTABLISHED; execution or a test enters as a Type A component; a clean-context second instance enters as Type B with ρ < 1 only if ρ was measured | stands |
| R4 | a reasoning value's validity is bound to the weight hash and version; a different model expires it | stands |
| R5 | divergent reasoning branches join by least upper bound, not by rejection | stands |

R2 is the GUM, not a new axiom: two estimates combined with ρ = 1 reduce no
uncertainty. The compiler already has the shape of it. Reusing the same
`Knowledge` slot in binary arithmetic requires `with Correlated` (`R6 Correlated
(id 29)`, `self-hosted/check/check.sio:8248`), and disjoint noise-symbol sets
are what licence independent quadrature (`check.sio:8221`, E230 anti-garbling).
Self re-reading is the extreme case of slot reuse. R4 has a home in the
`knowledge_validity_*` fields (`check.sio:2406`).

### 1.3 The objection that narrowed R1, and the repair

The *sharpening* hypothesis (Huang et al., arXiv:2412.01951) and later
versions of Zenil et al. concede the point that matters: self-improvement does not create
information about the world, but it can make latent information **more
accessible**. Fixing an arithmetic slip on a re-read is a legitimate gain.

So ρ is not a property of a model. **ρ is a property of the pair (generator
function, verifier function).**

- `reflect` — the same function again ("reason again"): ρ ≈ 1, Q does not rise.
- `check(f)` — a structurally different function on the same model (substitute
  values, test a boundary case, recompute a sum): ρ may be < 1, and Q rises by
  the ρ *measured for that f*.

That keeps the thesis and absorbs the objection without conceding it.

### 1.4 Prior art, and what is left

| piece | ancestor | consequence |
|---|---|---|
| "self-reading creates no evidence" | DPI argument, Zenil et al.; Stechly et al. | not new |
| ρ between judges | "nine frontier judges ≈ two effective votes" (Kish effective sample size against a Condorcet null) | the measurement exists |
| "Q may not rise without an oracle" as a typing rule | Biba integrity / *endorsement* in information-flow control; CaMeL (Debenedetti et al., arXiv:2503.18813) uses the same structure for prompt-injection safety | the rule is IFC with an epistemic label |
| typed provenance through LLM pipelines | CaMeL; its guarantees fail to compose across agents (arXiv:2610.05640) | in this vocabulary: **provenance laundering**, which the type must forbid |
| typed chain-of-thought | Curry–Howard checking of traces (arXiv:2510.01069) | logical correctness, not epistemic grade |
| `Knowledge<T>` itself | `Uncertain<T>` (Bornholt, Mytkowicz & McKinley, ASPLOS 2014) | direct ancestor; no reasoner provenance |
| step-wise uncertainty in agents | SAUP, UProp and relatives | runtime estimators, not static invariants |

**What is left, stated as the modest claim it is:** *epistemic
non-interference* — a typing theorem that no well-typed program can raise Q
along a path that contains no `Oracle` edge, with `Correlated` inferred from
provenance and `decorrelate` as the only, audited, escape. Its formalisation in
Lean (monotonicity of Q under `reflect`, conditional elevation under
`check(f)`) is the next step for this line.

## 2. Functional metacognition

### 2.1 Which metacognition

Two senses are routinely conflated:

1. **monitoring** (Nelson & Narens, 1990): a meta level receives signals from
   the object level through its own causal channel and controls it;
2. **self-interpretation** (Carruthers, 2011): the subject infers its own
   states from the evidence an outside observer would use.

Re-reading a trace is (2). The transformer case inverts the human one: given the
weights, every internal state is a deterministic function of the token prefix,
so on re-reading the model has *access* to everything it had on the first pass;
what it lacks is a trained *function* that routes that content into a report.
Access is not monitoring. The defensible thesis is **functional** metacognition,
with nothing conceded or claimed about phenomenal experience.

### 2.2 Criteria (transposed from comparative cognition; Hampton, PNAS 2001)

1. **privileged access** — self-report predicts own behaviour better than an
   observer with the same data (partial evidence: Binder et al., ICLR 2025);
2. **causality** — intervening on the internal state changes the report
   (real but unreliable: concept-injection detection, Lindsey 2025);
3. **control** — the signal changes conduct: abstain, spend more compute, seek
   verification;
4. **transfer** — the signal generalises, and is not mimicry of confidence
   language seen in training (weak evidence today).

Operational definition carried by the type system: a provenance class
`Introspect`, distinct from re-reading text. Functional metacognition exists
when ρ(Introspect) < ρ(reread) **and** the gap survives intervention (2) and
transfer (4). `Introspect` is a seventh provenance kind; today
`AstProvenanceKind` has six (`Derived`, `Source`, `Computed`, `Literature`,
`Measured`, `Input`; `self-hosted/parser/ast.sio:485`).

### 2.3 What died on 2026-10-22

| proposal | killed by |
|---|---|
| "the model re-reading itself vs another instance reading the same text estimates ρ" as a contribution | it is Binder et al.'s self- vs cross-prediction protocol renamed |
| "numerically undecided token" (argmax changes across round-down / nearest / round-up runs) | CESTAC (Vignes) and CADNA's **unstable branching**: a comparison whose difference is a computational zero. CADNA's test is C ≤ 0 with Student τ₀.₀₅ = 4.303 for N = 3 samples, which in this record's notation is s < log₂(4.303/√3) ≈ 1.31 bits (our conversion, not CADNA's statement); noisefloat (LIP6, 2026) already runs CESTAC in PyTorch and measures top-1 agreement |
| `s(t) = log₂(m/σ) < 1` means "the decision is random" | wrong quantity; under a Gaussian margin the right one is `P_flip = Φ(−m/σ)`, and s < 1 means only P_flip ≳ 2.3% |
| "absent from training data, therefore not mimicry" | non sequitur: a probe involves no self-report, so mimicry never arises; and decoding s(t) from activations is the **null** hypothesis, since later layers compute it from that state |
| a model could represent its own σ | σ depends on reduction order, dtype and hardware, which differed during training; no training signal could install it. The a-priori prediction is null |
| CESTAC in hardware at ~2× | at least 3.5–5× on AVX-512 (two rounded FMAs, a masked blend, mask generation); Intel AMX ignores MXCSR, fixes RNE and forces DAZ/FTZ, so no directed or random rounding is possible there at all |
| the phenomenon at FP32 | FP32 divergence is rare (Yuan et al., NeurIPS 2025, report ~2.2%); it lives in BF16, which has no hardware path for directed rounding |
| "exact formal analogue of aberrant precision in psychosis" | direction inverted: Adams et al. describe *reduced* precision of priors relative to sensory evidence, and Sterzer et al. show the direction is contested. At most a loose analogy |

### 2.4 What stands, as one falsifiable sentence (narrowed)

> In a transformer, the rounding-induced standard deviation σ(t) of the top-2
> margin — measured with per-operation random rounding, fixed reduction order,
> in the model's real dtype — carries per-token information in intermediate
> states **not explained** by margin, entropy, activation norm or logit-lens
> margin, and **causally modulates abstention** when σ is changed without
> changing the margin.

The design that makes this testable is **σ-only manipulation**: change the
arithmetic (reduction order, accumulation dtype) so that σ moves and the margin
does not, with weights and inputs identical. The expected result is null, and a
clean null is a result: it shows the model has no privileged access to its own
arithmetic, which tests the metacognition thesis at its hardest point.

**Minimal experiment, before any compiler work** (C++23, per ADR-009): Qwen2.5-0.5B
in FP32 over 200 prompts; MXCSR switched down/up *per layer*; two directed runs
plus ~30 software random-rounding runs on the final linear layer; (m, σ) per
token.

- **Kill 1:** regress log σ on position, residual norm, entropy and log m. If
  the remainder explains < 5% (partial R² < 0.05), stop.
- **Kill 2:** recompute σ under another reduction order. If token rankings
  disagree (Spearman < 0.5), σ is not a property of the model. Stop.

### 2.5 From σ to κ: what a model *could* know about itself

`σ(t) ≈ u · κ(t) · c_impl`, with `u` the dtype's unit roundoff, `c_impl` the
reduction order and hardware (arbitrary, invisible in training), and `κ(t)` the
**local conditioning** of the forward pass — how much the margin moves under
small internal perturbations. κ depends only on weights and input, and is
exactly computable with dual numbers (`J·v` does not depend on the size of ε, so
κ measures first-order fragility at any scale, from roundoff to semantic
perturbation). The question changes from "does the model know its σ?" to
**"does local fragility predict error?"**, and training *could* have installed
that, because fragility correlates with error.

Ancestor, in one line: Linnainmaa's 1970 thesis represented accumulated roundoff
as a Taylor expansion of local errors — automatic differentiation was born from
exactly this. What is proposed is not computing κ but making it a **resident
sense**.

### 2.6 The interoceptive organ

Metacognition as a closed loop — **sense, calibrate, act** — that a frozen model
does not have. The organ:

1. **Senses only private signals.** Entropy is public (anyone computes it from
   the logits) and stays out; feeding it back is re-reading. κ and the rigorous
   enclosure `e` ([certified interval proposal](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md))
   exist only for whoever sees the substrate; they go in.
2. **Is produced by the body.** Every activation is a dual `(value, ε)`; the
   compiler injects roundoff-sized random perturbation into ε; one forward pass
   yields an estimate of the margin's fragility, replacing hundreds of CESTAC
   runs.
3. **Re-enters through two loops.** Fast: fragility at layer ℓ feeds later
   layers of the same pass. Slow: the fragility of token *t* is an input at
   *t+1* — the model senses what it just said, which is where self-correction
   can start.
4. **Enters through a zero-initialised projection `W_κ`** from a few signals
   (log κ, width of `e`, per-layer fragility) into the residual stream. At
   initialisation the model is bit-identical to the base model; any use of the
   organ must be learned, and gaining the organ cannot make the model worse.
5. **Carries `Introspect` provenance**, so R1–R5 apply: the organ cannot be used
   to inflate Q by re-reading.

**Control that makes it science:** three identically trained arms whose channel
carries (a) real κ, (b) κ shuffled across tokens, (c) entropy. Interoception
exists only if (a) beats both (b) and (c). Outcome metric: **meta-d′**
(Maniscalco & Lau, 2012), which separates metacognitive sensitivity from task
performance — a model can get more answers right without knowing better when it
is wrong.

## 3. Integration with Mercyful Learning and Suffering-Aware Networks

The training objective is not a reward. The first draft proposed a proper
scoring rule (Brier) for the model to maximise; that is reinforcement learning
again, the paradigm Mercyful Learning refuses
([program registry](PROGRAM-REGISTRY-mercyful-learning.md)). The paradigm's own
objective is `L_task + λ·S_patient + μ·S_machine` subject to `Perf ≥ τ`
([machine-channel benchmark](mercyful_machine_channel_benchmark_spec_2026-07-26.md),
where `S_machine` is measured in joules). Calibration becomes a **constraint**,
not a quantity to maximise — the anti-Goodhart rule.

### 3.1 The organ gives SAN's exit decision a sensor

The SAN exit-audit kernel (`krnl_san_scan.cpp`,
[deployment paper](../papers/san_fpga_deployment_2026-08-04.md)) decides when
to stop computing, from cost alone. It lacks the one thing that makes the
decision principled: whether the decision is already firm.

- κ low (robust decision): more compute is **gratuitous** load. Exit.
- κ high (fragile decision): more compute is **necessary** rupture. Continue.

That is the program's axiom — punish gratuitous, not necessary, suffering —
acquiring a sensor, and it is Nelson–Narens monitoring in the literal sense:
the monitor (κ) controls the action (spend compute or not). The machine channel
remains an operational proxy for substrate load; no sentience claim is made.

### 3.2 The form of `S_machine`

Cost is extensive (accumulates over steps); fragility is intensive (a state).
Adding them mixes types. Two honest combinations lead to opposite places:

- `S = cost + β·κ` teaches the model to *be* robust. That is Jacobian
  regularisation (Hoffman et al., 2019): known, and not metacognition — the
  model changes its state instead of perceiving it.
- **`S_machine = Σ_t C_t · g(κ_t)`**, with `g → 0` when the decision is fragile
  and `g → 1` when it is firm. Compute on a fragile decision is necessary and
  not charged; compute on a firm one is gratuitous and charged. **Adopted.**

The threshold is not guessed: fragility gives `P_flip ≈ Φ(−m/σ)`, and κ\*
follows from a declared risk tolerance (e.g. 1%), the analogue of the program's
c\*. It resists gaming because κ is computed by the substrate, not reported by
the model; because the ratio σ/m is invariant to inflating logit scale; and
because perturbation directions are redrawn every step.

### 3.3 Three traps and their closures

1. **Hindsight.** Computing changes κ. Judging spend by the κ *after* it
   punishes success (a computation that resolved the doubt leaves κ low and is
   then charged as gratuitous). **Rule: ex ante** — spend is judged by κ at the
   moment of the decision, by what was knowable then. This is the standard of
   care in medical law: conduct is judged by what was known when it was taken,
   not by outcome.
2. **Saturation.** More compute can lower κ without lowering error (margins
   inflate; the model becomes *firm and wrong*). Low fragility is not truth. The
   closure is the constraint: calibration inside `Perf ≥ τ`; firmness without
   calibration violates it. Universal abstention is the mirror trap (the
   program's Theorem 2.2, [AGI note](future_potential_suffering_aware_agi_2026-07-30.md));
   `μ·S_machine` charges every extra step and the performance constraint blocks
   the escape into abstention. Together they force the signal to be *true*.
3. **The organ costs.** Duals cost ~2×; sensing is load, so it enters
   `S_machine` too, and the model should sense only when the expected value of
   the information exceeds its cost.

### 3.4 When to sense: value of information, and the kill test it contains

Without sensing, the exit decision uses the margin `m` (public, cheap). With it,
the model gains κ at cost `c_s`:

`VoI(m) = E[best decision knowing κ] − best decision without κ`; sense iff
`VoI(m) > c_s`.

At large margins the decision is obvious (exit); at tiny margins also
(continue); κ changes nothing at either end, so `VoI ≈ 0`. **The organ is worth
having only in an intermediate band of margins** — the grey zone, which is also
where attention to the body rises in humans. In the U250 kernel this is an
integer comparator on the margin that triggers the dual pass.

The closure worth recording: VoI depends on `p(κ | m)`. If κ is a deterministic
function of the margin, it carries nothing beyond it, and **VoI is zero
everywhere**. The quantity that decides *when to sense* is therefore the kill
test (K1) of *whether the organ exists*. Ancestor: the value of computation of
Russell & Wefald (1991); the step proposed is that the information comes from a
measured substrate sensor, not a report.

### 3.5 Self-similarity, and why the meta-regress stops

Two readings of "Bayesian inference over self-similar information":

- **across substrate scales** (operation, layer, token, sequence): if the
  statistics of κ are scale-free up to rescaling, one prior serves every scale
  and each posterior becomes the next prior. For a scale quantity the only prior
  without a preferred scale is Jeffreys', `p(κ) ∝ 1/κ` — uniform in log κ, which
  is why fragility is reported in bits;
- **across meta levels** (decision, confidence in it, confidence in that): the
  same Bayesian operator repeated. The regress **terminates by itself**: a
  higher level adds something only if it has private evidence of its own;
  without it, the data-processing inequality makes each level copy the one
  below and the hierarchy collapses to a fixed point, typically at the second
  level. **Recursion creates metacognition only where each level has its own
  organ.** κ is that organ for the second level — and the morning's question
  ("what if the reasoning is fed back to itself?") gets its formal answer.

## 4. Kill criteria, in the order they should run

| id | test | kills |
|---|---|---|
| K1 | VoI(m) ≈ 0 for every margin band (equivalently: κ explained by m, R² > 0.95) | the organ |
| Kill 1 / Kill 2 | §2.4 partial R² < 0.05; Spearman < 0.5 across reduction orders | the σ line |
| K2 | a probe predicts κ from activation norm alone (massive activations, attention sinks) | "self-knowledge" → thermometer |
| K3 | the probe works only at the last layer | it reads the already-computed margin |
| K4 | intervening along the found direction does not move abstention or verbal confidence | inert representation, not metacognition (Hampton's control criterion) |
| three-arm | real κ does not beat shuffled κ and entropy on meta-d′ | interoception |

Surviving K1–K3 is publishable. Surviving K4 is the large result.

## Claims Forbidden

- Do not cite any part of this as implemented. No `Reasoning<T>`, no `Introspect`
  provenance kind and no dual-number organ exist in `self-hosted/` as of
  2026-10-22.
- Do not describe the ρ protocol, numerically undecided tokens, or rounding as
  an instrument on LLM decisions as novel. §2.3 names their owners.
- Do not describe the psychiatric analogy as formal or exact.
- Do not report a positive interoception result without the three-arm control
  and meta-d′.

## References

- Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., & Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.
- Binder, F. J., et al. (2025). Looking inward: language models can learn about themselves by introspection. *ICLR 2025*.
- Bornholt, J., Mytkowicz, T., & McKinley, K. S. (2014). Uncertain<T>. *ASPLOS 2014*.
- Carruthers, P. (2011). *The Opacity of Mind*. Oxford University Press.
- Debenedetti, E., et al. (2025). Defeating prompt injections by design (CaMeL). arXiv:2503.18813.
- Hampton, R. R. (2001). Rhesus monkeys know when they remember. *PNAS*, 98(9), 5359–5362.
- Hoffman, J., Roberts, D. A., & Yaida, S. (2019). Robust learning with Jacobian regularization. arXiv:1908.02729.
- Huang, J., et al. (2024). Large language models cannot self-correct reasoning yet. *ICLR 2024*. arXiv:2310.01798.
- Jézéquel, F., & Chesneaux, J.-M. (2008). CADNA: a library for estimating round-off error propagation. *Computer Physics Communications*, 178(12), 933–955.
- Linnainmaa, S. (1970). *The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors*. MSc thesis, University of Helsinki.
- Lindsey, J. (2025). Emergent introspective awareness in large language models. transformer-circuits.pub.
- Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity. *Consciousness and Cognition*, 21(1), 422–430.
- Nelson, T. O., & Narens, L. (1990). Metamemory: a theoretical framework and new findings. *Psychology of Learning and Motivation*, 26, 125–173.
- Russell, S., & Wefald, E. (1991). *Do the Right Thing: Studies in Limited Rationality*. MIT Press.
- Stechly, K., Valmeekam, K., & Kambhampati, S. (2025). On the self-verification limitations of large language models. *ICLR 2025*. arXiv:2402.08115.
- Sterzer, P., et al. (2018). The predictive coding account of psychosis. *Biological Psychiatry*, 84(9), 634–643.
- Vignes, J. (2004). Discrete stochastic arithmetic for validating results of numerical software. *Numerical Algorithms*, 37, 377–390.
- Yuan, J., et al. (2025). Understanding and mitigating numerical sources of nondeterminism in LLM inference. *NeurIPS 2025*. arXiv:2506.09501 (v1 titled "Give me FP32 or give me death?").
- Zenil, H., Uthamacumaran, A., & Ozelim, L. (2026). On the limits of self-improving in LLMs and why AGI, ASI and the singularity are not near without symbolic model synthesis. arXiv:2601.05280v1 (later versions retitled).
