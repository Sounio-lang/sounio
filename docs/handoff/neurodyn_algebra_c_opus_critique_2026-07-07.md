<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-algebra-c-opus-critique-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-algebra-c-opus-critique-2026-07-07
-->

# NeuroDyn Algebra-C — Opus Theory/Literature Critique

Date: 2026-07-07
Reviewer role: theory criticism, SOTA critique, reviewer-attack surface, null
validity, claim-boundary pressure (Codex owns implementation/gates/execution).
Scope: critique of `docs/research/neurodyn_algebra_c_continuous_associator_prereg_2026-07-07.md`
before any Algebra-C Slurm run.

Claim boundary (inherited, reaffirmed): synthetic non-clinical algebra-necessity
assay only. No clinical, biomarker, biological-mechanism, treatment-response,
solved-associator, or broad O-SSM superiority claim.

## Decision

**BLOCK_ALGEBRA_C_CIRCULAR_OR_UNDERCONTROLLED**

This is not hostility to exploration. The lane can become a legitimate
inductive-bias methods result. But as preregistered it (a) inherits Algebra-B's
unexplained null failure through the same pipeline, (b) has a primary target that
is categorical/degenerate rather than continuous, and (c) lacks the one control
that would make a positive result mean anything. Two of these preconditions live
*outside* the Algebra-C document, so the honest artifact is a formal blocker with
a concrete acceptance gate, not an approval hedged with edits.

## Formal Blocker Record

```text
Blocker-ID: BLK-20260707-neurodyn-algebra-c-undercontrolled
Status: superseded — see Round 3 (2026-07-07): "methodology cleared" WITHDRAWN;
  a verified core-algebra defect (BLK-20260707-neurodyn-oct-mul-not-normed) now
  blocks the lane. Do not run an Algebra-C smoke.
Severity: B1
Class: evidence-gap
Owner: Codex (execution) / Opus (critique authored)
Lane: NeuroDyn Algebra-C continuous associator fidelity
Worktree: /workspace/sounio
Branch: coord/lane-8c-dossier
Files-Owned: docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md (Opus)
Files-Read-Only: docs/research/neurodyn_algebra_c_continuous_associator_prereg_2026-07-07.md,
  scripts/research/neurodyn_octonionic_associator_manifest.py,
  scripts/research/neurodyn_associator_vector_probe.py,
  scripts/research/neurodyn_noncommutative_temporal_manifest.py
Do-Not-Touch: examples/brain_ossm_abide.sio,
  scripts/research/neurodyn_algebra_b_decision_gate.py,
  scripts/gpu/prepare_abide_campaign_snapshot.sh,
  scripts/research/neurodyn_direct_slurm_smoke.sh
Repro: read associator target construction —
  neurodyn_associator_vector_probe.py:83 (categorical one-hot target) and
  neurodyn_octonionic_associator_manifest.py:236 (continuous mode still pins
  target_assoc_dim/target_assoc_sign)
Observed: primary "continuous associator" target is constant within (assoc_dim,
  sign) cells (~14-28 support points, heavy ties); default first-run pins a
  single fixed signed component (one-sided); Algebra-B pair-label null failure
  (null_08 BA 57.857143 > true 55.892857, 23/99) is not root-caused; no
  non-hypercomplex capacity control exists in the required-surfaces list.
Expected (to clear this blocker): see Acceptance-Gate.
Acceptance-Gate:
  1. Algebra-B null failure root-caused: state explicitly whether the 99 nulls
     RETRAIN the full train/test pipeline per permutation or SCORE a frozen
     model. Report which, with the exact command. (Determines whether B was a
     clean null p~=0.24 or a broken metric — C inherits the same pipeline.)
  2. Genuinely continuous per-sequence target: compute the associator scalar
     from each realized (jittered/noisy) input trajectory, not assigned per
     generator category; report distinct-support count and tie fraction; remove
     the "signed fixed component implied by fixed-dim6" default.
  3. Add a matched non-hypercomplex high-capacity control (real-valued GRU / S4 /
     small Transformer) to the required surfaces, parameter-matched and reported.
  4. Pre-specify the associative-projection quaternion subalgebra (which one, or
     worst-case/averaged over admissible choices) before execution.
  5. Nulls preregistered as full-pipeline-retrain permutations with a stated
     exchangeability argument for the continuous target; resolve the
     sign-AUC vs fixed-sign-target inconsistency.
  6. Claim-boundary paragraph acknowledging the generator/model circularity
     ceiling (below).
Evidence-Level: E2
Evidence: this file; source lines cited in Repro.
Fallback-Path: none (research lane, no merge target)
Legacy-Kept: n/a
LLM-Offload: required-pending (math-bearing; prior xAI channel returned
  "NO MATHEMATICAL CONTENT TO REVIEW" — not validation)
Next-Action: Codex addresses Acceptance-Gate items 1-6 (prereg edits + B
  root-cause) and requests re-review; Opus re-reviews before any smoke run.
```

Only the human author can downgrade/waive this to run Algebra-C without the gate.

## The load-bearing finding: the target is not continuous

`docs/...algebra_c...prereg` sells Algebra-C as escaping Algebra-B by moving from
binary label accuracy to "continuous associator fidelity." The implementation it
points at does not deliver a continuous target:

- The associator is computed from **noiseless basis triples**:
  `nonassociative_triples()` calls `associator(basis(a),basis(b),basis(c))`
  (`neurodyn_octonionic_associator_manifest.py:159`). Input noise
  (`noise_std=0.015`) and jitter do not enter the target.
- The probe target is a **signed one-hot**:
  `target = [+1 if d==assoc_dim and value>0 else -1 if d==assoc_dim else 0]`
  (`neurodyn_associator_vector_probe.py:83`). It is constant within a
  (dominant-dim, sign) cell.
- Even `--triple-source=continuous` rejection-samples to a **fixed**
  `target_assoc_dim` and `target_assoc_sign`
  (`neurodyn_octonionic_associator_manifest.py:236`). Inputs vary; the target
  category does not.

Consequence: the primary "held-out Spearman vs continuous ground-truth
associator scalar" is, as defaulted, rank-agreement against a ~14–28-level,
tie-heavy categorical variable — and under the prescribed first-run default
("signed fixed component implied by fixed-dim6"), a **single fixed sign**, i.e.
an essentially one-valued target for which Spearman is ill-posed and the
`sign-AUC` secondary is undefined (only one class present).

**Algebra-C, as written, is Algebra-B re-scored, not a new test.** If B's null
failure was pipeline-level (capacity fits permuted targets as well as real ones),
swapping the scoreboard to Spearman over the same categorical structure cannot
escape it. The prereg's own rationale — "continuous fidelity gives more
resolution/power" — is only true if the target is actually continuous, which
requires computing the associator from each realized trajectory (a real,
constructible change, since the generator already has a continuous mode). That is
Acceptance-Gate item 2, and it is not optional.

## The circularity ceiling (permanent, must be stated even if C passes)

The synthetic target is the octonion associator of inputs literally drawn from
the octonion multiplication table (`oct_mul`, line 122). The O-SSM's state update
*is* octonion multiplication. So "O-SSM recovers the associator better than an
associative control" is, at bottom, "the octonion basis fits octonion-generated
data better than a quaternion basis." That is a statement about representational
match to a hand-built generator, not evidence that a non-associative object
exists in any natural system.

This cannot be fully removed by any edit — it is intrinsic to a synthetic
generator/model built from the same algebra. It can be *contained*:

- Frame the strongest allowed claim as an **inductive-bias** result: octonionic
  composition recovers a synthetic non-associative target that matched-capacity
  associative *and generic* models cannot, on held-out splits, with retrain-nulls
  at chance. That is a legitimate methods contribution.
- Never let it read as "the brain / MDD / ADHD has a non-associative dynamic."
  Real-data bridge stays blocked (the prereg already blocks it; keep it).

## Direct answers to the coordination questions

**1. Is continuous associator fidelity a non-circular target?**
No, not fully — and as implemented it is not even continuous (see above). The
generator/model share the octonion algebra, so a positive result is a
representational-match statement. It is *usable* only if (a) made genuinely
continuous per-sequence, (b) benchmarked against a generic capacity control, and
(c) claimed strictly as inductive bias for this synthetic generator.

**2. Is Spearman vs ground-truth associator a sufficient primary endpoint?**
Not by itself, and not over a tie-heavy categorical target. Requirements:
- Make the target genuinely per-sequence continuous first; report distinct
  support and tie fraction; if ties dominate, Spearman is uninformative.
- Report a permutation p-value on the Spearman statistic itself (retrain-null),
  not just a point estimate vs a margin.
- Promote on the margin over the **best** control (including the new generic
  control), not only over associative hypercomplex controls.
- Keep R² secondary (scale-sensitive, gameable by variance matching).

**3. Are A8/H+H, H-SSM, associative projection, and raw probes sufficient
controls?**
No — they are all *within the hypercomplex/algebraic family or below it*. The
missing control is a **matched non-hypercomplex high-capacity sequence model**
(real GRU / S4 / small Transformer). If a generic model recovers the associator
scalar as well as O-SSM, octonionic structure is not necessary — capacity is.
The repo's own SOTA note already names S4 (fMRI-S4) and dynamic-GNN families as
reviewer-mandatory; requiring one at the synthetic stage is consistent, not
scope-creep. Also underspecified: the associative projection needs a named,
pre-committed quaternion subalgebra (there are many; choosing post-hoc is a
garden-of-forking-paths).

**4. Which nulls are exchangeability-valid for a continuous target?**
- Permuting the continuous target across sequences within site/pair-balance
  strata, **with full-pipeline retrain per permutation** — valid and required.
- Within-pair target swap (pos↔neg) — valid, and it is the sharpest null; it is
  also exactly where Algebra-B failed, so it must be run and reported, not
  avoided.
- Temporal/order shuffle preserving per-subject feature marginals — valid as an
  order-necessity null.

**5. Which nulls are invalid or too weak?**
- Any null that **scores a frozen model** on permuted targets instead of
  retraining — invalid for a trained model; it undercounts the model's capacity
  to fit noise and can produce the paradoxical "nulls beat truth" pattern.
- Temporal reverse — keep it, but only as a mechanistic diagnostic (as the
  prereg already says); it is not an exchangeability null.
- A single-null "bridge" is too weak to promote; 99 retrain-nulls for the
  primary, as the prereg states, is the floor.

**6. What SOTA baselines would reviewers demand before a real-data bridge?**
Beyond the synthetic controls: real-valued SSM/S4 (fMRI-S4, arXiv:2208.04166),
static-FC + linear/SVM, sliding-window dFC, and a dynamic GNN/attention model
(ASTNet-family). Reviewers will also demand site-held-out evaluation, checkpoint
persistence, multiplicity correction, and effect-size CIs. None of this is
in-scope for the synthetic gate, but the synthetic gate must not claim anything
that these baselines would immediately dominate.

**7. What exact claims remain disallowed even if Algebra-C passes?**
- "O-SSM detects a non-associative signal in the brain / in MDD / in ADHD." —
  disallowed (synthetic only; real-data bridge blocked).
- "Octonionic dynamics are necessary for temporal composition." — disallowed;
  the most C can show is necessity *for this synthetic target vs these controls*.
- "Continuous associator fidelity is a biomarker / mechanism." — disallowed.
- "O-SSM is superior to SSM/GNN/dFC." — disallowed without those baselines.
- Any language that drops the qualifier "synthetic, generator-matched,
  inductive-bias." The honest headline is bounded: *"an octonionic state model
  recovers a synthetic non-associative observable that matched-capacity
  associative and generic models do not, under retrain-nulls."*

**8. What result pattern would convince me O-SSM measures a real non-associative
object rather than capacity/shortcut/noise?**
All of the following on a genuinely continuous per-sequence target:
- O-SSM held-out Spearman positive and ≥0.10 above the **best** control,
  including the generic GRU/S4/Transformer, not just associative ones.
- Associative-projection O-SSM collapses to the control band.
- Raw/flat and generic-capacity probes stay in the control band (shortcut and
  capacity both excluded).
- 99 **retrain**-nulls do not reach true O-SSM Spearman (within-pair swap
  included); empirical p ≤ 0.01.
- The margin is stable across seeds and across the anchor/noise regimes, not a
  single lucky configuration.
- And even then, the claim stays "inductive bias for this synthetic generator."

## Required edits (Acceptance-Gate, restated as the to-do list)

1. State whether Algebra-B (and Algebra-C) nulls retrain or score-frozen; make
   Algebra-C nulls full-pipeline retrain. Root-cause B's 23/99 before pivoting.
2. Replace the categorical/fixed-sign default target with a genuinely
   per-sequence continuous associator scalar (norm or per-trajectory signed
   component); audit distinct support and tie fraction; drop "fixed-dim6"
   default.
3. Add a matched non-hypercomplex capacity control (GRU/S4/Transformer) to
   required surfaces; report parameter counts.
4. Pre-specify the associative-projection subalgebra.
5. Resolve sign-AUC vs fixed-sign inconsistency; if the target is one-sided,
   remove sign-AUC; if two-sided, generate both signs.
6. Add the circularity-ceiling claim-boundary paragraph.

## Claim-boundary warnings

- Do not describe the current or defaulted target as "continuous." It is
  categorical until Gate item 2 lands.
- Do not treat "O-SSM beats A8/H+H/H-SSM" as evidence of necessity — it is a
  within-family comparison until a generic capacity control is added.
- Do not let a synthetic pass migrate into MDD/ADHD language. Bridge stays
  blocked.
- Any math-bearing promotion needs a real offload review; the xAI channel
  returning "NO MATHEMATICAL CONTENT TO REVIEW" is a provenance record, not
  validation.

## Literature / SOTA sources used

- Repo SOTA note `docs/research/neurodyn_ossm_sota_deep_research_2026-07-05.md`
  (already enumerates the reviewer-mandatory baselines; cited rather than
  re-derived).
- Deep Octonion Networks, Wu et al., arXiv:1903.08478 — octonion nets are
  vision/CNN; associator recovery is **not** used as an endpoint there (its
  absence in the literature is both the novelty and the risk).
- fMRI-S4, arXiv:2208.04166 — the generic temporal baseline reviewers will
  demand; motivates Gate item 3.

## Whether Codex may proceed

No. Codex may implement the Acceptance-Gate edits and re-request review, but must
not run an Algebra-C smoke until Gate items 1–6 are addressed or the human author
explicitly waives this B1 blocker.

## Files

- Read: Algebra-C prereg, Algebra-B prereg, SOTA note, coordination prompt,
  PARALLEL_BLOCKER_CONTRACT.md, noncommutative-temporal manifest,
  octonionic-associator manifest, associator vector/head probes.
- Written: this file only. No Codex-owned file modified.

---

## Re-Review (Round 2, 2026-07-07): Acceptance Gate Met

Codex returned an implementation addressing all six acceptance-gate items. I
verified each against the actual source, not the summary.

| Gate item | Status | Evidence verified |
|---|---|---|
| 1. B null retrain root-cause | **Met** | `docs/handoff/neurodyn_algebra_b_null_retrain_audit_2026-07-07.md`: nulls call `init_model` inside each fold and train on permuted `LABELS` (full-pipeline retrain), with `run.rc=0`, `SHA256SUMS.output` verified, `null_08` BA 57.857143. B's failure is a **real instability** (retrain-nulls reach/exceed truth ⇒ empirical p≈0.24, no separable signal), not a frozen-score bug. Honestly reported. |
| 2. Genuinely continuous per-sequence target | **Met** | `neurodyn_octonionic_associator_manifest.py:660` computes the target from the **noisy stretched sequence the model consumes** (`realized_associator_target(pos_sequence,…)`, `target_source="realized_stretched_sequence"`). Audit: 56/56 distinct, tie fraction 0.0, both signs global + per pseudo-site. Fixed-dim6 one-sided default removed; `--target-assoc-sign=0`. |
| 3. Non-hypercomplex capacity control | **Met** | `neurodyn_algebra_c_external_baselines.py`: real `nn.GRU` regressor + `gru_wide` (2× width), trained **directly on `target_scalar` with MSE** — a *conservative* control (generic model gets the target as its objective; O-SSM is binary-trained + post-hoc linear readout). Param counts reported. Gate returns `WARN_UNDERCONTROLLED` if `gru_wide` absent. |
| 4. Associative-projection pre-specified | **Met** | Prereg pins `H_123 = span{1,e1,e2,e3}` (zero `e4..e7`) before results. |
| 5. Retrain nulls + sign-AUC consistency | **Met** | Prereg: all promotion nulls are full-pipeline retrains; adds trajectory-preserving continuous-target retrain null (20) + 99 standard retrain nulls; frozen-score nulls explicitly invalid. `sign-AUC` gated on both-signs-present. |
| 6. Circularity-ceiling claim boundary | **Met** | Prereg §Promotion Rule now states the strongest permitted claim is a bounded inductive-bias result; does not unblock MDD/ADHD/biomarker/mechanism/superiority. |

**Re-Review verdict: acceptance gate satisfied. Downgrade B1 → review-ready;
methodology cleared.** Codex may run the first Algebra-C smoke (5-null debug
bridge, then scale to the full 20+99 retrain-null envelopes). Per the blocker
contract, formal *closure* of BLK-20260707 is the human author's call, but I
record no remaining methodological blocker.

### What clearing does and does not mean
- **Cleared:** the *control/methodology* gate. The experiment is now well-posed
  and can produce information in either direction.
- **Not cleared (still required before any promotion/publication):**
  - full-scale retrain-null envelopes actually run and reported;
  - a *real* offload math review — the xAI channel still returns "NO
    MATHEMATICAL CONTENT TO REVIEW"; that governance-offload debt (`B3`) is
    open and independent of this B1;
  - the bounded inductive-bias claim language on any writeup.
- **Permanent:** the generator/model circularity ceiling holds regardless of
  outcome. Even a clean pass is "octonion inductive bias fits an
  octonion-generated synthetic target better than matched controls," never
  evidence of a real non-associative object.

### Residual notes (refinements, not blockers)
1. **Probe-family asymmetry.** The O-SSM primary is a linear ridge readout on
   traced hidden state; the GRU is a full nonlinear end-to-end regressor. This is
   conservative for a *positive* O-SSM claim (what the blocker guarded), so it
   does not re-block. But a *negative* route firing could partly reflect the
   linear-readout handicap rather than the algebra — treat an O-SSM loss as
   "not shown," not as "octonionic structure refuted."
2. **Prior expectation.** Given B's clean retrain-null (p≈0.24), the honest prior
   is that C likely returns negative or `WARN`. That is an acceptable, informative
   outcome — the point of the redesign was to make the test fair, not to
   guarantee a pass.

---

## Re-Review (Round 3, 2026-07-07): xAI grok-4.3 offload — VERIFIED CORE-ALGEBRA DEFECT

Per human instruction, the mandatory math offload was routed to xAI **grok-4.3**
(the fast-reasoning task template kept returning "NO MATHEMATICAL CONTENT"; the
top model, given a self-contained prompt, returned a substantive review). It
flagged a sign error; I then **verified every claim numerically** (harness
validated against canonical Cayley-Dickson octonions → composition/alternative
errors ~1e-15, so the harness is trustworthy). Results:

**The `oct_mul` used by both the model and the target generator is NOT a valid
octonion product.**
- Repo table gives `e2*e5 = -e7`; the octonion consistent with the stated Fano
  line `(2,5,7)` requires `e2*e5 = +e7`.
- That single wrong sign makes the product **fail the composition law**
  (‖a·b‖ vs ‖a‖‖b‖, max err **4.40**) and **fail the alternative law** (max err
  **72.2**). It is not the octonions, not split-octonions, not any
  composition/alternative algebra. (Symptom: 2 of 122 "non-associative" triples
  produce a *real-part* associator, impossible for genuine octonions.)
- The intended table built from the stated Fano lines *is* a valid octonion
  (err ~1e-15) and differs from the repo table in **exactly** the `e2*e5`/`e5*e2`
  entry.
- `examples/brain_ossm_abide.sio:1340-1347` (`do_oct_mul`) and
  `scripts/research/neurodyn_octonionic_associator_manifest.py` (`oct_mul`) are
  **byte-identical**, so both carry the same defect.

**Exact fix (Codex to apply in BOTH files — I did not touch Codex-owned files):**
in `TMP_OCT[7]` change `- a2*b5 + a5*b2` to `+ a2*b5 - a5*b2`; regenerate every
"octonionic" artifact afterward. Confirm post-fix: composition_err and
alternative_err ~1e-15.

**Second verified finding — associative-projection control is confounded.**
Of 122 non-associative triples, **60 have their associator dominant in e4..e7**,
which the `H_123 = span{1,e1,e2,e3}` projection zeroes. The prereg's default
target component is **6 ∈ {4..7}**, so the "associative-projection collapse"
required by the promotion rule is **vacuous — guaranteed by construction, not
evidence of anything**. Either restrict the target to a component the projection
preserves *and* re-derive what the projection then tests, or replace the control.

**Circularity — independently confirmed** by grok-4.3 as mathematically sound
(neither over- nor under-stated). This corroborates the permanent ceiling.

**Third finding (from the three-provider review) — the sign bug also breaks the
null.** grok-4.20-reasoning and Z.AI glm-5.2 disagreed on whether the negative
class `[b,a,c]` is a clean sign-flip of `[a,b,c]`. Ground truth resolves it and
the disagreement is *entirely* the sign bug:
- Under the **actual (broken) table**: `[e1,e2,e4]=+2e7` but `[e2,e1,e4]=0`;
  `[b,a,c]=-[a,b,c]` fails generally (violation 65.7). The negative class is
  **not** a clean sign-flip, so the within-pair target-swap null is **not
  exchangeable** (grok-4.20 correct for the repo table).
- Under the **fixed octonion**: `[e2,e1,e4]=-2e7` and `[b,a,c]=-[a,b,c]` holds
  exactly (2.5e-14) because the associator is alternating (glm-5.2 correct for a
  real octonion).

Consequence: **the single `e2*e5` sign fix simultaneously (i) restores a valid
normed/alternative octonion and (ii) repairs the null-exchangeability defect.**
Until it is fixed, both the algebra label *and* the within-pair null are invalid.
Circularity is now confirmed unanimously (grok-4.3, grok-4.20, glm-5.2); the
projection confound is confirmed by all three plus the numeric count.

---

## Re-Review (Round 4, 2026-07-07): fix applied and INDEPENDENTLY VERIFIED

Codex applied the two-term flip in both files (`examples/brain_ossm_abide.sio:1347`
and the manifest `oct_mul` `c7`) and validated via py_compile, `bin/souc check`,
a lean_single `.sio` harness, and the numeric gate. I re-ran the gate against the
on-disk files, independently:

- `TMP_OCT[7]` / `c7` now read `+ a2 * b5 - a5 * b2` in **both** files (confirmed).
- **composition_err = 3.55e-15, alternative_err = 2.84e-14, e2*e5 = +e7** → a
  genuine normed **alternative octonion**.
- Non-associative triples 122 → **168**; associator dominant-dim histogram is
  balanced across e1..e7 (24 each) with **0 e0-dominant** — the impossible
  real-part associator artifact from the broken `(2,5,7)` line is gone.

**The algebra defect (BLK-...-oct-mul-not-normed) is repaired and verified.**
Two items remain before Algebra-C can move:

1. **A/B numeric re-audit (open).** Algebra-A/B headline numbers were produced on
   the invalid table and must be regenerated on the corrected octonion; report
   whether any number changes. Not yet done.
2. **Madaros f64-arg-ABI runtime-proof blocker (new, Codex-registered
   `BLK-20260707-madaros-f64-arg-abi-oct-mul`).** The default Madaros engine
   miscompiles a 16×f64-argument probe of `do_oct_mul` (FAIL default, OK under
   `SOUNIO_SOUC_ENGINE=lean_single`). This blocks the *compiled default-engine*
   runtime proof — **not** the algebra, which is proven via Python + lean_single.
   It is a compiler-lane concern (`compiler-semantics`), consistent with the
   known Madaros octonion/f64 codegen fragility (`lean_single` is the sanctioned
   path). It does not reopen the math finding.

Plus the original Algebra-C control items (genuine continuous target, generic
capacity control, projection confound where k∈{4..7} zeroes the target) still
stand. Net: the math is fixed; Algebra-C stays blocked on A/B re-audit + the
control items, with the Madaros ABI issue tracked separately.

---

## Re-Review (Round 5, 2026-07-07): A/B re-audit closed NEGATIVE — verified

Codex re-ran A/B on the corrected octonion (lean_single Slurm). I verified the
report's numbers and interpretation:

- **Algebra-B: terminal negative.** O-SSM 53.48 BA / 52.11 AUROC vs **H-SSM 55.27
  BA / 54.41 AUROC** — the associative control now *beats* O-SSM, both < 55%.
  Gate `ALGEBRA_B_ROUTE4_TERMINAL_FIXEDDIM6_NEGATIVE`. The pre-fix "attribution
  precondition" (O-SSM 55.89) was an artifact of the invalid non-alternative
  table. This closes the A/B re-audit gate on `BLK-...-oct-mul-not-normed`.
- **Algebra-A lineage (Fano/noncommutative-temporal): negative** (O-SSM 36.9,
  far below promotion).

**`BLK-20260707-neurodyn-oct-mul-not-normed` → closeable:** algebra fixed +
verified + prior results re-audited (all negative). Human author may mark closed.

**Two caveats I add on verification (do not overturn the negative):**

1. **Under-reported 91% shortcut.** Codex's data audit (`data_audit/
   associator_data_audit.json`) shows a raw nearest-centroid probe on the
   "core associator" feature at **91.07% leave-site (94.64% z-scored)**; the
   report cites only raw-flat (32%). So the negative is "trained models fail to
   extract a signal that is *near-trivially present as a direct feature* once
   shortcuts are off" — NOT "no signal." This is a leakage channel that must stay
   disabled, and it reinforces the Algebra-C degeneracy concern (the target is
   close to a direct function of the label-defining construction, so the generic
   GRU control will likely match O-SSM).
2. **`UNKNOWN_SITE`/`site_count=1` in run summaries** (Codex blocker
   `BLK-20260707-neurodyn-fixed-octmul-site-reporting-unknown`): non-fatal for a
   negative (pooled BA below threshold is sufficient to close the line) but
   blocks any leave-site/positive claim. Agreed.

Net state: math defect closed; A/B negative and verified; Algebra-C remains
blocked on its original control items + Madaros f64-ABI runtime proof (or an
explicit lean_single waiver).

### New blocker (supersedes the Round-2 "cleared" status)

```text
Blocker-ID: BLK-20260707-neurodyn-oct-mul-not-normed
Status: fix-verified (Codex applied the two-term flip; Opus re-verified on-disk:
  composition_err 3.55e-15, alternative_err 2.84e-14, e2*e5=+e7, 0 e0-dominant
  associators. Residual to close: A/B numeric re-audit + the Madaros f64-arg-ABI
  runtime-proof blocker below.)
Severity: B1 (B0 for any external "octonionic" claim/artifact)
Class: compiler-semantics (math-correctness) + doc-claim
Owner: Codex (fix) / Opus (found+verified)
Lane: NeuroDyn (all "octonionic" O-SSM work, not just Algebra-C)
Files-Owned (Opus): docs/handoff/neurodyn_algebra_c_opus_critique_2026-07-07.md
Files-To-Fix (Codex): examples/brain_ossm_abide.sio:1347,
  scripts/research/neurodyn_octonionic_associator_manifest.py (oct_mul)
Repro: numeric composition/alternative test on oct_mul vs canonical
  Cayley-Dickson (see .claude/llm_offload_log.md 2026-07-07 row); repo
  composition_err=4.40, alternative_err=72.2; canonical ~1e-15.
Observed: product is non-normed, non-alternative; e2*e5 sign wrong.
Expected: a valid octonion (composition_err, alternative_err ~1e-15).
Acceptance-Gate: flip the e2*e5/e5*e2 sign in both files; re-run the numeric
  validity test (both errors ~1e-15); regenerate all octonionic artifacts;
  re-audit whether any prior "octonionic" result changes.
Evidence-Level: E2 (numeric proof; independently corroborated by two xAI models
  — grok-4.3 and grok-4.20-0309-reasoning — reaching the same conclusions)
Evidence: .claude/llm_offload_log.md (2026-07-07 grok-4.3 + grok-4.20-reasoning
  rows); artifacts/research/neurodyn/resp.json (raw model output); Opus numeric
  harness vs canonical Cayley-Dickson octonions.
LLM-Offload: logged (xai/grok-4.3, math-review)
Next-Action: Codex applies the two-term sign fix and regenerates; Opus
  re-reviews the corrected algebra before any Algebra-C smoke.
```

---

## Re-Review (Round 6, 2026-07-08): real-data smoke executed, bridge remains blocked

Codex executed the computational-psychiatry framework lane on a bounded real
ADHD-200 / PCP pilot and then corrected the first overclaim caught by xAI.
The final gate state is:

```text
FRAMEWORK_EXECUTED_LOW_POWER_PILOT
pilot_decision = UNDERCONTROLLED_LOW_POWER
```

Verified points:

- The pipeline is now real-data executable: public FCP-INDI/PCP ADHD-200 S3
  bootstrap, access audit, manifest preparation, O-SSM/H-SSM state trace export,
  generic recurrent controls, and decision gate all ran end-to-end.
- The pilot used a bounded `n=24` ADHD-200 PCP cache (`KKI=16`, `NYU=8`,
  `ADHD=12`, `TD=12`) and emitted 960 `STATE_TRACE` rows plus the dynamic
  feature table.
- The decision gate now correctly refuses to call this a negative claim:
  `row_count 24 < min_decision_subjects 50` and
  `min null_permutations_mean 1 < min_decision_null_permutations 20`.
- The previous metric-only verdict is retained only as `pilot_metric_verdict`:
  O-SSM hidden was non-competitive on the phenotype-regression surfaces
  (inattention -0.049, hyperactivity/impulsivity -0.063, ADHD total -0.032;
  H-SSM hidden was ~0.14-0.17).

Strategic readout:

1. **There is currently no evidence base for an O-SSM advantage in ADHD.** The
   corrected Algebra-B synthetic lane is negative, and the first real ADHD-200
   smoke is underpowered but non-competitive for O-SSM on downstream phenotype
   regression. Scaling exactly this phenotype-regression target is likely to
   produce a powered negative or null result, not a clinical story. That can
   still be a legitimate methods contribution if framed as a controlled
   falsification/benchmark result.
2. **The next migration should target dynamic state observables, not phenotype
   regression as the primary bridge.** The preregistered scientific bridge
   remains switching rate, dwell time, transition entropy, temporal
   irreversibility, and related path-dependent state descriptors. Phenotype
   association may remain secondary/descriptive only after those dynamic
   observables are defined, exported, and controlled.

Conclusion: the real-data-bridge blocker remains closed against positive
clinical claims. The framework is executable; the next scientific move is
dynamic-state instrumentation, not clinical-story escalation.

**Net effect:** the Round-2 downgrade to "review-ready / methodology cleared" is
**withdrawn**. The control redesign was good work, but it sits on top of a
multiplication table that is not the octonions. No Algebra-C smoke, and no
"octonionic" language anywhere, until the algebra is fixed and re-verified. This
also means every prior NeuroDyn "octonionic" result (Algebra-A/B included) was
computed on a non-octonion product and must be re-audited after the fix.
