<!-- docs:meta
topic_id: repo.docs.audit.epistemic-pbpk28-test6-confidence-independent-verify-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-pbpk28-test6-confidence-independent-verify-2026-08-18
-->

# Independent verify — `epistemic_pbpk28` TEST 6: fabrication closed, formula unprovenanced

**Date:** 2026-08-18
**Lane:** grok-cli5 / pbpk28-t6-conf-verify-20260818
**Question that remains:** is `0.671038` a number the thesis can defend, or only a number the compiler can print?

**Short answer.** The `4.6e18` was IEEE bits. lean_single prints `0.671038` because that is what `Σ εᵢ Sᵢ` evaluates to on the authored priors and this Jacobian. That closes fabrication. It does not close exactness. The aggregation formula has **no external source** and **no written derivation** outside the implementation comments. That is a thesis finding, not a compiler finding.

A same-formula echo in another language is not an audit of the formula. It is not done here.

---

## 0. What this receipt is, and is not

grok-cli2 reported that TEST 6 stops fabricating `4.6e18` and prints
`0.671038`. This hand measured TEST 6 independently (§4) and then asked
where the *formula* comes from (§1–§3). Those are different questions.

Not in scope:

- Recalculating `Σ εᵢ Sᵢ` in Python, MPFR, or a third Sounio file. That
  repeats the arithmetic. Two engines already agree. An error in the
  aggregator would survive every echo.
- Source-building Madaros (not done). The prebuilt ELF here predates
  `#1882` and still fabricates, as expected.
- The E230 patch-split (minimax-cli1).

---

## 1. Where does the TEST 6 formula come from?

The live expression is `ep28_run` in
`stdlib/darwin_pbpk/epistemic_pbpk28.sio` (~413–419):

```text
conf_auc = Σ_c  priors.kn[c].confidence · sensitivity[c]
```

`sensitivity[c]` is the normalised first-order GUM variance share
`(∂AUC_blood/∂θ_c)² · Var(θ_c) / Var(AUC_blood)`. That share **is**
JCGM 100:2008 §5.1.3. The **weighting of authored ε by those shares**
is not.

### What the file itself cites

The PBPK28 header cites JCGM 100:2008 for four steps: reference run,
±h perturbations, central-difference `cᵢ`, and `Var(y) = Σ cᵢ² Var(θᵢ)`.
Ferron 1997, Lampen 1998, Schreiber 1991, Valoteau 1996, Rodgers &
Rowland 2006 are cited for **parameter means / CVs / Kp structure**.
None of those references is attached to the `conf_auc` loop. The loop
has no citation.

### What the house says the formula *is*

The same aggregator appears first in PBPK14
(`a41dafc33b`, `epistemic_pbpk14.sio`) and is copied into PBPK28
(`652133d7d7` stack, Knowledge-consuming form `2e67d5a9da`). The
clearest in-repo statement is a comment, not a derivation:

```text
tests/run-pass/dissertation_pbpk14_gum.sio ~620–624, ~913
  Evidence-weighted confidence (Novel Contribution #2 prerequisite)
  Confidence of AUC = Σ s_i * conf_i  (sensitivity-weighted average)
```

That **names** the rule. It does not derive it. It does not say why
the aggregator is a sensitivity-weighted *mean of ε* rather than
`min(ε)`, a weakest-link floor, or the RSS-of-gaps used elsewhere in
the same dissertation.

PBPK14 TEST 6 (`epistemic_pbpk14.sio` ~655–657) repeats the description:

```text
Confidence is a weighted average of per-parameter evidence quality,
weighted by their sensitivity fractions.
```

Again: what the code computes. Not why.

### What Contribution 2 actually is

The dissertation truth table's **Contribution 2** is the *compile-time*
`with Epistemic(N)` gate: reject a function whose weakest input ε is
below N (`pbpk_claim_truth_table.md`). That is a **min** rule, not
`Σ Sᵢ εᵢ`.

`stdlib/epistemic/ode.sio` (~1353–1366) operationalises the *gate* and
cites ISO/IEC Guide 98-3:2008 §3.3 (“target uncertainty”) and
ISO 17025:2017 §7.6 (decision rules). Those citations attach to
“refuse to use an output below a threshold.” They do not give
`conf_Y = Σ sᵢ confᵢ`, and GUM target uncertainty is a requirement on
`u(y)`, not a recipe for combining evidence-quality scores.

So the runtime TEST 6 number is labeled a *prerequisite* of Contribution 2
in one comment, while Contribution 2 as claimed is a different
combination rule.

### Per-parameter ε has a heuristic; the sum does not

`stdlib/darwin_pbpk/lib.sio` and `drugs/rapamycin.sio` give a **score
heuristic for each prior** (0.40–0.50 single in vitro; 0.65–0.80
replicated in vivo; …) and an explicit refusal: ε is not a Beta
posterior. That is provenance for `ε(cl_hepatic) = 0.65`, not for
folding the seven scores through the Jacobian.

### The house has more than one unpublished aggregator

Same dissertation, other files, other rules, also without an external
table:

| Rule | Where | Combines ε how |
|---|---|---|
| `Σ Sᵢ εᵢ` | PBPK14 / PBPK28 TEST 6 | sensitivity-weighted mean |
| `min(ε)` | Contribution 2 compile-time gate | weakest input |
| `1 − √(Σ(1−cᵢ)²)` | PGx `aggregate_rss` (`docs/dissertation/handoff/psychiatric_pgx_mtor_168_pop_package.md` §2.6) | RSS of evidence *gaps* |
| `Sᵢ · (1−εᵢ)` | VoI ranking (`bbb_voi.sio`, `d2_voi.sio`) | share × residual ignorance |

Three of these can be applied to the same seven rapamycin priors and
will not return `0.671038`. That is what “derived here, not uniquely”
looks like.

**Answer 1.** The TEST 6 formula was derived in this repository. It is
stated in implementation comments as a sensitivity-weighted average.
There is no publication, no GUM clause, and no written derivation that
justifies this aggregator over the others the same thesis already
uses. The derivation lives in the code.

---

## 2. If there is a source, does it give a reference value?

Split the nearby sources. They do not all fail the same way.

| Source | Gives a method? | Gives a value for these inputs? |
|---|---|---|
| JCGM 100:2008 §5.1.2–§5.1.3 | Yes: `cᵢ` and `Var(y)`. | No value for `Σ εᵢ Sᵢ`. The method is not this method. |
| ISO/IEC Guide 98-3 §3.3 / ISO 17025 §7.6 (as cited on the *gate*) | A threshold / decision-rule idea, loosely. | No numeric target for this score. |
| Ferron 1997 | AUC / CL. | TEST 9 uses `0.403226` for **AUC**, a different quantity. |
| Jiao 2009, Schreiber 1991, Lampen 1998, Valoteau 1996 | Means and CVs of `θᵢ`. | No ε vector and no aggregated score. |
| In-repo score heuristic (`lib.sio`) | Bands for *individual* ε. | No combined value. |
| TEST 5 `ref_s0 = 0.697376` … | Pins `S` to lean_single. | Circular as an oracle of `S`, and silent on `ε`. |
| Historical `m6_epistemic_pbpk28_v1.txt` | Prints `AUC confidence: 0.671038`. | Same implementation, earlier run. Not a reference. |

**Answer 2.** No cited source gives this method *and* a number for
these inputs. JCGM gives a neighbouring method (variance) and no
value. The PK papers give inputs to the ODE, not to this score. “The
source gives the method but not the value” would be a legitimate
weaker finding. That is **not** this case. The method itself is
unprovenanced.

There is therefore nothing to compare `0.671038` against except the
implementation that produced it.

---

## 3. If there is no source — that is the finding

**Answer 3.** There is no source. Record it as a thesis fact, not a
compiler bug.

| What is closed | What is not |
|---|---|
| `#1792` / `KCONF-BITCAST-SITOFP`: the `4.6e18` print is `sitofp` of the IEEE bits of `0.6710384899851078` (`0x3fe57925b61afc00`). `#1882` restores the representation. | Whether `Σ Sᵢ εᵢ` is the right scientific object. |
| lean_single TEST 6 prints `0.671038` on the named ELF (§4). Matches grok-cli2 after the fact. | Whether that print is a literature- or GUM-correct *confidence*. |
| The truth table already forbids calling the score a patient-level probability. | A derivation of why this aggregator, and a calibration of the number. |

A number that is formula-exact and unprovenanced is **defensible as
implementation** (the compiler is no longer lying; the file computes
the rule it states) and **indefensible as science** (the rule has no
external warrant and no written argument). Demetrios needs that
distinction in hand before September. TEST 6's band `(0.20, 0.90)` is
a fabrication detector plus a coarse convex check (`ε ∈ [0.40, 0.72]`
⇒ the weighted mean sits in that interval). It is not a literature
pin. The literature pin on this file is TEST 9 (Ferron AUC).

---

## 4. Measurement (instrument, not the formula audit)

Binaries used. Neither was source-built in this lane. E230-v3 is not
in source (`runtime_context_size() -> i64 { 248 }`).

| Role | Path | How obtained | mtime (UTC) | size | sha256 |
|---|---|---|---|---:|---|
| lean_single | `bin/souc-lean-single-x86_64` | committed bootstrap ELF; `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile … -o` | 2026-08-17 14:49 | 2 555 805 | `337d5a86f44ef9320a0485f181283df7d0662b944fe83ada3e536ca45ce48db7` |
| Madaros prebuilt (negative control) | `artifacts/self-hosted/madaros` | ELF already in the worktree; default `./bin/souc compile … -o` | 2026-08-17 15:32 | 99 964 760 | `05d95342e42b36d4ccb8b694b401df92c9918637e7a4cc6dcf4568cf424d9963` |

The Madaros ELF predates `d33cf5856b` / `#1882`. It is the wrong
vintage for a post-fix success cell. It is the right vintage for
“fabrication is a representation defect.”

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile \
  stdlib/darwin_pbpk/epistemic_pbpk28.sio -o /tmp/g5-t6/t6.lean_single.elf
# 133203 bytes, magic \x7fELF; run_rc=0
# TEST 6: AUC confidence: 0.671038  [PASS]   ALL 9 TESTS PASSED

./bin/souc compile \
  stdlib/darwin_pbpk/epistemic_pbpk28.sio -o /tmp/g5-t6/t6.madaros_prebuilt.elf
# 192968 bytes, magic \x7fELF; run_rc=1
# TEST 6: AUC confidence: 4604219396932172800.000000
# EPISTEMIC_FABRICATION (range + bit-pattern)
```

Same-run TEST 5 (this Jacobian, not a literature table):
`S = [0.697376, 0.000452, 0.301837, 5.234690e-8, 0.000223, 0.000111, 3.240925e-8]`.
Authored ε: `[0.65, 0.50, 0.72, 0.63, 0.60, 0.55, 0.40]`.
The prebuilt Madaros TEST 5 still prints the same dominant pair and
scaled tails; only the confidence channel was sitofp'd.

I did not independently re-run source-built post-`#1882` Madaros.

---

## 5. Halt

Receipt is written on
`lane/grok-cli5/pbpk28-t6-conf-verify-20260818`. Commit waits for the
grok-cli2 registry window. Do not `--no-verify`. Request already sent
to `fab-sweep`.
