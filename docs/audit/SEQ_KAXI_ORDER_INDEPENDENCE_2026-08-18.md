<!-- docs:meta
topic_id: repo.docs.audit.seq-kaxi-order-independence-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.seq-kaxi-order-independence-2026-08-18
-->

# Seq reductions — when inverse-variance fusion is honest

**Lane:** grok-cli5 / `seq-kaxi-independence-20260818`  
**Parent:** Madaros Seq checker on `main` (`#1820`, `c76038e500`)  
**Instrument:** `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run`  
**Not claimed:** Madaros `run`. Prebuilt `bin/souc` still E137 on `seq_new`. `self-hosted/ir/lower.sio` is not in this write set (fable-1 CEI claim).

```text
Semantic-Lane-ID: seq-kaxi-independence-20260818
Owner: grok-cli5
Concept-IDs: proposed:seq.reduction.independence-vs-order
Intent-Preserved: Seq is an ordered observation history; a reduction may discard order only when the estimator is commutative. Independence is a separate assumption from order.
Transformation: none to language meaning. Adds a measured witness that kaxi_fuse (independent inverse-variance) equals sequential normal-normal update and ignores permutation, and that a two-observation BLUE with ρ=0.5 does not equal kaxi_fuse.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: under independent Gaussian observations of one scalar, kaxi_fuse ≡ sequential conjugate update (measured |Δμ|=0, |Δσ|=1 ulp); permutation of the three synthetic individuals leaves the fused posterior unchanged at 1e-15; pair probe ρ=0.5 reports 1.920553 vs BLUE 2.333141 vs true kaxi-weight SD 2.345752; dissertation 4-tuple with prior ⊥ individuals and individuals equicorrelated at ρ=0.5: kaxi reports σ=1.641761, BLUE4 σ=2.211221 (relative gap 0.346859), true kaxi-weight SD 2.277336. Instance, not a general inflation law.
Claims-Forbidden: that rapamycin troughs are correlated at 0.5; that this individualizes a patient; that Madaros can run Seq; that order matters for independent normals; that kaxi_fuse is wrong (it is honest iff independence holds); that positive correlation always inflates the reported independent SD.
Assumptions: normal-normal conjugate; two-observation BLUE of one scalar; ρ is a sensitivity probe; same illustrative numbers as rapamycin_kaxi_fuse_prior.sio (Ferron 1997 prior + synthetic individuals).
Write-Set: tests/run-pass/seq_kaxi_order_independence.sio, docs/audit/SEQ_KAXI_ORDER_INDEPENDENCE_2026-08-18.md
Read-Set: stdlib/epistemic/kaxi.sio, stdlib/epistemic/observe.sio, tests/run-pass/rapamycin_kaxi_fuse_prior.sio
Positive-Witness: lean_single run prints PASS and the three W* lines below
Negative-Witness: BLUE ρ=0 recovers kaxi; if bu<=pu the test FAILs
Acceptance-Gate: SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/seq_kaxi_order_independence.sio exits 0 and prints PASS
Integration-Target: dissertation method appendix / pbpk claim table as language capability, not clinical use
Authoritative-Only-If: the printed numbers are reproduced from the same engine on the committed source
```

---

## Why this is the Seq science

`kaxi_fuse` is inverse-variance over `Seq<Knowledge<f64>>`. The sum commutes, so the type's order is discarded. That is honest when the observations are independent. It is a silent lie when they share a source.

Seq is the type that keeps the history visible so a *different* reduction can be applied. This witness measures both.

---

## Estimators

Independent inverse-variance (what `kaxi_fuse` implements):

\[
\sigma^2_{\mathrm{post}} = \Bigl(\sum_i \sigma_i^{-2}\Bigr)^{-1},\qquad
\mu_{\mathrm{post}} = \sigma^2_{\mathrm{post}}\sum_i \mu_i\sigma_i^{-2}
\]

Sequential normal-normal update (stdlib `observe_with_prior`, inlined because it is not `pub`):

\[
\mu' = \frac{\sigma_p^2 y + \sigma_y^2 \mu_p}{\sigma_p^2+\sigma_y^2},\qquad
\sigma'^2 = \frac{\sigma_p^2\sigma_y^2}{\sigma_p^2+\sigma_y^2}
\]

Two-observation BLUE with correlation \(\rho\):

\[
\mathrm{Var}(\hat\mu)=\frac{v_1 v_2(1-\rho^2)}{v_1+v_2-2\rho\sqrt{v_1 v_2}},\qquad
w_1=\frac{v_2-\rho\sqrt{v_1 v_2}}{v_1+v_2-2\rho\sqrt{v_1 v_2}}
\]

At \(\rho=0\) this is inverse-variance. At \(\rho\to 1\) and \(v_1=v_2\) it refuses to pretend two copies of one measurement are two measurements.

Three-observation equicorrelated GLS (pairwise \(\rho\)), then inverse-variance fuse with an independent prior \(v_0\):

\[
\Sigma_3[i,j]=\rho\sigma_i\sigma_j\ (i\neq j),\qquad
\hat\mu_3=(1^\top\Sigma_3^{-1}1)^{-1}1^\top\Sigma_3^{-1}y,\qquad
\sigma^2_4=\bigl(\sigma_3^{-2}+v_0^{-1}\bigr)^{-1}
\]

\(\rho=0\) recovers `kaxi_fuse` on the four-tuple (positive control).

---

## Measured 2026-08-18 (lean_single)

Command:

```bash
SOUNIO_STDLIB_PATH=$(pwd)/stdlib \
SOUNIO_SOUC_ENGINE=lean_single \
./bin/souc run tests/run-pass/seq_kaxi_order_independence.sio
```

| Witness | Result | Reading |
|---|---|---|
| W1 kaxi vs sequential (4-element Ferron+synthetic) | Δμ = 0, Δσ = 2.220446e-16; μ = 10.982609, σ = 1.641761 | order is disposable; matches `rapamycin_kaxi_fuse_prior` closed form |
| W2 kaxi(fwd) vs kaxi(prior + reversed individuals) | Δμ = −1.776357e-15, Δσ = 0 | permutation is disposable under independence |
| W3 pair 10.8±3.0, 11.5±2.5 | reported independent σ = 1.920553; BLUE ρ=0.5 σ = 2.333141; Δσ = 0.412588; reported-vs-BLUE relative = 0.214828 | **if** ρ=0.5 on *these* variances, the number kaxi prints is 21.5% below BLUE. Not a general inflation law. |
| W4 true var of kaxi weights under Σ | σ = 2.345752 ≥ BLUE 2.333141 > reported 1.920553 | GLS optimality holds; kaxi both reports too-small σ and uses slightly suboptimal weights |
| W5 4-tuple (prior ⊥, individuals ρ=0.5) | BLUE4 μ=11.262710 σ=2.211221 vs kaxi μ=10.982609 σ=1.641761; relative gap 0.346859 | the number `rapamycin_kaxi_fuse_prior` prints is 34.7% below BLUE **if** that probe Σ is true |
| W6 true var of kaxi-on-4 weights | σ = 2.277336 ≥ BLUE4 2.211221 > reported 1.641761 | same GLS split on the dissertation tuple |

W3 ρ=0 recovers independent kaxi (positive control). Compact BLUE and explicit 2×2 inverse agree to 1e-9.

Prebuilt Madaros `./bin/souc check` of this file still reports E137 `seq_new` — that binary does not include `#1820`. Do not treat that as a test defect.

---

## Non-claims

- Not a patient. Not TDM. Not a measured rapamycin correlation.
- Not a Madaros run-path close. Lowering remains parked (`/workspace/.tmp/grok-cli5/seq-lowering-wip-20260817.patch`) and `lower.sio` is held by another lane.
- `kaxi_fuse` is not being replaced. The witness names the assumption it already documents (`N INDEPENDENT`).
