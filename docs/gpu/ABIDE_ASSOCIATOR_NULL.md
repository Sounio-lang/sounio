<!-- docs:meta
topic_id: repo.docs.gpu.abide-associator-null
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.abide-associator-null
-->

# Real data: the octonion associator does not classify ABIDE-I ASD/TD (an honest null)

The synthetic results (`NONASSOC_HEADTOHEAD.md`) show the octonion associator is provably useful when
the signal *is* non-associative. The obvious question a reviewer will ask: **does it transfer to real
clinical data?** We tested it on ABIDE-I autism classification. It does not. We report the null.

## Setup
- **Data:** ABIDE-I, 500 subjects (250 ASD / 250 TD), 20 sites, balanced within site. The
  `brain_ossm.abide.v2` 8×8 feature layout — each subject is 8 octonions `h_0..h_7`.
- **Octonion features:** the associators `[h_t, h_{t+1}, h_{t+2}]`, t=0..5 (48 components + 6 norms),
  computed on the **compiler-lowered `oct_assoc` tensor-core kernel on the GB10** — the same operation
  validated bit-for-bit in #1204/#1212.
- **Classifier:** L2 logistic regression under **leave-one-site-out** cross-validation. Site is a known
  confound (it produced a spurious trend in the earlier G2 study); LOSO removes it.

## Result (leave-one-site-out balanced accuracy; chance = 50%)
| Features | bal-acc |
|---|---|
| RAW 8×8 (associative) | 49.9% ± 8.2 |
| Quaternion associator ≡ 0 (control) | 51.4% ± 9.7 |
| **Octonion associator** `[h_t,h_{t+1},h_{t+2}]` | **45.4% ± 9.2** |
| Octonion associator + RAW | 47.9% ± 7.9 |

Every model is at chance; the octonion associator adds nothing (if anything it is below chance, within
noise). This is the **third independent null** for octonion methods on ABIDE ASD/TD:
- G2-Gram of Laplacian eigenmodes: Cohen's d = 0.06, p = 0.30 (`project_g2_bridge`);
- O-SSM vs H-SSM balanced accuracy: 49.5% vs 50.0%;
- octonion sequence-associator (this): 45.4%.

## What this means (and does not)
This is a **boundary of the empirical claim, reported as such** — not a failure of the artifact. The
contribution is (1) the compiler that lowers a non-associative algebra and its exact associator/VJP to
tensor cores, and (2) the *provable* synthetic separation where the octonion associator solves a task
associative models cannot. On ABIDE ASD/TD at the representations tried (eigenmodes, O-SSM 8×8,
sequence-associator), non-associativity carries no signal beyond chance. We did **not** search
representations until one crossed 50% — that would be p-hacking a null.

## The decisive test: the FULL 200×200 connectome (with a positive control)

The 8×8 summary above is lossy, so we ran one **pre-registered** test on the genuinely untested door —
the full 200×200 CC200 connectome (Fisher-z Pearson correlation from the raw timeseries), with a
per-fold-fitted (leakage-safe) PCA-8 node embedding, the octonion associator field `[o_i,o_j,o_k]` over
a fixed sample of 300 node triples, leave-one-site-out CV, run **once, no representation search**. The
key addition is a **positive control**: a standard associative connectome classifier (PCA-50 of the
upper triangle).

| Features (leave-one-site-out) | bal-acc | 95% CI lower |
|---|---|---|
| **RAW connectome PCA-50 (associative, positive control)** | **63.9% ± 9.2** | **59.9** ✓ beats chance |
| Quaternion associator field ≡ 0 (control) | 50.3% ± 5.6 | 47.8 |
| **Octonion associator field** | **52.1% ± 9.4** | 47.9 — at chance |
| Octonion field + RAW | 62.7% ± 9.6 | 58.5 (≤ RAW alone) |

This is the strongest form of the null, because **the positive control works**:
- the ASD/TD signal **is present and real** — a standard associative classifier reads it at 63.9%
  (consistent with the ABIDE literature), CI excluding chance;
- the **octonion associator field carries none of it** (52.1%, CI includes 50);
- adding the associator to the associative features **does not help** — it slightly hurts (62.7% ≤ 63.9).

So the four nulls can no longer be blamed on a poor representation or on LOSO being too hard: the full
connectome has signal, an associative model captures it, and the non-associative structure adds nothing.

## What this means (and does not)
This is a **boundary of the empirical claim, reported as such** — not a failure of the artifact. The
contribution is (1) the compiler that lowers a non-associative algebra and its exact associator/VJP to
tensor cores, and (2) the *provable* synthetic separation where the octonion associator solves a task
associative models cannot. On ABIDE ASD/TD — across four representations (eigenmodes, O-SSM 8×8,
sequence-associator, full-connectome associator field) and now against a working associative positive
control — non-associativity carries no clinical signal. We did **not** search representations until one
crossed chance; the full-connectome test was pre-registered and run once.

**The honest state: the non-associative advantage is synthetic-provable and real-clinical null.** A
preprint that says so — with a positive control proving the signal was there to be found — is stronger
than one that hides it. Harnesses `run_abide_assoc.cu` (8×8, GB10 kernel) and
`abide_connectome_assoc.py` (full connectome, pre-registered).
