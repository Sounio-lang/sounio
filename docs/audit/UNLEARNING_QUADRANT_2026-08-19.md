<!-- docs:meta
topic_id: repo.docs.audit.unlearning-quadrant-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.unlearning-quadrant-2026-08-19
-->

# Unlearning quadrant — is it occupied? 2026-08-19

**Question.** Does the published field already occupy the conjunction

> exact unlearning **without retraining**, **by algebraic annihilation**,
> **verified at compile time**

that a Sounio paper would claim for `ExactlyPrivate<T>`?

**This receipt exists to try to knock that claim down**, not to confirm it.
A result that occupies the quadrant, or that occupies one of the three
qualifiers, is the useful result.

**Verdict on the conjunction:** **FREE** as a *published triple*.
Nobody was found who does all three at once.

**Verdict on the attractive over-claim:** **occupied in pieces.**
"Exact without retrain" is not unique. "Privacy checked at compile
time" is not unique. "The field of machine unlearning is entirely
approximate and statistical" is **false**. Those three knockdowns
are the load-bearing output of this search.

No code was changed.

```text
Semantic-Lane-ID: unlearning-quadrant-20260819
Owner: grok-cli3
Concept-IDs: none created
Intent-Preserved: a green claim must survive an attempt to occupy it
Transformation: none — literature census
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced:
  - Q1 PARTIAL: exact unlearning without full retrain exists for restricted models
  - Q2 PARTIAL: compile-time privacy types exist; they verify DP/IFC, not forgetting
  - Q3 FREE: no published use of Cayley–Dickson zero-divisors as an unlearning operator
  - Q4 PARTIAL: published algebra already implies annihilation is unstable off the ZD variety
  - the three-qualifier conjunction is FREE; the over-claim "the field is all approximate" is false
Claims-Forbidden:
  - Sounio uniquely has exact unlearning
  - Sounio currently verifies forgetting at compile time
  - Lean unlearning_kernel_exact is a theorem about trained models
  - left-multiplication by a zero-divisor is a projection onto the complement
  - IEEE f64 residual on the exact ±1 ZD pair is the kill-shot (it is 0.0)
Assumptions: English-language peer-reviewed + arXiv through 2026-08-19
Write-Set:
  docs/audit/UNLEARNING_QUADRANT_2026-08-19.md
  docs/audit/UNLEARNING_QUADRANT_2026-08-19.tsv
Read-Set: formal/lean4/SounioSurgicalInterventions.lean;
  self-hosted/check/check.sio lower_exactly_private_type;
  examples/zd_machine_unlearning.sio;
  grok-cli4 docs/audit/ZD_ANNIHILATE_BUILTIN_DISPATCH_2026-08-19.md (read-only)
Positive-Witness: Cao & Yang IEEE S&P 2015; Fuzz ICFP 2010; Moreno 1998
Negative-Witness: no sedenion-ZD × unlearning hit in the search set
Acceptance-Gate: each of Q1–Q4 is OCCUPIED / FREE / PARTIAL with method;
  peer-reviewed distinguished from preprint; engines/Lean not over-read
Integration-Target: docs (audit)
Authoritative-Only-If: n/a — observational
```

Companion TSV: `docs/audit/UNLEARNING_QUADRANT_2026-08-19.tsv`.

## What Sounio actually has (local baseline, not the field)

State this first so the literature is not compared to a stronger
Sounio than exists.

| Surface | What it is | What it is not |
|---|---|---|
| `unlearning_kernel_exact` in `formal/lean4/SounioSurgicalInterventions.lean` | `native_decide` that four listed primitives are right-annihilated by `primA`. No `sorry`, no Mathlib. | A theorem that a trained model forgot a subject. The file itself labels the type correspondence a **propositional scaffold** and says the full semantics of `ExactlyPrivate<T>` are out of scope. |
| `every_primitive_has_4_annihilators` | Finite census: 84 primitives, degree 4. | A forgetting operator. |
| E201 / `lower_exactly_private_type` | If the type constructor `ExactlyPrivate` is written, effect id 18 (`ZD`) must be declared. The inner type is then returned unchanged. | A check that annihilation occurred, that `x` lies in `ker(A)`, or that the result is indistinguishable from never-trained. |
| `examples/zd_machine_unlearning.sio` Method 3 | Euclidean projection `W − ⟨W,u1⟩u1 − ⟨W,u2⟩u2`. The file **says** the sedenion product is not a projection and is not `W_bob`. | Left-multiplication by `e3+e10` as the unlearning map. |
| grok-cli4 forensic `ZD_ANNIHILATE_BUILTIN_DISPATCH_2026-08-19.md` | Closed builtin `zd_annihilate` proposed, not shipped. Recognition ≠ contribution-zero. | A compile-time forgetting proof. |

So even if the field were empty, the candidate sentence would still
over-read the Lean file and the checker.

## Method

Searched 2026-08-19. Absence is reported only with the query in front.

| # | Query family | Surfaces |
|---|---|---|
| S1 | `exact machine unlearning without retraining algebraic closed form` | web index |
| S2 | `SISA Bourtoule machine unlearning` | web + arXiv `1912.03817` (confirmed live) |
| S3 | `Cao Yang Towards Making Systems Forget` | web; IEEE S&P 2015 |
| S4 | `Cauwenberghs Poggio incremental decremental SVM` | NIPS 2000 proceedings |
| S5 | `Ginart Making AI Forget You k-means` | arXiv `1907.05012` (confirmed live) |
| S6 | `Guo Certified Data Removal` | arXiv `1911.03030` (confirmed live) |
| S7 | `Izzo Approximate Data Deletion Regression` | arXiv `2002.10077` |
| S8 | `Closed-form Machine Unlearning Matrix Factorization CIKM 2023` | ACM `10.1145/3583780.3614811` |
| S9 | `Fuzz Reed Pierce differential privacy type system` | ICFP 2010 |
| S10 | `DFuzz Gaboardi POPL 2013` | POPL 2013 |
| S11 | `Duet Near type system differential privacy` | OOPSLA 2019 / arXiv `1909.02481` |
| S12 | `Jif Myers Liskov information flow` | POPL 1999 |
| S13 | `"zero divisor" (privacy OR unlearning OR forgetting OR GDPR) (sedenion OR octonion OR "Cayley-Dickson")` | web index |
| S14 | `sedenion zero divisor floating point numerical instability` | web; then primary algebra |
| S15 | `Moreno zero divisors Cayley-Dickson Bol Soc Mat Mexicana` | arXiv `q-alg/9710013`; *Boletín* 1998 |
| S16 | `Cawagas structure zero divisors sedenion` | *Discussiones Mathematicae* 2004 |
| S17 | `"certified unlearning" OR "verifiable unlearning" type system OR compile-time` | web + arXiv `2210.09126` (confirmed live) |
| S18 | `Nguyen Survey of Machine Unlearning` | arXiv `2209.02299` (confirmed live) |
| S19 | `Higham Cholesky downdating unstable` | Higham ASNA 2002; Bojanczyk–Brent–van Dooren–de Hoog *SISC* 1987 |

arXiv titles confirmed by `export.arxiv.org` on 2026-08-19 for
`2209.02299`, `1912.03817`, `1911.03030`, `1907.05012`, `2411.18881`,
`2210.09126`.

Not treated as evidence: unsourced survey-tool paraphrases; arXiv IDs
that did not resolve; patents except as a negative (octonion FHE has
no zero-divisors).

Coverage hole, declared: non-English venues, dissertations not on
arXiv, and 2026 workshop papers not yet indexed. That hole is why
Q3 is FREE and not a proof of non-existence.

## Q1 — Exact unlearning without retraining

**PARTIAL.**

The field is **not** "entirely approximate and statistical". Exact
removal without a full from-scratch retrain is published for
restricted model classes. SISA is exact **and retrains fragments**;
it occupies exactness, not the "no retrain" cell.

| Work | Venue / date | Review | What is exact | Retrain? |
|---|---|---|---|---|
| Cauwenberghs & Poggio, incremental/decremental SVM | NIPS 2000 | peer-reviewed | decremental step yields the batch SVM without the point | no full QP restart |
| Cao & Yang, *Towards Making Systems Forget with Machine Unlearning* | IEEE S&P 2015 | peer-reviewed | subtract the sample from SQ / summation statistics and recompute the model | no full pass over the retain set |
| Ginart, Guan, Valiant, Zou, *Making AI Forget You* | NeurIPS 2019; arXiv `1907.05012` | peer-reviewed | deletion-efficient *k*-means, equal in distribution to never-trained | not full Lloyd from scratch; supporting structures |
| Bourtoule et al., SISA / *Machine Unlearning* | IEEE S&P 2021; arXiv `1912.03817` | peer-reviewed | aggregated model ≡ train without the point | **yes** — the affected shard/slice |
| Sherman–Morrison / Woodbury on ridge / OLS | classical NLA; used by Izzo et al. AISTATS 2021 (`2002.10077`) | peer-reviewed (Izzo: **approximate** for logistic; OLS downdate is exact) | rank-1 downdate of \((X^\top X)^{-1}\) | no |
| Zhang, Lou, Xiong, Zhang, Liu, CMUMF | CIKM 2023, ACM `10.1145/3583780.3614811` | peer-reviewed | closed-form Newton / Hessian step for matrix factorisation | no full MF retrain |
| Guo, Goldstein, Hannun, van der Maaten, *Certified Data Removal* | ICML 2020; `1911.03030` | peer-reviewed | **certified**, not exact: Newton + loss perturbation | no |

**Occupied cell:** exact, no-full-retrain, for linear / SQ / SVM /
some clustering / some MF.

**Free cell:** exact, no-retrain, for a general non-convex net, by
an operator that does not look at the retain set. Surveys
(Nguyen et al., arXiv `2209.02299`, v6 2024) still split the field
into exact (usually shard/retrain) and approximate (influence,
ascent, distillation).

Sounio Method 3, as written, is ordinary orthonormal projection
onto the complement of a known subspace. That operator has been
available since least squares. It occupies nothing new.

## Q2 — Compile-time verification of a privacy property

**PARTIAL.**

Languages **do** verify privacy-adjacent properties statically.
They do **not** verify "this programme forgot subject Alice".

| System | Venue / date | Review | What the types actually check |
|---|---|---|---|
| Fuzz (Reed & Pierce) | ICFP 2010 | peer-reviewed | metric sensitivity; well-typed + Laplace ⇒ ε-DP |
| DFuzz (Gaboardi, Haeberlen, Hsu, Narayan, Pierce) | POPL 2013 | peer-reviewed | linear dependent sensitivity; still DP, not forgetting |
| Duet (Near, Darais, Abuah, …) | OOPSLA 2019; `1909.02481` | peer-reviewed | dual linear types for sensitivity and privacy composition, including SGD-shaped programmes |
| Jif / JFlow (Myers; Myers & Liskov labels) | POPL 1999; SOSP 1997 | peer-reviewed | information-flow confidentiality / integrity, not unlearning |
| Eisenhofer, Riepel, Chandrasekaran, Ghosh, Ohrimenko, Papernot | arXiv `2210.09126`; IEEE SaTML 2025 | preprint then peer-reviewed | **cryptographic** proof that an agreed unlearning procedure ran; SNARKs + hash chains. Not a type system. Runtime / protocol. |

Search S17 (`certified unlearning` / `verifiable unlearning` +
`type system` / `compile-time`) returned certified *statistical*
removal and cryptographic *execution* proofs. It did not return a
type system whose judgement is "the contribution of this value has
been annihilated".

Sounio E201 is closer to an effect-annotation mandate than to Fuzz.
Fuzz proves a metric bound. E201 proves the letters `with ZD` are
present.

## Q3 — Zero-divisors used for privacy or unlearning

**FREE**, for the search set in the Method table.

Hits that look adjacent and are **not** the cell:

| Work | Why it is not the cell |
|---|---|
| Saoud & Al-Marzouqi, metacognitive sedenion NN | IEEE Access 2020, peer-reviewed. Time-series forecasting. Zero-divisors are an algebraic fact of the algebra, not an unlearning operator. |
| Reggiani, *The geometry of sedenion zero divisors* | arXiv `2411.18881`, 2024, preprint. Riemannian geometry of \(Z(\mathbb{S})\). Mentions ML as motivation, does not erase training points. |
| Cawagas 2004; Moreno 1998 | Structure of the 84 pairs / \(\dim\ker L_a \equiv 0 \pmod 4\). Algebra, not privacy. |
| US patent US20190007196A1 (octonion FHE) | Octonions are a **division** algebra: no zero-divisors. Wrong algebra. |
| Visconti, *The Right to Be Zero-Knowledge Forgotten* | ARES 2024. ZK + GDPR. No Cayley–Dickson. |

Why this absence is informative rather than empty-handed:

1. The 84-pair sedenion ZD literature is well indexed (Moreno,
   Cawagas, de Marrais "box-kites"). A privacy application would
   cite those names.
2. Unlearning surveys 2022–2024 (`2209.02299` and siblings) classify
   exact vs approximate, shard vs influence vs certified. None of
   the abstracts retrieved mention hypercomplex annihilation.
3. Query S13 is the direct conjunction. It produced the rows above,
   not a forgetting paper.

Declared hole: a thesis or a 2026 workshop not on arXiv could
exist. That would move Q3 to PARTIAL. It was not found today.

## Q4 — Published reason not to do it

**PARTIAL.**

There is **no** paper that says "do not unlearn with sedenion
zero-divisors", because Q3 is empty. There **are** published facts
that kill the identity once it leaves the exact variety, and a
local measurement that kills the "product *is* the unlearn" reading.

### Published algebra — the useful kill

Moreno, *The zero divisors of the Cayley-Dickson algebras over the
real numbers*, *Boletín de la Sociedad Matemática Mexicana* (3) 4
(1998) 13–28; arXiv `q-alg/9710013`. For \(n \ge 4\),
\(\dim\ker L_a \equiv 0 \pmod 4\) and \(\ker L_a\) is a proper
subspace. A generic perturbation of \(a\) has trivial kernel.

Consequence, not a quotation: **exact annihilation is a property of
a thin variety**. Move \(A\) by \(10^{-8}\) in a transverse
direction and \(A\circ v\) is no longer zero. Measured here, IEEE
f64 Cayley–Dickson doubling:

| pair | \(\lVert A\circ v\rVert_2^2\) |
|---|---|
| \(A=e_3+e_{10}\), \(v=e_6-e_{15}\) | `0.0` (bit-exact; ±1 coordinates) |
| same \(A\), \(v=(e_6-e_{15})+2^{-52}e_0\) | `9.86e-32` (linear in the leak) |
| \(A'=A+10^{-8}e_2\), \(v=e_6-e_{15}\) | `2.00e-16` (\(\lVert\cdot\rVert_2 \approx 1.41\times 10^{-8}\)) |

So "floating point destroys the identity" is **the wrong slogan**
for the exact pair. The published reason not to *rely* on it is
Moreno's kernel dimension: the identity does not survive a
perturbation of the annihilator, nor a contribution that is not
exactly in the kernel.

### Published numerics — occupies the Woodbury cousin, not ZD

Bojanczyk, Brent, van Dooren, de Hoog, *A note on downdating the
Cholesky factorization*, *SIAM J. Sci. Stat. Comput.* 8 (1987).
One of three natural downdating algorithms is unstable; the others
are only mixed-stable. Higham, *Accuracy and Stability of Numerical
Algorithms*, SIAM 2002, preface to the 2nd ed.: updating/downdating
were omitted for space; later Higham (2021 note on
pseudo-orthogonal matrices) records that hyperbolic downdating
transforms can be arbitrarily ill-conditioned.

That is a published reason not to treat **ridge / OLS exact
unlearning** as bit-stable. It is not a paper about sedenions.

### Local, not published — product is the wrong map

`examples/zd_machine_unlearning.sio` already writes that
\(A\circ W\) is not \(W_{\mathrm{bob}}\). Measured:

\[
\lVert A\circ(e_0+0.5\,e_1) - (e_0+0.5\,e_1)\rVert_2^2 = 3.75.
\]

Left-multiplication by a zero-divisor **kills the kernel and
moves the complement**. Unlearning-as-indistinguishable-from-never-
trained wants a **projection**. The flagship example therefore
implements a projection and cites the ZD theorem only as a reason
to pick the subspace. That gap is already in-tree. It is not a
literature occupation of the quadrant.

The grok-cli4 forensic already records the same limitation for a
future `zd_annihilate` builtin: recognising the product is not a
proof that a contribution is zero.

## The conjunction

| Qualifier | Status | Occupied by |
|---|---|---|
| exact, no full retrain | **PARTIAL / occupied for restricted models** | Cao 2015; Cauwenberghs 2000; Woodbury; Ginart 2019; CMUMF 2023 |
| by algebraic annihilation (ZD / CD) | **FREE** | — |
| compile-time verification of *forgetting* | **FREE** (compile-time *DP/IFC* is occupied) | Fuzz family occupies a different property |
| all three | **FREE** | — |

**What a paper may still say, if it stays inside the evidence:**

- Lean proves a finite ZD census, not a trained-model forget.
- No prior paper was found that uses sedenion annihilators as an
  unlearning map.
- Compile-time privacy types exist; they check sensitivity or
  information flow, not erasure of a subject's contribution.
- Exact no-retrain unlearning already exists for linear / SQ / SVM
  / some clustering. Claiming novelty there is a correction waiting
  to happen.

**What a paper must not say:**

- "Machine unlearning is entirely approximate." False since at
  least Cao & Yang 2015 and Cauwenberghs & Poggio 2000.
- "`ExactlyPrivate<T>` is compile-time verification of forgetting."
  Today it is an effect gate.
- "Annihilation is a projection that leaves the complement
  untouched." False as a map; the in-tree example already recants.

## What this is not

- Not a patch to `self-hosted/`, `formal/`, or the example.
- Not a claim that Q3 cannot be occupied next month.
- Not a repeat of the grok-cli4 builtin forensic. That document is
  cited, not rewritten.
- Not an LLM-offload of a paper draft. This is an internal audit.
  A dissertation or submission that uses these sentences still owes
  the offload policy.

## Commands

```text
# arXiv title check, 2026-08-19
curl -sS 'https://export.arxiv.org/api/query?id_list=2209.02299,1912.03817,1911.03030,1907.05012,2411.18881,2210.09126'

# local product-vs-projection and Moreno-style perturbation (IEEE f64)
# Cayley–Dickson doubling; A = e3+e10
#   ||A*(e6-e15)||^2              = 0.0
#   ||A*(e0+0.5 e1) - (e0+0.5 e1)||^2 = 3.75
#   ||(A+1e-8 e2)*(e6-e15)||^2    = 2.00e-16
```
