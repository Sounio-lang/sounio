<!-- docs:meta
topic_id: repo.docs.research.prior-art-k3-signed-switching-2026-08-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.prior-art-k3-signed-switching-2026-08-04
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Prior-art audit — the "k = 3 phenomenon" is classical: switching classes, two-graphs, spectral moments

**Date:** 2026-08-04 · **Mode:** `deep-research` / lit-review (prior-art scan) · **Scope:** the §35–§37
finding that a seam and its Fano reference are cospectral as *unsigned* graphs at every power, agree
as *signed* graphs at `k ≤ 3` up to a universal constant, and diverge label-dependently at `k ≥ 4`.

**Verdict: the register is not novel. The universality is what remains candidate-novel.**

---

## 0. What was asked, in field-neutral terms

> Two signed graphs on isomorphic underlying graphs; the underlying graphs are cospectral at every
> power; the signed adjacency traces agree at `k ≤ 3` up to a constant that does not depend on a
> label parameter, and disagree at `k ≥ 4` in a way that does. The "sign defect" between them is a
> `±1` edge function, balanced in exactly one of two regimes.

Stated that way the literature answers most of it immediately.

## 1. Method and its limits

Six web searches; three sources fetched and **verified by extracting the PDF text locally** rather
than trusting search-engine summaries. No MathSciNet/zbMATH access; US-only web search; several
relevant items sit behind paywalls (ScienceDirect, Springer) and were **not** read. Every claim
below is tied to text I actually read. Anything I could not verify is marked as such and is *not*
cited as support. Absence of a hit here is weak evidence, especially for §4.

## 2. Verified findings

### 2.1 `tr(Aᵏ)` of a signed graph counts *signed* closed walks — so `k = 3` is the first informative moment

> **Theorem 2.2 (Spectral Moments).** Let `Γ` be a signed graph with eigenvalues `λ₁ ≥ … ≥ λₙ`. If
> `W±ₖ` denotes the difference between the number of positive and negative closed walks of length
> `k`, then `W±ₖ = Σᵢ λᵢᵏ`.
> — Belardo, Cioabă, Koolen & Wang, *Open problems in the spectral theory of signed graphs*,
> arXiv:1907.04349, §2 (attributed there to Zaslavsky)

Reading the ladder off that theorem:

| `k` | `tr(Aᵏ)` is | consequence for our pair |
|---|---|---|
| 1 | `0` | vacuous |
| 2 | closed 2-walks, each of sign `σ(e)² = +1` ⟹ `2·\|E\|` | **depends only on the underlying graph** |
| 3 | `6(t⁺ − t⁻)`, signed triangles | **first moment that sees the signature at all** |
| ≥4 | signed closed `k`-walks | no reason to agree unless the signatures are switching-equivalent |

> ⚠ **Correction to §35.** Its table lists `k = 2` agreement alongside `k = 3` as if both were data
> about the signature. `k = 2` is **forced** by §37's unsigned isomorphism `Φ = τ_j` and carries no
> information about signs. And `k = 3` being the register is not a razor-thin coincidence — it is
> the *first* moment that can carry any. §35 presented both as surprising.

### 2.2 The "sign defect" `ε` is *switching*, by name, and its balance is Harary's theorem

> **Switching.** Switching `Σ` … in terms of a function `θ : V → {+,−}`, called a *switching
> function*. Switching `Σ` by `θ` means changing `σ` to `σ^θ` defined by `σ^θ(vw) := θ(v)σ(vw)θ(w)`.
> — Zaslavsky, *Matrices in the Theory of Signed Simple Graphs*, arXiv:1303.3083, §I.G

That is verbatim §37's `ε(a,b) = μ(a)μ(b)`. And:

> **Theorem I.2 (Harary's Balance Theorem).** A necessary and sufficient condition for `Σ` to be
> balanced is that there be a bipartition of `V` into `X` and `Y` such that an edge is negative
> precisely when it has one endpoint in `X` and one in `Y`.
> — ibid., §I.F–I.G

> It is not difficult to see that each cycle in `Γ` maintains its sign after a switching. Hence `Γ^U`
> and `Γ` have the same positive and negative cycles. Therefore, **the signature is determined up to
> equivalence by the set of positive cycles**. … **Switching isomorphic signed graphs are
> cospectral**, and their matrices are signed-permutationally similar.
> — Belardo et al., §2

So §37.2's dichotomy has a standard name:

> **The seam and its Fano reference are switching-equivalent exactly when `popcount(g)` is odd.**

and §35's clause `C1` — that (c)'s pair agrees at *every* `k` — is the textbook consequence of
switching-equivalence being a similarity. Not an empirical discovery.

### 2.3 The triangle data is a *complete* invariant of the switching class — Seidel's two-graphs

> A two-graph consists of a finite set provided with a collection of triples, called *coherent*, such
> that each 4-set contains an even number of coherent triples. … Given any graph on vertex set `V`, a
> two-graph arises by taking all triples containing an odd number of edges. Importantly, **two graphs
> are switching equivalent precisely when they give rise to the same two-graph.** … If `Γ` and `Δ` are
> switching equivalent, their Seidel adjacency matrices have the same spectrum.
> — A. E. Brouwer, *Two-graphs*, <https://aeb.win.tue.nl/graphs/twographs.html> (attributions there:
> J. J. Seidel for switching classes; G. Higman for regular two-graphs)

Primary sources named in Zaslavsky's bibliography: J. J. Seidel, *Linear Algebra Appl.* **1** (1968),
281–298; and *A survey of two-graphs*, Colloq. Int. sulle Teorie Combinatorie (Rome 1973), Accad. Naz.
Lincei, Rome, 1976, 481–511.

**So the whole "k = 3 register" question is answered by 1968–76 theory**: the switching class of a
signature *is* its triple system, so the triangle signs are exactly the invariant, and no lower moment
can see the signature while higher ones carry more than the class.

### 2.4 §39.3's sign formula is *antibalance*, and the one-line trace follows

`A_σ(a,b) = −μ(a)μ(b)` says `−A_σ` has a switching function, i.e. `−A_σ` is balanced — the standard
term is that `A_σ` is **antibalanced** (Zaslavsky uses "antibalanced" throughout §I.F). Then every
triangle of `A_σ` is negative and `tr(A³) = −#triangles` is the routine consequence, not a bespoke
derivation. The finding survives; its billing does not.

## 3. What the literature does **not** appear to contain

Reported as *searched-and-not-found*, which is weaker than *absent*:

1. **The universality.** I found no treatment of a *family* of switching-class pairs whose
   signed-triangle difference is **constant across a label parameter**. Switching-class differences
   are studied one pair at a time (frustration index, negative-cycle vectors); a parametrised family
   with a closed form is not something I located.
2. **`[j,3]₂` as a signed-triangle count.** No hit connecting Gaussian binomial coefficients to
   negative-triangle counts or switching-class differences. The two literatures — `q`-binomials and
   signed-graph switching — did not co-occur in any result I saw.
3. **Signed spectra of Cayley–Dickson zero-divisor graphs.** — *superseded, see §3A. The full texts
   were obtained and this absence is now verified, but so is a positive hit that matters more.*

## 3A. The Cayley–Dickson relation-graph literature, read (added 2026-08-04, second pass)

The Springer paywall is not the only route: the **Russian originals are open on mathnet.ru**. I
obtained and text-extracted both, and checked the bodies with **Russian** keywords, not only English.

### 3A.1 What is published

| work | content |
|---|---|
| **Guterman & Zhilina**, *Relation graphs of the sedenion algebra*, Zap. Nauchn. Sem. POMI **496** (2020) 61–86 (= *J. Math. Sci.* 2021, DOI 10.1007/s10958-021-05367-6) | orthogonality graph `Γ_O(𝕊)`; every pair of zero divisors produces a **double hexagon**; explicit vertex set of an arbitrary connected component and its **diameter**; **a bijection between the connected components of `Γ_O(𝕊)` and the lines in the imaginary part of the octonions**; commutativity graph has one big component, diameter in `[3,4]` |
| **Guterman & Zhilina**, *Cayley–Dickson split-algebras: doubly alternative zero divisors and relation graphs*, Fundam. Prikl. Mat. **23**:3 (2020) 95–129 (= *J. Math. Sci.* **269**:3 (2023) 331–355) | annihilators and orthogonalizers of doubly alternative zero divisors; centralizer↔orthogonalizer relation; a real-Jordan-form analogue for split-octonions; commutativity, orthogonality and zero-divisor graphs of split-ℂ, split-ℍ, split-𝕆 **via diameters and cliques** |
| **Zhilina**, *Orthogonality graphs of real Cayley–Dickson algebras. Part I*, Int. J. Algebra Comput. **31**(4) (2021) 663 | for the algebras of the **main sequence the zero-divisor graph coincides with the orthogonality graph**; hexagons extend to double hexagons (located via search; abstract only) |
| **Moreno**, *The zero divisors of the Cayley–Dickson algebras over the real numbers*, Bol. Soc. Mat. Mex. (1998), arXiv:q-alg/9710013 | algebraic description of the zero divisors; `dim Ker L_a ≡ 0 (mod 4)` and `dim Ker L_a ≤ 2^n − 4`. **The word "graph" does not occur in the paper** (0 hits over 31 pages) |

### 3A.2 The verified negative

Keyword sweep over the **full texts** (English abstracts + Russian bodies): `спектр`/`spectr` — **0
hits in both Guterman–Zhilina papers**; no `signed`/`знак`-as-signature usage; no signed adjacency
matrix anywhere. Their graphs are **unsigned**, and the results are structural (components,
diameter, cliques, hexagons), not spectral.

> **So the lane's *signed* layer — `A_σ`, its spectrum, `tr(A²)`, `tr(A³)`, switching classes — is
> not touched by this literature.** That absence is now verified on the sources, not inferred from
> abstracts.

### 3A.3 The hit that matters, and it is uncomfortable

> "Then we establish a bijection between the connected components of `Γ_O(𝕊)` and **the lines in the
> imaginary part of the octonions**."
> — Guterman & Zhilina (2020), abstract, verbatim

The lines in `Im 𝕆` are the seven lines of the **Fano plane**. So *the Fano structure of the
sedenion zero-divisor/orthogonality graph is published prior art (2020)*, at `n = 4`.

**This is a second firewall, alongside the Kirshtein one on the "168".** Consequences:

- any statement of the lane's Fano/7-line structure **at the sedenion level** must cite
  Guterman–Zhilina (2020), not be presented as found here;
- the lane's `n = 4` base case is, structurally, their theorem;
- what is **not** theirs, on present evidence: the ∀`n` statements, the signed layer and its traces,
  the fibre/seam classification, and everything downstream of it.

### 3A.4 One check the lane now owes

Zhilina's "for the main sequence the zero-divisor graph coincides with the orthogonality graph"
raises a concrete question: **is the lane's "annihilation graph" the orthogonality graph restricted
to basis-indexed elements?** If it is, the *support* of `A_σ` is a named, studied object and should
be cited as such — leaving the signature as the contribution. This is a definitional comparison, not
a measurement, and it should be done before any write-up.

### 3A.5 Limits of this pass

I read the two abstracts verbatim and ran keyword sweeps over the extracted full texts. I did **not**
read the Russian bodies in detail, so I can state what these papers are *about* and what they do
*not* use, but not their theorem statements beyond the abstracts. The IJAC Part I paper was not
obtained. A proper reading of all three, in Russian, is still owed.

## 4. Consequences for the lane

**Deflations (act on these):**

- §35's "no proof can go through a graph isomorphism, switching equivalence, or spectral identity"
  is *correct* but for a classical reason, and its supporting table over-reads `k = 2`. The `k ≥ 4`
  failure is not evidence of anything exotic: it is what one expects once the two signatures are in
  different switching classes.
- §37's `ε`, its balance dichotomy, and §39.3's `μ` should be stated in the standard vocabulary —
  *switching function*, *balanced*, *antibalanced*, *two-graph* — with the 1968–76 attributions.
  Presenting them as new objects would not survive a referee.

**What is strengthened:**

- The right frame for (III)'s open core is now nameable: `δ(n,j)` is the difference of the two
  **two-graphs'** coherent-triple counts, and the open claim is that this difference is independent
  of `g`. That is a sharper and more standard statement than "the deviation ignores `g`".
- §39.3 gains a one-line classical justification (antibalance ⟹ all triangles negative ⟹
  `tr(A³) = −#triangles`), which is exactly the step the lane wanted to formalise.

**Candidate-novel, pending a proper database search:** the `g`-universality and the `[j,3]₂` closed
form. Both should be re-checked against MathSciNet/zbMATH before any claim. The CD relation-graph
literature has now been read to the extent possible (§3A): it does **not** contain the signed layer,
but it **does** contain the Fano-line structure of the sedenion orthogonality graph
(Guterman–Zhilina 2020), which is a second firewall next to Kirshtein 2012.

## 5. Limitations

Six searches; three sources verified; no bibliographic-database access; paywalled CD-graph papers
unread. This is a scan, not a systematic review — no PRISMA flow, no screening protocol, no second
screener. The three "not found" results in §3 are *absence of located evidence* and should not be
cited as novelty.

## References

- Belardo, F., Cioabă, S. M., Koolen, J., & Wang, J. (2019). *Open problems in the spectral theory of
  signed graphs*. arXiv:1907.04349. <https://arxiv.org/pdf/1907.04349> — verified (text extracted).
- Brouwer, A. E. *Two-graphs*. <https://aeb.win.tue.nl/graphs/twographs.html> — verified (fetched).
- Zaslavsky, T. (2013). *Matrices in the Theory of Signed Simple Graphs*. arXiv:1303.3083.
  <https://arxiv.org/pdf/1303.3083> — verified (text extracted).
- Seidel, J. J. (1968). *Linear Algebra and its Applications*, 1, 281–298. — cited **via** Zaslavsky's
  bibliography; not read directly.
- Seidel, J. J. (1976). *A survey of two-graphs*. Accad. Naz. Lincei, 481–511. — cited **via**
  Zaslavsky's bibliography; not read directly.
- Guterman, A. E., & Zhilina, S. A. (2020). *Relation graphs of the sedenion algebra*. Zap. Nauchn.
  Sem. POMI, 496, 61–86; *J. Math. Sci.* (2021), DOI 10.1007/s10958-021-05367-6.
  <https://www.mathnet.ru/eng/znsl7014> — abstract verified verbatim; full text extracted and
  keyword-swept (RU+EN); body not read in detail.
- Guterman, A. E., & Zhilina, S. A. (2020). *Cayley–Dickson split-algebras: doubly alternative zero
  divisors and relation graphs*. Fundam. Prikl. Mat., 23(3), 95–129; *J. Math. Sci.*, 269(3) (2023),
  331–355. <https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=fpm&paperid=1900> — same.
- Zhilina, S. A. (2021). *Orthogonality graphs of real Cayley–Dickson algebras. Part I: Doubly
  alternative zero divisors and their hexagons*. Int. J. Algebra Comput., 31(4), 663. — located via
  search; **abstract only, not obtained**.
- Moreno, G. (1998). *The zero divisors of the Cayley–Dickson algebras over the real numbers*. Bol.
  Soc. Mat. Mex.; arXiv:q-alg/9710013. — verified (text extracted); contains **no** graph structure.
- Moreno, G. *Large annihilators in Cayley–Dickson algebras II*. arXiv:math/0702075. — located, not
  read.

*AI disclosure: this scan was produced with AI assistance (Claude Opus 5); every quoted passage was
extracted from the source PDF/page and checked against it, and unverified items are marked.*
