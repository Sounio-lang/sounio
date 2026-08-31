<!-- docs:meta
topic_id: repo.docs.research.paper-a-section1-draft-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-section1-draft-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — §1 *Introduction* (full draft, 2026-08-25)

> Full-prose introduction, replacing the skeleton in `paper_A_antigarbling_skeleton`.
> The abstract stays in the skeleton file.

---

## 1. Introduction

A measurement is never a number; it is a number and a doubt. Scientific and engineering
software that takes measurements seriously has, over the last decade, learned to keep the
doubt attached: *uncertainty-typed* languages and libraries — `Uncertain⟨T⟩`,
`Measurements.jl`, GUM tooling — give a quantity a type like `Knowledge⟨f64⟩` that carries
its variance, and propagate that variance automatically through arithmetic, so that a result
arrives already annotated with how much it can be trusted. The promise is that uncertainty
becomes a first-class, compiler-tracked property rather than a comment in a lab notebook.

That promise rests on a hypothesis the type never states and the compiler never checks: that
the operands of every operation are **statistically independent**. The propagation laws these
systems implement — `Var(X+Y) = Var(X) + Var(Y)`, `Var(XY) ≈ Y²Var(X) + X²Var(Y)` — are the
independence special cases of the general laws, which carry an additional covariance term
`±2Cov(X,Y)`. When the operands are correlated, the covariance term is real and the shipped
formula omits it. For addition and multiplication the omission runs in one direction: the
reported variance is *smaller* than the truth. The program claims more precision than its
inputs justify.

This failure is uniquely resistant to discovery, for a reason that has nothing to do with
how carefully the code is written. An understated variance produces a **tighter error bar** —
a more confident, more precise-looking answer. It is the one kind of wrong result that never
looks wrong. A test asserting the value is correct passes; a test asserting the uncertainty
is not too large passes; only a test written by someone who already knows the correct, larger
variance can fail — that is, only someone who has already found the bug. The defect hides in
exactly the blind spot testing cannot cover, because the symptom is indistinguishable from
success.

**It is not hypothetical.** In a production uncertainty library we examined, the operation
`x · x` has two implementations that disagree. Multiplying a value by itself through the
general product operator returns a variance of `2m²v`; squaring the same value through the
dedicated square operator returns `4m²v` — twice as much, and correct. The two are the same
mathematical operation; both ship; nothing in the library routes `x · x` to the sound one.
The discrepancy is precisely the covariance term the product operator drops, which for
`x · x` is not negligible but equal to the entire result. A programmer who writes `x * x`
gets, silently, half the uncertainty they are owed. The same omission makes correlated
addition understate and — revealingly — makes correlated *subtraction* overstate, an
asymmetry we show is the fingerprint of the underlying error rather than random sloppiness.

**Our thesis is that manufacturing precision is a type error, and it is the same error as
manufacturing information.** Quantitative information flow has a mature theory of when a
computation is allowed to change what is known: the Blackwell / data-processing order, in
which an admissible transformation is a *garbling* — it may lose information but never create
it. Understating variance is the forbidden move, an **anti-garbling**: it reports less
uncertainty, hence more information, than the operands contain. Read this way, the library's
`x + x` is not merely inaccurate; it violates a conservation law. And conservation laws are
what type systems are good at enforcing.

We enforce it. The reason the library cannot condition on `Cov(X,Y)` is that its scalar
representation `(mean, variance)` cannot express which measurements a value depends on. We
restore that information to the type. Borrowing the *noise-symbol* representation of affine
arithmetic — where a value is an affine form over independent per-source symbols and two
values are correlated exactly when they share a symbol — we carry each value's **source-set**
in its type, `Knowledge⟨T, N⟩`. An independence-assuming operator is then well-typed only
when its operands' source-sets are provably disjoint; otherwise it is a compile-time error.
The programmer discharges it either with a proof that the operands are uncorrelated or by
switching to an operator that takes the correlation explicitly. The unsound path does not
type-check.

**What we claim, and what we do not.** Neither half of the mechanism is new on its own, and
we are explicit about this up front so the contribution is not mistaken for either piece.
Tracking noise-symbol identity to preserve correlations is the defining idea of affine
arithmetic (Comba & Stolfi 1993) and of the Fluctuat static analyzer (Goubault & Putot) — but
there it lives in an external analyzer that *computes an enclosure* and rejects nothing. The
Blackwell/garbling soundness order is standard in quantitative information flow (McIver,
Morgan & Smith; Alvim et al.) — but there it orders confidentiality channels and measures
leakage. Our contribution is their **intersection**, which neither field occupies: the
source-set lifted *into the type* of an uncertainty-typed language, and the independence
assumption of uncertainty arithmetic made a *checked precondition* whose violation is a type
error with a discharging certificate. We track source identity like affine arithmetic, forbid
anti-garbling like QIF, and enforce it in the type like neither.

**Contributions.**

1. **A characterization of the defect class** — independence assumed and unchecked in
   uncertainty propagation — grounded in a shipping library, together with its framing as a
   Blackwell anti-garbling, including the add/sub directional signature that the framing
   predicts and the source confirms (§2, §4).
2. **A type discipline** that carries the noise-symbol source-set in the type and makes the
   independence assumption a checked precondition: the E230 rejection rule, the noise-set
   lattice and its transfer functions, and the proved-disjoint / correlation-aware escape
   valves (§5).
3. **A soundness result** — no well-typed program contains a first-order anti-garbling —
   whose two load-bearing lemmas (the covariance-exactness criterion and the base calculus's
   type safety) are machine-checked in Lean, with the remaining step scoped honestly to the
   pending checker extension (§6).
4. **An implementation and evaluation** in the self-hosted Sounio compiler: running
   prototypes and a kernel-checked model that reproduce the defect, establish that the
   rejection is *caused* by source-set propagation (a sabotage-controlled experiment), and a
   clinical case study — vancomycin therapeutic-drug-monitoring — where understatement
   silences the safety warning the system exists to raise (§7, §8).

The guarantee is deliberately narrow — first-order, conservative, in the variance channel —
and §10 states its boundaries plainly. Within that scope it eliminates exactly the defect
class of §2: the answer that looks too precise to be true becomes the one the compiler
refuses to print.
