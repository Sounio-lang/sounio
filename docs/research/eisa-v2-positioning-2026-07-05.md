<!-- docs:meta
topic_id: repo.docs.research.eisa-v2-positioning-2026-07-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.eisa-v2-positioning-2026-07-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# EISA v2 — scientific positioning: a deterministic, receipt-carried roundoff lane against the prior art (2026-07-05)

Status: W5 positioning revision after adversarial review; adopted on
`origin/gpu/epistemic-tensor-core-next` at `059062c2c` and imported into the
Madaros v2 canon lane as historical research context.
Lineage: positions the stack defined in `eisa-stack-architecture-2026-07-05.md`
(format, Metron VM, conformance bridge), `eisa-v1-plan-2026-07-05.md`
(branches, fuel, frail counting, I6) and `eisa-v2-arch-2026-07-05.md`
(qd128 err lane, closure contract, receipt v3, dd64-failure lane).
Naming follows decision #14: the executor is the **Metron VM (MVM)** in
external-facing text; internal identifiers (`eisa::evm`, `.eisax`) are unchanged.

## 1. Thesis

EISA v2 occupies a design point we fix by three explicit criteria: **(C1)** a
mandatory, per-operation, EFT-measured roundoff-correction lane the machine
model cannot bypass; **(C2)** a separate first-order GUM uncertainty lane; and
**(C3)** an observation-gate audit receipt that cites the hash of the program
that produced it. No system meeting C1–C3 *together* appears among the
traditions and references surveyed in §7; to our knowledge EISA v2 is the
first *executable format* to combine all three, and §6.8 states the criteria
under which that bounded claim is falsifiable. Each register carries three
lanes: `val` (the bit-exact IEEE-754 double the plain hardware computes),
`err` (a measured correction — double-double in v1, quad-double in v2 —
defined by the closure contract `true(z) = round_qd(true(x) op true(y))`),
and `u` (first-order GUM uncertainty, JCGM 100:2008). There is no fast path
that forgets the error: propagation is a property of the format's semantics,
enforced identically by three executors (reference chains, the Metron VM,
and an x86-64 conformance bridge) that must agree byte-for-byte on receipts.
The concrete evidence is the Rump (1988) receipt: on the flagship v2 corpus
lane the final register's `val` lane is floating-point debris (−2⁷⁰ in that
evaluation order), while `val + err` reconstructs the exact value
−54767/66192 to quad-double rounding, on derived bits, with the dd64 twin
lane kept as a mandatory *failure* witness at the same ~122-bit cancellation
boundary. Everything else in this document is the honest comparison of that
design point against five prior traditions, including where each of them is
plainly stronger than EISA today.

The bounded "first" claim is a *negative* result over that survey: no
surveyed tradition meets the conjunction C1 ∧ C2 ∧ C3. The first criterion
each fails (a per-tradition exclusion, so the claim is checkable rather than
merely asserted):

| Tradition (§) | First criterion failed | Why |
|---|---|---|
| DSA / CADNA (§2.1) | C1 | roundoff is a *statistical* significant-digit estimate from random-rounding samples, not a deterministic per-operation EFT-measured correction |
| MCA — Verrou / Verificarlo (§2.2) | C1 | stochastic rounding yields a *distribution* over runs, not a per-operation deterministic correction lane |
| Static — Herbie / FPTaylor (§2.3) | C1 | a-priori rewriting / Taylor-form bounds; nothing is carried per operation in the running executable (these are not executable formats) |
| Interval / affine / ball (§2.4) | C2 | a single rigorous enclosure conflates rounding with input uncertainty — there is no *separate* first-order GUM lane |
| Extended precision — dd / qd / MPFR (§2.5) | C1 | more working precision *replaces* the value; there is no roundoff-*correction* lane against a bit-exact IEEE `val`, nor a GUM lane |

C3 (a receipt citing the program-hash provenance of the result) is met by
none of the surveyed systems, so it is an independent second discriminator.
The claim retires the moment a reader exhibits a system — surveyed or not —
meeting all three.

## 2. The comparison landscape

### 2.1 Discrete Stochastic Arithmetic — CADNA

CADNA (Jézéquel & Chesneaux 2008) implements Discrete Stochastic Arithmetic
(Vignes 2004), built on the CESTAC method: each floating-point result is
computed N times (N = 3 in practice) under random rounding, and the number
of decimal digits common to the samples estimates the number of exact
significant digits, via a Student's-t argument at a 95% confidence level
(β = 0.05, τ_β ≈ 4.4303 for N = 3). A result whose samples share no
significant digits is a "computational zero". CADNA additionally detects
numerical anomalies at run time — including *unstable branchings*, flagged
when the difference between the two comparison operands is a computational
zero — and dynamically checks the validity of the first-order model that
CESTAC relies on (terms in 2^(−2p) are neglected; multiplications and
divisions are monitored to keep the neglect legitimate).

What it guarantees: a probabilistic estimate of the number of exact
significant digits of every intermediate, plus run-time detection of a
taxonomy of instabilities (branching, cancellation, unstable intrinsics),
under a first-order model whose hypotheses are themselves checked during
the run. What it costs: source-level instrumentation (stochastic types
replace floating-point declarations; the `cadnaizer` tool automates part of
this), at least a 3× slowdown from the synchronous samples, and the
guarantee is statistical, not per-execution-deterministic.

What EISA does differently: the err lane is a *deterministic, measured*
quantity — the EFT residual of each operation combined under the v2 closure
contract — not an estimate over rounding samples. The same image produces
the same receipts, bit-for-bit, on every conforming executor; a receipt is
therefore reproducible evidence, not a summary of one randomised run. And
the detection artefact lives in the executable format itself (receipts cite
the program hash), not in a diagnostic side-channel file.

What CADNA does better, plainly: it makes **no exactness assumptions**. EISA's
err lane is only as good as its EFTs, which require strict IEEE round-to-nearest
with no FMA contraction and no reassociation, plus the Priest renormalisation
assumption inherited from the qd literature (§6). CADNA's random-rounding
disagreement signal survives all of that — it needs no EFT to be exact and no
expansion to stay non-overlapping. CADNA is also mature software, applied to
large industrial C/C++/Fortran codes (and CUDA) for over two decades; EISA is
a 64-register research VM with a conformance corpus.

### 2.2 Monte Carlo Arithmetic — Verificarlo and Verrou

Verificarlo (Denis, de Oliveira Castro & Petit 2016) instruments
floating-point operations at the LLVM intermediate-representation level,
replacing IEEE-754 operations with Monte Carlo Arithmetic counterparts so
that executions become trials of a Monte Carlo simulation; statistics over
runs estimate significant digits at a configurable virtual precision.
Because instrumentation happens after the front-end and middle-end, it is
transparent to the user (no source changes, C/C++/Fortran) and — unlike
source-to-source approaches — captures the numerical effect of compiler
optimisations. Verrou (Févotte & Lathuilière 2016) pushes the entry cost
lower still: Valgrind-based dynamic binary instrumentation perturbs the
rounding mode of each floating-point instruction in the *unmodified,
already-compiled* binary — an asynchronous CESTAC variant, or equivalently
a random-rounding subset of MCA — and has been applied to industrial codes
at EDF (code_aster; Févotte & Lathuilière 2017) and to HEP tracking software
(Grasland et al. 2019).

What they guarantee: a statistical picture of how rounding perturbations
propagate to outputs, over the *real* code, at production scale, including
whatever the compiler actually emitted. What they cost: multiple runs for
sampling; the answer is a distribution, not a per-operation number; and the
diagnosis is attached to the run, not to the artefact that ran.

What EISA does differently: one execution, one deterministic answer — the
err lane is a point measurement of the accumulated residual, per register,
per operation, available at every gate and serialised into the receipt with
program provenance. MCA answers "how unstable is this code under rounding
perturbation?"; EISA's err lane answers "what is the measured correction
between what this execution computed and the closure-contract true value?"
— per execution, with no sampling variance.

What MCA does better, plainly: **it scales to real HPC codes today**. Verrou
needs neither source nor recompilation; Verificarlo needs only a recompile.
Both inherit the entire existing toolchain. EISA requires the computation to
be expressed in a 64-register, 256-instruction `.eisax` image — hand-lowered
or compiled from the minimal Metron surface — which today admits kernels,
not applications. MCA also probes the code *as optimised by the compiler*,
a surface EISA deliberately fixes instead (the val lane is defined to be
plain SSE2 scalar IEEE semantics; there is nothing to probe). To be exact
about what "deterministic" claims here: the conformance bridge chooses its
*own* fixed SSE2-scalar, non-FMA lowering and is **not** required to
reproduce whatever an arbitrary optimising compiler emits. Determinism in
this document means the three executors — the reference chains, the Metron
VM, and that one fixed bridge lowering — produce byte-identical receipts for
a given `.eisax` image; it is not a claim about "whatever a compiler emitted"
(which is the surface MCA probes and EISA fixes).

### 2.3 Static tools — Herbie and FPTaylor

Herbie (Panchekha, Sanchez-Stern, Wilcox & Tatlock, PLDI 2015) automatically
rewrites floating-point expressions into more accurate equivalents, using
sampled-point error estimation, a database of algebraic rules, series
expansions and regime inference; it improved textbook examples by up to
60 bits. FPTaylor (Solovyev, Jacobsen, Rakamarić & Gopalakrishnan, FM 2015;
journal version Solovyev et al., TOPLAS 2018) computes *rigorous worst-case
round-off error bounds* via Symbolic Taylor Expansions and rigorous global
optimisation, handles transcendental functions, and emits per-instance
certificates as machine-checkable HOL Light proofs.

These tools operate on a different axis: static (pre-execution) versus
EISA's runtime measurement. Where static analysis is stronger, and it
genuinely is: FPTaylor's bound is an *a priori proof* over the whole input
domain — no execution, no runtime cost, machine-checkable — where EISA's
receipt certifies one execution on one input. Herbie removes the error
rather than measuring it, which is strictly more useful when a rewrite
exists. If the question is "is this kernel accurate for all inputs in this
box?", FPTaylor answers it and EISA does not.

Where EISA differs: the err lane is *measured on the actual execution*, so
it is input-exact rather than domain-worst-case (worst-case bounds on
Rump-class cancellations are uselessly wide precisely where the receipt is
bit-exact); it composes across control flow taken at run time (loops with
data-dependent trip counts, the fixed-point kernel of §4) without needing
loop invariants; and the result is carried in the execution artefact itself
— receipt plus program hash — rather than in an offline report. The two
axes are complementary, not competing: nothing in EISA prevents running
FPTaylor on a Metron kernel ahead of time.

### 2.4 Enclosure arithmetics — interval, affine, ball

The "carry a rigorous enclosure" tradition runs from Moore's interval
arithmetic through affine arithmetic (de Figueiredo & Stolfi 2004), which
tracks first-order correlations between quantities to fight interval
arithmetic's dependency problem, to arbitrary-precision midpoint-radius
("ball") arithmetic in Arb (Johansson 2017), and to verification methods
built on floating-point and directed rounding generally (Rump 2010). The
guarantee is containment: the true value provably lies inside the enclosure.
That is a stronger *kind* of statement than anything EISA emits.

The contrast is semantic and must be stated precisely: **EISA's err lane is
a measured correction, not an enclosure.** It is a point estimate of the
exact residual, computed through EFTs under the closure contract — exact to
quad-double rounding for add/sub/mul and correctly rounded for div/sqrt by
the Hida–Li–Bailey algorithm guarantees, *given* the EFT preconditions.
There is no interval, no radius, and no containment theorem. Input
uncertainty is deliberately not folded into err either: the u lane carries
first-order GUM propagation separately, because "how wrong is the float"
and "how uncertain is the measurement" are different epistemic quantities
and collapsing them is how enclosures become uninformatively wide.

What enclosures do better, plainly: rigour. An interval or ball result is a
theorem about the true value; an EISA receipt is a measurement whose
exactness rests on the EFT/no-FMA/renormalisation assumptions of §6.
Enclosure libraries (INTLAB, Arb) are also mature, complete (special
functions, linear algebra, series) and production-grade. What they cost:
overestimation under dependency (interval), first-order-only correlation
tracking (affine), and — common to all — the enclosure describes a set,
not the specific rounding history of the specific execution that ran.

### 2.5 High-precision substrates — dd/qd/MPFR as libraries versus machine semantics

The technical substrate of EISA's err lane is the double-double/quad-double
lineage: Dekker (1971) exact addition and multiplication of doubles;
Priest (1991) arbitrary-precision expansions with proven renormalisation;
Shewchuk (1997) adaptive expansions for robust geometric predicates;
Hida, Li & Bailey (2001) — the qd library — packaging quad-double with
renormalisation, accurate addition and division/sqrt algorithms. MPFR
(Fousse et al. 2007) provides correctly rounded arbitrary precision as a
library. The EFT canon is consolidated in Muller et al. (2018).

EISA adds no new arithmetic algorithm to this lineage and claims none: the
v2 err lane is Hida–Li–Bailey quad-double, non-FMA variants, implemented in
Sounio (`stdlib/math/qd128.sio`). The difference is *where the substrate
sits*. As libraries, dd/qd/MPFR are opt-in: the programmer chooses which
variables get the precision, the choice is invisible in the shipped binary,
and nothing in the artefact records what was measured. In EISA the same
algorithms are the mandatory error semantics of the machine: every
arithmetic instruction of every v2 image propagates the qd correction, the
format version word selects the depth (dd64 for v0/v1, qd128 for v2), and
the receipt exposes all four err components with the program hash. A
library gives you precision where you asked for it; the format gives you a
measured roundoff trail whether or not you thought to ask.

What the libraries do better, plainly: performance and generality. qd and
MPFR are optimised native code used in production scientific computing;
EISA's interpreted qd steps are an *estimated* ~20–50× slower than its own
v1 steps (an order-of-magnitude projection from the qd operation counts in
`eisa-v2-arch-2026-07-05.md §8`, not a benchmarked figure) and the
instruction set is nine arithmetic-relevant opcodes plus branches.

## 3. Comparison table

| | DSA / CADNA | MCA (Verificarlo / Verrou) | Herbie | FPTaylor | Interval / affine / ball | dd/qd/MPFR (libraries) | EISA v2 (MVM) |
|---|---|---|---|---|---|---|---|
| Detection semantics | statistical disagreement across N=3 random-rounding samples | statistical, over sampled perturbed runs | static rewrite search (sampled error estimate) | static worst-case bound, proved | containment enclosure | none (substrate only) | deterministic EFT-measured residual per op |
| Determinism | no (random rounding; estimate at 95% confidence) | no (sampling) | n/a (offline) | yes (a priori proof) | yes (enclosure) | yes | yes (same image ⇒ same receipts, bit-exact) |
| Per-op residual | no (digit-count estimate per value) | no (run-level statistics) | no | no (domain-level bound) | no (set, not residual) | possible, manual | yes (err lane, qd128) |
| Uncertainty lane (input uncertainty ≠ roundoff) | partial (`data_st` perturbs inputs into the same stochastic estimate) | no | no | no | folded into the enclosure | no | yes (separate GUM u lane) |
| Executable-format provenance | no (diagnostic file) | no (run report) | no | proof certificate, separate from binary | no | no | yes (receipts cite EISA-hash of the image) |
| Branch instability handling | yes (unstable branching on computational-zero operand difference) | indirect (output divergence across runs) | no | no (straight-line expressions) | possible via enclosure overlap, manual | no | yes (frail counter, count-only, receipt-visible) |
| Scale today | large industrial codes, decades of use | full HPC applications, no source change | expressions | kernels / expressions, proofs | production libraries | production libraries | 64-register VM, conformance corpus of kernels |

Each cell honestly summarises §2; the last row is the one that keeps this
document a positioning draft rather than a superiority claim.

## 4. Frail branches — instability at the decision point

The reviewed one-line framing (math-review, 2026-07-05, adopted verbatim as
the basis of this section): *EISA deterministically measures a post-facto
residual bound via EFTs and flags the branch when the decision occurs
inside that bound; CADNA/DSA (Vignes 2004) detects the same instability
class via disagreement across random-rounding samples. Same failure class
flagged, different detection semantics.*

Mechanically: a v1+ branch (`ebrz`/`ebrn`) decides on the `val` lane alone;
the frail test is the exact Sounio predicate (`stdlib/eisa/evm.sio`), with no
unstated total order over the mixed lanes:

```
fn branch_frail_predicate(val: f64, ehi: f64, u: f64) -> bool {
    let band = if u > abs_f64_local(ehi) { u } else { abs_f64_local(ehi) }
    if band == 0.0 { return false }
    abs_f64_local(val) <= band
}
```

The `max` is a plain f64 comparison of the declared uncertainty `u` against
`|err.x0|` (the err leading component, passed as `ehi`); the exact case
(`band == 0.0`) returns early so an exact operand is never frail; and the
decision is a single f64 `<=`. When the predicate holds the branch direction
is epistemically unsupported, and a per-run counter `frail_branches`
increments. CADNA's analogue is the
*unstable branching* anomaly: the difference between the two comparison
operands is a computational zero, i.e. the samples cannot agree that the
comparison has a definite outcome. The correspondence is close but not an
equivalence: CADNA's band is a statistical significant-digit estimate,
EISA's band is the measured residual magnitude plus the declared input
uncertainty; CADNA perturbs and observes disagreement, EISA measures and
compares against the measurement.

The v1 policy decision is deliberately minimal and stated as such: frail is
**count-only** — `frail > 0` never forces `poisoned = 1`; the count is
carried in every receipt (`frail=<n>`) and is therefore part of the
byte-for-byte receipt identity the conformance harness enforces — the three
executors must agree on `frail=<n>`, so the "frail == 1" observation on the
fixed-point loop is reproducible across executors, not just within one. This
makes the instability visible to
the auditor without prejudging the escalation policy (warn versus poison),
which is deferred until a corpus kernel shows which default is right. The
witnessed case is the fixed-point loop of the V1e showcase: the loop exit
`while delta != 0.0` fires exactly once inside the err band (`frail == 1`),
and that single firing is derivable from the iteration arithmetic — the
penultimate delta sits 10–20× above the accumulated err band — not fitted
to the observed output.

## 5. The Rump case study — three regimes, one receipt

Rump's polynomial, f(x, y) = 333.75y⁶ + x²(11x²y² − y⁶ − 121y⁴ − 2) + 5.5y⁸
+ x/(2y) at x = 77617, y = 33096, has exact value −54767/66192 =
−0.827396059946821… = x/(2y) − 2. The polynomial part (everything except
x/(2y)) is exactly −2, hidden under intermediates of magnitude ~2¹²²·⁶, so
recovering it needs ≈122 significand bits of cancellation.

The textbook telling requires care, and we follow Loh & Walster (2002) in
getting it right. Rump constructed the example for IBM S/370 arithmetic,
where single, double and extended precision all returned ≈ +1.172603… —
the seductive convergence to a wrong-sign answer. That convergence is *not*
reproducible on IEEE-754 machines with the expression as written: Loh &
Walster showed the oft-cited behaviour fails to appear on many modern
computers, and rewrote the expression so that Rump's phenomenon is
reproducible under IEEE 754. On IEEE doubles, in the evaluation order used
throughout this repository, plain f64 lands on ≈ −1.18 × 10²¹ — wrong by
some 21 orders of magnitude, with the sign unstable across precisions and
evaluation orders, which is exactly Rump's point (see also Rump 2010, §1,
for his own account of the S/370 history).

The three regimes, as witnessed in this repository:

1. **f64 — debris.** `test_qd128_rump.sio` R4 pins the plain-double result
   at more than 10¹⁵ absolute error in this evaluation order (the
   ≈ −1.18e21 figure and this repository's specific evaluation order are
   *measured from the repository corpus* at R4, not transcribed from Rump
   1988 or Loh & Walster 2002, which are cited only for the phenomenon and
   its IEEE-754 reproducibility history). In the hand-lowered v2 image
   (different but fixed order),
   the final register's val lane is exactly −2⁷⁰ (`s1e1093m0` in the
   receipt decomposition). Both land ~21 orders of magnitude from the true
   −0.827…, their exact debris fixed by the evaluation order — no stability
   anywhere in this regime.
2. **dd64 — wrong sign, derivably.** dd64 carries ~106 bits against a
   ~122-bit cancellation requirement. Every polynomial term *except* the
   exact −2 residue is exactly representable in dd64 and cancels to zero;
   the −2 lies ~2¹²¹ below the largest intermediates and is annihilated
   entirely. The dd64 result is therefore truth + 2 = +1.17260… — the
   famous wrong-sign value, here as a *derived* consequence of the
   representation boundary, not an anecdote (`test_qd128_rump.sio` R3,
   asserting > 100% relative error and the sign flip). Decision #15 makes
   this a mandatory corpus lane in v2 (`v2-rump-dd`): the receipt showing
   dd64 visibly failing at the boundary is standalone scientific evidence,
   not decoration superseded by the qd success.
3. **qd128 — exact.** Quad-double carries ~212 bits; every intermediate of
   the kernel fits exactly (the largest needs ~127 bits) until the final
   division. The v2 Metron VM lane (`test_eisa_evm_v2.sio` W-H) runs the
   hand-lowered 30-instruction image and asserts, on derived bits: the val
   lane is the −2⁷⁰ debris (H1); the err lane is exactly
   round_qd(truth − val) (H2); and val + err reconstructs the truth
   components bit-identically (H3) — with one honest exception, next.

**The anchor-limit honesty (W-H).** The final register's err is anchored at
its val debris, 2⁷⁰, so a renormalised quad-double correction can reach down
only to ~2⁷⁰ × 2⁻²¹¹ ≈ 2⁻¹⁴¹. Truth's fourth component (~2⁻¹⁶³) lies below
that floor: full four-component bit-identity from *one* register is
arithmetically impossible (the required span 70 + 163 + 53 exceeds 212
bits). The single-register reconstruction therefore equals truth on
components x0..x2 with x3 = 0 — and the witness asserts exactly that,
rather than a tolerance fitted to the miss. The exact value *is*
receipt-recoverable bit-identically, from the two gated source registers:
s2, whose true value is −2 exactly, and t4, whose true value is
round_qd(77617/66192); their val + err reconstructions sum in quad-double
to qd_div(−54767, 66192) with zero tolerance (H4, the flagship assertion).
The receipt v3 lines for those gates — `val=<s,e,m>` debris plus
`roundoff0..3=<s,e,m>` corrections, under `prog=<hash>` — are the evidence
artefacts this document's thesis rests on: the wrong number the hardware
computed and the measured correction that recovers the exact rational,
carried in the same executable's audit trail. They are reproduced verbatim,
with the program hash, the ELF SHA-256, the reconstruction arithmetic and the
one-command replay, in the §8 reproducibility appendix — including the
byte-identical agreement between the Metron VM and the x86-64 bridge.

## 6. Honest limitations and future work

1. **Finite domain only.** The v2 qd operations define no NaN/Inf
   semantics; constants are validated finite, and non-finite values enter
   execution only as computed poison (I3). A general numeric machine needs
   the full special-value algebra; EISA v2 does not have it.
2. **Known compiler unsoundness in the validation lane.** The witnesses run
   on the `lean_single` engine, whose NaN-comparison unsoundness is
   documented in the repository's compiler audits
   (`docs/audit/LEAN_SINGLE_NAN_SEMANTICS_2026-07-05.md`); receipts canonicalise
   NaN (`s0e2047m1`) rather than relying on comparison semantics. The
   validation evidence is only as strong as the documented lane.
3. **No production compiler path.** The Metron surface language is minimal
   (single-identifier conditions, define-before-use, a 256-byte source
   buffer with a documented truncation tripwire); the flagship kernels are
   hand-lowered. EISA is a research VM plus a conformance corpus — nothing
   in this document should be read as a production toolchain claim.
4. **Performance.** The Metron VM is interpreted Sounio; v2 qd steps cost
   an estimated ~20–50× a v1 step (a projection from qd operation counts,
   not a benchmark). Acceptable at corpus scale (< 10³ steps per lane),
   not at application scale. The x86-64 bridge exists so that conformant
   fast execution remains possible, not because it is fast today.
5. **The Priest-renorm assumption.** Necessary conditions for Priest
   renormalisation to always succeed are not known; sufficiency holds for
   ≤ 51-bit overlap, which the Hida–Li–Bailey operators maintain. EISA
   inherits this assumption explicitly from the qd literature and makes
   its failure observable (the conformance harness checks the non-overlap
   bound on every receipt's reconstructed roundoff components) rather than
   proving it away.
6. **EFT preconditions.** The err lane's exactness requires strict IEEE-754
   round-to-nearest, no FMA contraction, no reassociation — the bridge is
   pinned to SSE2 scalar non-FMA templates for this reason. On hardware or
   toolchains that violate these, the measured correction is wrong, and
   CADNA-style stochastic detection (which needs none of this) is the more
   robust instrument.
7. **Corpus conformance is corpus conformance.** Three-executor bit-exact
   agreement is checked on the witness corpus, not proved for all programs;
   the claim is falsifiable (any counterexample indicts an executor) but
   deliberately not overstated. The closure contract
   `true(z) = round_qd(true(x) op true(y))` is defined *normatively* in
   `eisa-v2-arch-2026-07-05.md §2` and enforced by the differential harness;
   a machine-checked Lean obligation (call it `closure_sound`) is named
   **deferred future work**, not asserted here. This document states no
   theorem it has not proved and adds no `sorry` placeholder in its place —
   the adversarial reviewer's call for the contract as a Lean theorem is
   acknowledged and scheduled, not papered over.
8. **Priority hedge — criteria C1–C3.** The §1 "first" claim is bounded to
   the *conjunction* of **C1** (mandatory per-operation EFT-measured roundoff
   lane), **C2** (separate GUM uncertainty lane) and **C3** (receipt citing
   program-hash provenance), over the traditions and references surveyed in
   §7. It is stated to our knowledge and is falsifiable: any prior system
   meeting C1–C3 together — or a reference we did not survey — retires it.
   This remains pre-publication research prose, but the W5 adversarial-review
   revisions described in the handoff have been adopted.

Future work, in the order the W-plan fixes: the frail escalation policy once
corpus kernels discriminate the defaults; per-site gate policies in receipts;
the `closure_sound` Lean obligation (§6.7); and a static companion
(FPTaylor-style a priori bounds over Metron kernels) so that the a priori and
per-execution axes can be carried by the same artefact. (The W4 qd128 bridge
templates, listed here as future work in earlier drafts, are now **landed and
gated byte-identical** — the receipt evidence in §8 is produced by that
bridge; W4 is retained in the W-plan only as completed context.)

## 7. References

Verification status: **[V]** = verified against abstract, publisher page or
full text during the web research for this draft (2026-07-06); **[V-2nd]**
= verified via a secondary source (citing paper or bibliography). No
unverified claims are load-bearing in the text above.

- **[V]** Dekker, T. J. (1971). A floating-point technique for extending the
  available precision. *Numerische Mathematik* 18(3), 224–242.
  doi:10.1007/BF01397083.
- **[V]** Denis, C., de Oliveira Castro, P., & Petit, E. (2016). Verificarlo:
  checking floating point accuracy through Monte Carlo Arithmetic.
  *23rd IEEE Symposium on Computer Arithmetic (ARITH)*, Santa Clara, USA.
  doi:10.1109/ARITH.2016.31.
- **[V]** de Figueiredo, L. H., & Stolfi, J. (2004). Affine arithmetic:
  concepts and applications. *Numerical Algorithms* 37(1–4), 147–158.
  doi:10.1023/B:NUMA.0000049462.70970.b6.
- **[V]** Févotte, F., & Lathuilière, B. (2016). VERROU: a CESTAC evaluation
  without recompilation. *International Symposium on Scientific Computing,
  Computer Arithmetics and Verified Numerics (SCAN)*, Uppsala, Sweden.
  (See also: Févotte & Lathuilière (2017), Studying the numerical quality of
  an industrial computing code: a case study on code_aster, *10th Int.
  Workshop on Numerical Software Verification (NSV)*, LNCS, 61–80,
  doi:10.1007/978-3-319-63501-9_5; and Févotte & Lathuilière (2019),
  Debugging and optimization of HPC programs with the Verrou tool,
  *Correctness@SC*, doi:10.1109/Correctness49594.2019.00006.)
- **[V]** Fousse, L., Hanrot, G., Lefèvre, V., Pélissier, P., & Zimmermann, P.
  (2007). MPFR: a multiple-precision binary floating-point library with
  correct rounding. *ACM Transactions on Mathematical Software* 33(2),
  article 13. doi:10.1145/1236463.1236468.
- **[V]** Grasland, H., Févotte, F., Lathuilière, B., & Chamont, D. (2019).
  Floating-point profiling of ACTS using Verrou. *EPJ Web of Conferences*
  214, 05025. doi:10.1051/epjconf/201921405025.
- **[V]** Hida, Y., Li, X. S., & Bailey, D. H. (2001). Algorithms for
  quad-double precision floating point arithmetic. *15th IEEE Symposium on
  Computer Arithmetic (ARITH-15)*, 155–162. doi:10.1109/ARITH.2001.930115.
- **[V]** Jézéquel, F., & Chesneaux, J.-M. (2008). CADNA: a library for
  estimating round-off error propagation. *Computer Physics Communications*
  178(12), 933–955.
- **[V]** Johansson, F. (2017). Arb: efficient arbitrary-precision
  midpoint-radius interval arithmetic. *IEEE Transactions on Computers*.
  doi:10.1109/TC.2017.2690633. (Earlier: Johansson (2014), Arb: a C library
  for ball arithmetic, *ACM Communications in Computer Algebra* 47(3/4),
  166–169, doi:10.1145/2576802.2576828.)
- **[V]** Loh, E., & Walster, G. W. (2002). Rump's example revisited.
  *Reliable Computing* 8, 245–248. doi:10.1023/A:1015569431383.
- **[V]** Muller, J.-M., Brunie, N., de Dinechin, F., Jeannerod, C.-P.,
  Joldes, M., Lefèvre, V., Melquiond, G., Revol, N., & Torres, S. (2018).
  *Handbook of Floating-Point Arithmetic*, 2nd edition. Birkhäuser Boston,
  632 pp. ISBN 978-3-319-76525-9.
- **[V]** Panchekha, P., Sanchez-Stern, A., Wilcox, J. R., & Tatlock, Z.
  (2015). Automatically improving accuracy for floating point expressions.
  *ACM SIGPLAN Conference on Programming Language Design and Implementation
  (PLDI)*. doi:10.1145/2737924.2737959. (Distinguished Paper.)
- **[V-2nd]** Parker, D. S., Pierce, B. A., & Eggert, P. (2000). Monte Carlo
  arithmetic: how to gamble with floating point and win. *Computing in
  Science & Engineering* 2(4), 58–68. doi:10.1109/5992.852391. (Verified via
  the reference lists of Vignes (2004) and Denis et al. (2016).)
- **[V-2nd]** Priest, D. M. (1991). Algorithms for arbitrary precision
  floating point arithmetic. *10th IEEE Symposium on Computer Arithmetic
  (ARITH-10)*, 132–143. (Verified via the bibliographies of Hida–Li–Bailey
  (2001) and the *Acta Numerica* 2023 survey "Floating-point arithmetic".
  See also Priest (1992), PhD thesis, UC Berkeley, cited by Hida–Li–Bailey.)
- **[V]** Rump, S. M. (1988). Algorithms for verified inclusions: theory and
  practice. In R. E. Moore (ed.), *Reliability in Computing: The Role of
  Interval Methods in Scientific Computing*, Academic Press, Boston,
  109–126. doi:10.1016/B978-0-12-505630-4.50012-2.
- **[V]** Rump, S. M. (2010). Verification methods: rigorous results using
  floating-point arithmetic. *Acta Numerica* 19, 287–449.
  doi:10.1017/S096249291000005X. (Source of the IBM S/370
  single/double/extended ≈ +1.172603 account and the author's own
  1983-construction footnote.)
- **[V]** Shewchuk, J. R. (1997). Adaptive precision floating-point
  arithmetic and fast robust geometric predicates. *Discrete &
  Computational Geometry* 18(3), 305–363. doi:10.1007/PL00009321.
- **[V]** Solovyev, A., Jacobsen, C., Rakamarić, Z., & Gopalakrishnan, G.
  (2015). Rigorous estimation of floating-point round-off errors with
  Symbolic Taylor Expansions. *20th International Symposium on Formal
  Methods (FM)*, LNCS 9109, 532–550. doi:10.1007/978-3-319-19249-9_33.
- **[V]** Solovyev, A., Baranowski, M. S., Briggs, I., Jacobsen, C.,
  Rakamarić, Z., & Gopalakrishnan, G. (2018). Rigorous estimation of
  floating-point round-off errors with Symbolic Taylor Expansions. *ACM
  Transactions on Programming Languages and Systems* 41(1), 2:1–2:39.
  doi:10.1145/3230733.
- **[V]** Vignes, J. (2004). Discrete Stochastic Arithmetic for validating
  results of numerical software. *Numerical Algorithms* 37(1–4), 377–390.
  doi:10.1023/B:NUMA.0000049483.75679.ce.
- **[V-2nd]** Vignes, J. (1993). A stochastic arithmetic for reliable
  scientific computation. *Mathematics and Computers in Simulation* 35(3),
  233–261. doi:10.1016/0378-4754(93)90003-D. (Verified via the reference
  list of Vignes (2004); cited here for the CESTAC lineage only.)

Repository-internal evidence cited above:
`docs/research/eisa-stack-architecture-2026-07-05.md`,
`docs/research/eisa-v1-plan-2026-07-05.md`,
`docs/research/eisa-v2-arch-2026-07-05.md`,
`stdlib/eisa/core_v2.sio`, `stdlib/math/qd128.sio`,
`tests/stdlib/math/test_qd128_rump.sio` (R1–R4),
`tests/stdlib/math/test_dd64_cancellation.sio`,
`tests/stdlib/eisa/test_eisa_evm_v2.sio` (W-A…W-H, esp. the W-H honesty
note), `tests/stdlib/eisa/test_eisa_v1e_showcase.sio` (S1 framing),
`.claude/decisions.md` #14–#16.

## 8. Appendix — reproducibility (v2-rump-qd)

This appendix makes the §1/§5 thesis checkable from the artefact: the actual
receipt v3 text, its hashes, the reconstruction arithmetic, and a one-command
replay. Every figure below is emitted by the shipped code, not transcribed.

**The image.** Rump 1988 at (77617, 33096), hand-lowered to a 30-instruction
version-2 `.eisax` image (registers e0..e14, fuel 64), built by `rump_build`
in `tools/eisa/eisa_evm_run.sio` and `tools/eisa/eisa_bridge_emit.sio`, and by
`wv2_build_wh_rump` (witness W-H) in `tests/stdlib/eisa/test_eisa_evm_v2.sio`.
Its version-1 twin (`v1-rump-dd`) is the mandatory dd64 *failure* lane.

**Replay (from the repository root):**

```
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"; export SOUNIO_SOUC_ENGINE=lean_single
./bin/souc run tools/eisa/eisa_evm_run.sio      # Metron VM receipts (the oracle)
./bin/souc run tools/eisa/eisa_bridge_emit.sio  # emit the x86-64 AOT ELFs
./artifacts/eisa/v2-rump-qd.eisax.elf           # the AOT bridge's own receipts
bash scripts/ci/eisa_bridge_conformance_gate.sh # EVM-vs-AOT byte-diff, all lanes
```

**The receipt v3 lines** (identical from the Metron VM and from the x86-64
bridge ELF; `prog=845863096942225452`):

```
eisa-receipt: v=3 prog=845863096942225452 gate=1 reg=e13 val=s1e1093m0 roundoff0=s0e1093m0 roundoff1=s1e1024m0 roundoff2=s0e0m0 roundoff3=s0e0m0 u=s0e0m0 poisoned=0 frail=0
eisa-receipt: v=3 prog=845863096942225452 gate=2 reg=e12 val=s0e1023m777339040106175 roundoff0=s1e969m612890159586558 roundoff1=s0e915m713042725629121 roundoff2=s0e861m1170260961910390 roundoff3=s0e0m0 u=s0e0m0 poisoned=0 frail=0
eisa-receipt: v=3 prog=845863096942225452 gate=3 reg=e14 val=s1e1093m0 roundoff0=s0e1093m0 roundoff1=s1e1022m2948921547158147 roundoff2=s0e968m3277819308197381 roundoff3=s1e914m3077514176112253 u=s0e0m0 poisoned=0 frail=0
```

- Fields are `f64_parts` sign/exponent/mantissa decompositions (`s<sign>e<biased-exp>m<mantissa>`); e.g. `s1e1093m0` = −2⁷⁰, `s0e1093m0` = +2⁷⁰.
- **Byte-identity.** The Metron VM output and the AOT ELF output for these
  three lines are byte-for-byte identical; the conformance gate asserts this
  over all 33 lanes (`PASS eisa_bridge_conformance`). AOT ELF:
  `artifacts/eisa/v2-rump-qd.eisax.elf`, 71 819 bytes,
  SHA-256 `b04f7795f3a9558bee428ab9fdfaaf649b8056515cb0500ab7cd3323cb581a5f`.

**The reconstruction (derived bits, zero fitted tolerance).** Decode the
final gate (`gate=3`, register e14) and sum val + roundoff0..3 as exact
rationals:

| field | value |
|---|---|
| `val` (s1e1093m0) | −2⁷⁰ |
| `roundoff0` (s0e1093m0) | +2⁷⁰ |
| `roundoff1` (s1e1022m…) | −0.827396059946821… |
| `roundoff2` (s0e968m…) | +… (2⁻⁵⁵-scale) |
| `roundoff3` (s1e914m…) | −… (2⁻¹⁰⁹-scale) |

The ±2⁷⁰ debris cancels exactly (`val + roundoff0 = 0`), and the surviving
quad-double sum agrees with the exact rational digit-for-digit through the
~49th significant decimal, then diverges (the space marks the first differing
digit):

```
val + roundoff0..3 = −0.827396059946821368141165095479816291999033115784 2…
      −54767/66192 = −0.827396059946821368141165095479816291999033115784 3…
```

i.e. |Δ| ≈ 1.27 × 10⁻⁴⁹ ≈ 2⁻¹⁶² — about 163 significand bits of agreement.

**Self-contained check (no binary needed).** The reconstruction above is
verifiable from the printed receipt digits alone: each `s<σ>e<ε>m<μ>` field
decodes to the IEEE-754 double `(−1)^σ · (1 + μ/2⁵²) · 2^(ε−1023)` for ε ≠ 0
(and `s0e0m0 = 0`); sum the five decoded doubles of the `gate=3` line (`val`
+ `roundoff0..3`) as exact rationals and compare to −54767/66192. So the
*central* claim — that val + err recovers the exact rational on derived bits
— is checkable from this document's own numbers; only the byte-identity and
SHA-256 lines require the repository artefact (`artifacts/eisa/`, regenerated
by the replay commands above and pinned by the conformance gate). The `val` lane *alone* is −2⁷⁰ — wrong sign, ~21 orders of
magnitude off — so the entire recovered result lives in the measured
`roundoff` correction, on derived bits, exactly as the thesis claims. (The
single-register anchor-limit caveat of §5 — x3 unreachable from one register
anchored at 2⁷⁰ — is why the flagship assertion reconstructs from the two
gated source registers s2 and t4; the three-line receipt above is the
end-to-end demonstration.)
