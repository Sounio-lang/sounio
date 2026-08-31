<!-- docs:meta
topic_id: repo.docs.papers.witness-based-compilation-2026-07-28
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.witness-based-compilation-2026-07-28
-->

# Witness-Based Compilation: When the Verdict Is Right but the Evidence Is Wrong

**Status:** `DRAFT` — every empirical claim below cites the rung of the
self-falsifying compilation line that measured it, and the companion gate
(`scripts/ci/witness_based_compilation_paper_gate.sh`) fails if a cited verdict
token drifts from the spec that declares it.
**Date:** 2026-07-28
**Orthography:** EN-UK
**Evidence:** rungs R0, R1, R2, R14–R19 of the self-falsifying compilation
line, `docs/research/self_falsifying_compilation_line*_2026-07-2[6-8].md`
**Reference implementation:** the Sounio compiler (self-hosted; witness binding
in `self-hosted/compiler/claim_executor.sio`)

---

## Abstract

Verified software can be wrong in a way no verifier can see. A check that
establishes a *proposition* — "there are exactly 24 distinct spectra" — binds a
build to the truth of that proposition, not to the *evidence* that established
it. We exhibit a transformation, measured on a real research contract in the
Sounio repository, that leaves the proposition true and replaces the evidence
entirely: a single Cayley–Dickson sign flip changes 126 of 128 fibre graphs and
every one of their spectra while preserving their count — 24 before, 24 after,
and the set of 24 wholly exchanged. A verifier bound to the proposition reports
success on both sides of the flip. The verdict is right; the evidence is wrong.

We develop the theory of this phenomenon. For any proposition $p$ the
transformations that preserve it form a group, the *invariance group* of $p$,
and a verifier bound to $p$ is blind to exactly this group. Where $p$ factors
through an evidence function $w$ — the count is a function of the set counted —
the stabiliser of $w$ is a subgroup, and the part of the invariance group that
moves the witness is precisely the class of errors proposition-based
verification cannot catch. *Witness binding* closes the gap: the claim declares
a fingerprint of its evidence, the check emits the fingerprint of the evidence
it actually used, and the compiler refuses to emit an artifact when the two
disagree — even when the proposition holds and its verdict token agrees.

We implement witness binding in the Sounio compiler, a self-hosted compiler
whose claims run after type-checking and before code generation, and we apply
it to the motivating case: `zd_fiber_spectra_count_law_holds` is, to our
knowledge, the first production claim whose build is bound to a cryptographic
hash of its evidence. A perturbed twin of its check — one that satisfies the
proposition and emits the identical verdict token — is refused at compile time
with `CLAIM_WITNESS_MISMATCH`. We prove the soundness and completeness of
witness binding relative to its fingerprint function, state the inherited limit
plainly (a witness binds *which* evidence was used, not whether that evidence
is well-founded), and report the cost: the witness adds no measurable overhead
beyond the computation it fingerprints.

---

## 1. Introduction

### 1.1 The proposition is not the evidence

Programming-language verification has become very good at binding artifacts to
propositions. Proof-carrying code attaches a proof that the program satisfies a
safety policy [1]; certified compilers carry a proof that the translation
preserves semantics [2]; certifying algorithms emit, beside their answer, a
witness that a cheap checker can validate [3]. In each case the obligation
discharged is a *proposition*: a statement, decidable or provable, about a
program or a computation.

This paper is about a distinction those frameworks do not draw, because within
them it does not arise: the distinction between a proposition being true and
the *evidence on which it was established* being the evidence intended. The
distinction arises as soon as the check is *empirical* — when the proposition
quantifies over computed objects rather than over the program text. A contract
in the Sounio repository states that the adjacency spectra of the ZD fibres of
the Cayley–Dickson tower number $3 \cdot 2^{n-5}$. At $n = 8$ that is 24
distinct spectra. Rung R15 of the repository's self-falsifying compilation line
[15] measured the following: flip the sign of a single product,
$\sigma(64, 192)$, and

- 126 of 128 fibre graphs change;
- **every** spectrum of every fibre changes;
- the number of distinct spectra is 24 before and 24 after — and the set of 24
  is entirely replaced.

The contract's check passes on both sides of the flip. The proposition is a
cardinality; a cardinality cannot see a transformation that swaps the things it
counts. **The verdict is right about the proposition and wrong about the
witness.**

This is not a bug in the check, nor a false claim: a perturbed sign table is
not a Cayley–Dickson algebra, so nothing here bears on the underlying
mathematics (a scope limit fixed before any measurement [15, §3]). It is a
*resolution* limit, and it is structural: a verifier bound to a proposition is
exactly as fine as the proposition, and propositions that aggregate — counts,
existence statements, cardinalities of classifications — are coarse.

### 1.2 Contributions

This paper makes the distinction precise, builds the mechanism it points to,
and applies it.

| # | Contribution | Rung | Verdict token |
|---|---|---|---|
| 1 | A formal theory of proposition-blindness: the invariance group of a proposition, the stabiliser of a witness, and the theorem that a proposition-bound verifier is blind to exactly the former (§2) | R15 | `TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE` |
| 2 | Identification of the group in the motivating case: not count-preserving but *partition-preserving* maps — the flip preserves the entire classification and relabels every block (§2.3) | R16 | `INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING` |
| 3 | Witness binding: a compiler that refuses to emit an artifact whose declared evidence fingerprint disagrees with the fingerprint its check emits, even when the proposition holds and its token agrees (§3) | R17 | `WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION` |
| 4 | The first production claim bound to a witness: `zd_fiber_spectra_count_law_holds`, with its perturbed twin refused at compile time (§3.4, §4) | R17 | `WITNESS_BINDING_IMPLEMENTED__REFUSES_ON_PRESERVED_PROPOSITION` |
| 5 | Soundness and completeness of witness binding relative to its fingerprint function, and the limits that survive it (§2.4, §4.3) | R0 | `SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE` |

Each row cites the verdict token of the repository rung that measured it; the
paper's gate fails if any cited token drifts from its spec. Concretely, the
companion gate parses this paper's source and checks three things against the
repository: that every cited verdict token still matches the token its rung's
spec declares; that the witness fingerprints quoted in §3.4 still match the
claim in the bound-claims manifest; and that the measured/derived status
distinctions of §2.3 have not been edited away. The paper, in other words, is
held to the same discipline it studies. Contribution 4 is
the rung the line calls R18; it shipped as §1.3 of the R17 spec [13] rather
than as a separate document, and we cite it there.

The theory (§2) is the primary contribution and is implementation-independent.
Sounio (§3) is the reference implementation: the first compiler we know of
whose code generation is conditioned on the identity of a claim's evidence,
not merely on the truth of a claim's proposition.

### 1.3 Prior work, positioned briefly

Proof-carrying code [1], certifying compilers [4], certified compilers [2], and
translation validation [5] all attach machine-checkable evidence to software —
but the evidence is *about the program*, and the verifier checks a proof, not
the world. Certifying algorithms [3] are the closest algorithmic neighbour:
they return a witness beside the answer so that correctness reduces to
checking. The difference is one of direction: a certifying algorithm's witness
*establishes* the proposition for a checker; our witness *identifies* which
evidence established it, for a builder. Reproducible builds [6], hash-pinned fetching — Nix fixed-output derivations,
Bazel's `download(sha256=…)`, Go's `go.sum` — and software-provenance
frameworks such as SLSA bind artifacts to their inputs by
hash — the same primitive — but towards the opposite design goal: they make the
build independent of the world, where witness-based compilation makes the build
depend on the world on purpose, and refuses when the world's relevant face
changed. Metamorphic testing, finally, studies exactly the group this paper
names — which transformations preserve a property — and exploits it to
generate tests where no oracle exists; we use the same group to delimit where
an oracle that exists is insufficient. Section 6 develops these comparisons.

---

## 2. The theory

### 2.1 Propositions, witnesses, and verifiers

Fix a set $S$ of *states*: everything about a computation relevant to a check —
inputs, intermediate objects, outputs, and the world they describe. A check is
a procedure that, run in a state, reports on that state.

**Definition 2.1 (proposition).** A *proposition* is a predicate $p : S \to
\{0,1\}$.

**Definition 2.2 (evidence, witness).** An *evidence function* is a map $w : S
\to W$ into a set of *witnesses*. We say $p$ *factors through* $w$ when there
is a $\pi : W \to \{0,1\}$ with $p = \pi \circ w$: the truth of the proposition
is determined by the evidence.

Factoring is the normal case for empirical claims: "there are exactly $N$
distinct spectra" is determined by *which* spectra there are; "the minimum is
attained" by the argmin; "the classification has these blocks" by the
labelling. A proposition that does not factor through any natural evidence
function is one whose check carries no residue — and such propositions are not
the ones this paper is about.

**Definition 2.3 (verifier).** A *verifier* for a claim is a predicate on what
the check observably emits. Three grades concern us:

- *exit-code gating* observes only whether the check ran to success:
  $V_{\mathrm{rc}}(s) = \text{``check exits 0 in } s\text{''}$;
- *token binding* observes a reported token, canonically the value of $p$:
  $V_{p}(s) = [\,p(s) = 1\,]$ (the token the check emits reports this boolean
  outcome; the claim declares the expected token in advance);
- *witness binding* observes a fingerprint $f : W \to F$ of the evidence:
  $V_{w,h}(s) = [\,f(w(s)) = h\,]$ for a declared reference fingerprint $h$.

Exit-code gating is `build.rs` and its cousins; token binding is the mechanism
rung R2 of the line built and bounded [16]; witness binding is this paper's
mechanism. Each refines the previous one.

### 2.2 The invariance group

**Definition 2.4 (invariance group).** For a proposition $p$, let
$$\mathrm{Inv}(p) = \{\, \sigma : S \to S \text{ bijective} \;\mid\; p \circ
\sigma = p \,\},$$
the transformations that leave the truth value of $p$ unchanged, pointwise.

**Proposition 2.5.** $\mathrm{Inv}(p)$ is a group under composition.

*Proof.* Composition of bijections is bijective; if $p \circ \sigma = p$ and $p
\circ \tau = p$ then $p \circ (\sigma \circ \tau) = p \circ \tau = p$; and from
$p \circ \sigma = p$, composing with $\sigma^{-1}$ on the right gives $p = p
\circ \sigma^{-1}$. The identity preserves $p$. ∎

**Definition 2.6 (stabiliser).** For an evidence function $w$, let
$$\mathrm{Stab}(w) = \{\, \sigma : S \to S \text{ bijective} \;\mid\; w \circ
\sigma = w \,\},$$
the transformations that leave the evidence itself unchanged.

**Proposition 2.7.** $\mathrm{Stab}(w)$ is a group, and if $p$ factors through
$w$ then $\mathrm{Stab}(w) \le \mathrm{Inv}(p)$.

*Proof.* The group property is as in Proposition 2.5. For the inclusion: if
$w \circ \sigma = w$ then $p \circ \sigma = \pi \circ w \circ \sigma = \pi
\circ w = p$. ∎

The subgroup relation is the whole story in one line: **everything that
preserves the evidence preserves the proposition; the converse fails exactly
where proposition-based verification has blind spots.**

**Theorem 2.8 (blindness of proposition binding).** Let $V$ be a verifier that
factors through $p$ — i.e. $V(s) = V(s')$ whenever $p(s) = p(s')$. Then for
every state $s$ and every $\sigma \in \mathrm{Inv}(p)$, $V(\sigma(s)) = V(s)$.
In particular, if $\sigma \in \mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$, then
$V$ cannot distinguish a state whose evidence is $w(s)$ from one whose evidence
is $w(\sigma(s)) \neq w(s)$.

*Proof.* $p \circ \sigma = p$ by membership in $\mathrm{Inv}(p)$, so $p$ takes
the same value at $s$ and $\sigma(s)$, and $V$ factors through $p$. ∎

Theorem 2.8 is deliberately elementary — it is a scoping statement, not a deep
theorem, and its force is that the blind spot is *exact*, characterised by a
group, rather than anecdotal. A verifier bound to the proposition is blind to
exactly the proposition's symmetries. This is the abstract content of rung
R15's verdict `TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE` [15].

### 2.3 The invariance group of a cardinality

Aggregate propositions have large invariance groups, and for classification
claims the group has a precise description.

**Proposition 2.9 (the invariance group of a count).** Let the evidence be a
subset of a finite ground set of computable objects, $w(s) \subseteq X$ with
$|X| = M$, and let the proposition be a cardinality statement, $p(s) =
[\,|w(s)| = N\,]$ with $N \le M$. Then every transformation of $S$ induced by
permuting the underlying objects of $X$ lies in $\mathrm{Inv}(p)$ —
permutations preserve cardinality — while such a permutation lies in
$\mathrm{Stab}(w)$ if and only if it maps the produced subset to itself *as a
set*: the witness is a single set value, and $w \circ \sigma = w$ asks that
value to be preserved, not its elements. The stabiliser is therefore the
**setwise** stabiliser of $w(s)$ in $\mathrm{Sym}(X)$ — permute within the
produced subset and within its complement independently, order
$N!\,(M - N)!$ — not the pointwise stabiliser, which fixes each produced object
individually (order $(M - N)!$) and is a proper subgroup whenever $N \ge 2$.
The number of evidence values indistinguishable from the true one under $p$ is
$\binom{M}{N} - 1$, the orbit of $w(s)$ under $\mathrm{Sym}(X)$ minus the true
value: *any* exchange of contents at fixed cardinality preserves the
proposition and moves the witness, while a relabelling *internal* to the
produced set moves nothing — not even the fingerprint, which in the
implementation is a hash over the sorted enumeration (§3.4).

The pointwise/setwise distinction is not pedantry: the earlier draft of this
proposition required every produced object to be fixed, conflating the witness
*value* (a set) with the group elements that act on its members, and thereby
shrinking the stabiliser by a factor of $N!$ — with the orbit of supposedly
distinct evidence values inflated by the same factor. For set-valued evidence
the setwise reading is the correct one; the pointwise reading would be right
only for *ordered* evidence — a tuple whose positions carry meaning — which is
not the case at hand.

The content of Proposition 2.9 is not the arithmetic but the shape: for a
count, the invariance group contains the *largest* symmetry compatible with the
predicate (all of $\mathrm{Sym}(X)$, order $M!$), and the stabiliser of the
evidence contains exactly what permutes within the produced set and within its
complement (order $N!\,(M - N)!$). The blind spot spans the gap between them —
a factor of $\binom{M}{N}$, matching the orbit count above. Rung R15
described the group in the motivating case as "maps preserving $|X|$"; rung R16
measured that the actual flip does something narrower and more surprising — it
preserves the whole set partition of fibres into spectrum-classes, identical
blocks with identical sizes, while replacing the spectrum labelling every block
[14]. So in the motivating case the group is better described as
**partition-preserving**: count-preservation is a consequence of it, not the
mechanism (verdict
`INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING`).

Two facts about the concrete case are arithmetic and hold for all $n$; the rest
is measured, and we keep the statuses separate because the line that produced
them does.

- *(Derived.)* With $H = 2^{n-1}$ and $h = H/2$, the flipped pair $(h, H+h)$
  satisfies $h \oplus (H+h) = H$ for every $n$ (verified mechanically for
  $n = 5 \ldots 12$ [14, §1.1]): its home fibre is the single fibre the
  contract does not examine, so the flip cannot alter any vertex's internal
  product.
- *(Derived.)* The flip reaches a fibre only through the vertex-pairs $P$
  (with $\mathrm{lo} = h$) and $Q$ (with $\mathrm{hi} = H+h$). $P$ and $Q$
  coincide only in the unexamined home fibre of the previous bullet (low label
  $0$), and $Q$ fails to exist only when the fibre's low label is exactly $h$
  — so exactly one *examined* fibre is untouched, the fibre with low label
  $H/2$, which is not the home fibre [18, §1.1, lemmas L1–L2].
- *(Measured, not proved.)* The flip changes exactly two edges per fibre in all
  but that one fibre, and the partition of fibres into spectrum-classes is
  identical before and after while every spectrum differs (measured at
  $n = 5, 6, 7$ [14, §§1.2–1.3]). Why the assignment $\mathrm{Llo} \mapsto \{h,
  h \oplus \mathrm{Llo}\}$ is equivariant with the spectrum-block structure is
  an **open lemma**, stated precisely in [18, §2]; two natural explanations
  were tested there and refuted.

We flag the boundary because a theory paper earns trust the same way a check
does: by saying exactly which of its statements are proved, and which are
measured with their probes disclosed.

### 2.4 Witness binding

The repair is to bind the evidence, not the truth value.

**Definition 2.10 (witness-bound claim).** A *witness-bound claim* is a triple
$(p, w, h)$ where $p$ factors through $w$ and $h \in F$ is a declared
fingerprint. Its verifier is $V_{w,h}(s) = [\,f(w(s)) = h\,]$.

**Theorem 2.11 (soundness and completeness, relative to the fingerprint).**
Let $f : W \to F$ be injective. Then for every state $s$,
$$V_{w,h}(s) = 1 \iff w(s) = w(s_0),$$
where $s_0$ is any reference state with $f(w(s_0)) = h$. That is, witness
binding accepts **iff** the evidence is exactly the reference evidence: sound
(no other evidence passes) and complete (the correct evidence always passes).

*Proof.* $V_{w,h}(s) = 1 \iff f(w(s)) = h = f(w(s_0)) \iff w(s) = w(s_0)$, the
last step by injectivity. ∎

Two honest caveats attach to Theorem 2.11, and both appear again in §§4.3–4.4.

1. **The fingerprint is idealised.** Concrete $f$ is a cryptographic hash
   (SHA-256 in the implementation), injective only up to collision resistance.
   The theorem then holds *up to the adversary's inability to find collisions*;
   for a compiler guarding a research corpus against drift rather than against
   an active adversary, this is the right strength, and we say so rather than
   claim more.
2. **The reference is declared, not derived.** Nothing computes $h$ for the
   claim's author; the claim asserts which evidence it means. Witness binding
   therefore inherits the scope limit of the whole framework (rung R0's
   proposition [19, §3]): a witness binds *which* evidence was used, not
   whether that evidence is well-founded. If claim and check were authored from
   the same misunderstanding, they agree on a witness. What Theorem 2.11 buys
   is exactness about identity, not soundness about the world.

**Theorem 2.12 (strict refinement).** If $p$ factors through $w$ and $f$ is
injective, then witness binding refines token binding: $V_{w,h}(s) = 1$
implies $p(s) = p(s_0)$. The refinement is strict on the reachable states
exactly when some $\sigma \in \mathrm{Inv}(p)$ maps a reachable state $s_0$ to
a reachable state while moving the witness there: $w(\sigma(s_0)) \neq w(s_0)$.
(The condition must be local in this way: membership in $\mathrm{Inv}(p)
\setminus \mathrm{Stab}(w)$ only moves $w$ *somewhere*, and a transformation
whose witness movement is confined to unreachable states does not separate the
two verifiers on the states that can occur.)

*Proof.* If $V_{w,h}(s) = 1$ then $w(s) = w(s_0)$ by Theorem 2.11, so $p(s) =
\pi(w(s)) = \pi(w(s_0)) = p(s_0)$. Strictness: if such a $\sigma$ exists with
$s_0$ and $s = \sigma(s_0)$ both reachable, then $p(s) = p(s_0)$ but $w(s) \neq
w(s_0)$, so token binding (relative to the reference) accepts $s$ and witness
binding rejects it. Conversely, suppose reachable $s, s_0$ have $p(s) = p(s_0)$
but $w(s) \neq w(s_0)$; then the transposition of $S$ that swaps $s$ and $s_0$
and fixes everything else preserves $p$ pointwise (the two swapped points have
equal $p$-values) and moves $w$ at $s_0$, so it is an element of
$\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ mapping reachable to reachable. In
the absence of any such element, then, $p(s) = p(s_0)$ implies $w(s) = w(s_0)$
on reachable states, and the two verifiers agree there. ∎

The motivating case is exactly the strict regime, and it is not hypothetical:
the flip $\sigma(H/2, H + H/2)$ is an exhibited element of $\mathrm{Inv}(p)
\setminus \mathrm{Stab}(w)$ at every level $n = 5, 6, 7, 8$, moving the witness
at the measured — hence reachable — states, with generic sign flips at the
same levels changing the count — the controls that make it a finding rather
than a robustness observation [15, §1.1].

**Corollary 2.13 (coarseness of classification claims).** Any claim of the
form "there are exactly $N$ equivalence classes" is strictly refined by witness
binding whenever more than one classification into $N$ classes is reachable.
The stronger the classification theorem, the coarser its verdict token: the
token states a cardinality, and the content is a labelling.

### 2.5 What the theory does not say

Three limits belong in the theory section, not in the fine print.

- **Shared misinterpretation is untouched.** Rung R0's scope-limit proposition
  [19, §3]: no compile-time procedure whose only evidence about $p$ is the
  behaviour of the claim's own check can detect claim and check being wrong
  *together*. Witness binding is such a procedure, and it inherits the
  limitation unchanged. The repository's own history measured three real
  self-corrections of which zero were reachable by any grade of binding
  [19, §2]; witness binding changes the reach of *identity*, not of *meaning*.
  This is Pollack's regress — believing a machine-checked result requires
  believing what the checker checked, and that belief is not itself
  machine-checked [27] — read at the scale of a build: witness binding narrows
  what must be believed to the identity of the evidence, and stops there.
- **The group is characterised, not computed.** We exhibit elements of
  $\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$; we do not compute either group.
  Whether the partition-preserving maps in the motivating case form a group
  with more exhibitable elements was not investigated [14, §3].
- **Measured premises remain measured.** The locality of the flip is derived
  (§2.3); the preservation of the classification under it is measured at
  $n = 5, 6, 7$ and reduced — but not proven — to one equivariance lemma
  [18]. The theory in §§2.2–2.4 does not depend on that lemma; the case
  study's strength does.

---

## 3. The implementation: witness binding in Sounio

### 3.1 Setting

Sounio is a self-hosted systems language with an epistemic focus: scientific
premises of a program are first-class `claim` blocks in source, and the
compiler (`--verify-claims`) executes each claim's gate after type-checking and
**before code generation**, refusing to emit an artifact whose premises fail.
The claim executor is `self-hosted/compiler/claim_executor.sio` in the
self-hosted compiler (the Madaros line). Before this work the executor knew two
grades of refusal: the gate's exit status (rung R0's baseline) and, since rung
R2, the *verdict token* — a claim may declare `verdict_token = "…"`, the gate
must emit `<PREFIX>_VERDICT <TOKEN>`, and disagreement or absence refuses
codegen (`CLAIM_TOKEN_MISMATCH`, `CLAIM_TOKEN_ABSENT`) even when the gate exits
0 [16].

R2's own accounting is the setup for this paper: token binding catches *drift*
(claim and check diverging) and provably cannot catch *shared
misinterpretation* [16, §2]. R15 then measured a third class that neither exit
codes nor tokens reach: the check is right, the claim is right, and the
*evidence changed underneath both*. Witness binding is the mechanism for that
class.

### 3.2 The mechanism

The implementation is confined to the claim executor; the parser needed no
change, since claim field names are not allowlisted. A claim may now declare

```
witness = "<fingerprint>"
```

and the executor reads the gate's captured output for `<PREFIX>_WITNESS
<fingerprint>` — taking the last occurrence, exactly as the token is read.
Comparison runs *after* the token decision, in a fresh variable at each stage
(`outcome` → `decided` → `settled`), so a claim declaring both must satisfy
what it asserts *and* the grounds it asserts it on. Disagreement refuses
codegen with `CLAIM_WITNESS_MISMATCH`; a declared witness the gate never emits
refuses with `CLAIM_WITNESS_ABSENT`. Both are compile errors: no ELF is
produced.

Two implementation facts are load-bearing, because each is an instance of the
phenomenon the paper studies.

- **One derivation, two readers.** The token reader and the witness reader both
  delegate to a single extraction function (`ce_extract_after`); the rung's
  contract checks there is exactly one scan body in the source. Writing the
  scan twice would be one derivation in two shirts — two routes that agree
  because they are the same code, not because they concur — committed inside
  the very arc that measures that failure [13, §2.1].
- **A behaviour receipt, not a surface check.** Executor source passing a text
  audit is not evidence the compiler behaves: R2 recorded a build whose source
  was correct and whose compiler SIGSEGV'd on every claim [16, §1.1]. The
  witness-binding contract therefore refuses to certify without a receipt of
  observed probe behaviour, hashed to the executor's own SHA-256; edit the
  executor and the receipt goes stale [13, §4]. Source surface is not
  behaviour — which is the paper's thesis applied to the paper's tooling.

### 3.3 Observed behaviour

Measured on a compiler built from the executor source
(`artifacts/self-hosted/madaros-witness-binding`), the four witness probes and
the regressions along the shared code path [13, §1]:

| Probe | Gate behaviour | rc | ELF | Outcome |
|---|---|---:|---|---|
| W1 | token ✓, witness ✓ | 0 | yes | `CLAIM_PASS` |
| **W2** | **exit 0, token ✓, witness ✗** | **1** | **no** | **`CLAIM_WITNESS_MISMATCH`** |
| W3 | token ✓, no witness emitted | 1 | no | `CLAIM_WITNESS_ABSENT` |
| W4 | witness-changing gate, claim declares no witness | 0 | yes | `CLAIM_PASS` |
| R2 regressions | token match / drift / absent | 0/1/1 | yes/no/no | unchanged |
| R0/R1 regressions | exit-code gating, no claims | 0 | yes | unchanged |

W2 is the mechanism: its gate exits 0, so exit-code gating passes it; it emits
exactly the declared token, so token binding passes it; the build is refused
anyway. W4 is the safety property: a claim that declares no witness behaves
exactly as before, even against a gate whose witness moved. The mechanism is
opt-in.

### 3.4 The first production claim with a witness (R18)

The motivating case is now bound. In the repository's bound-claims manifest
(`examples/epistemic/rupture_claims_verified.sio`), the claim
`zd_fiber_spectra_count_law_holds` declares both a token and a witness:

```
claim zd_fiber_spectra_count_law_holds {
    hypothesis = "the ZD-fiber adjacency spectra number 3*2^(n-5) for n=5,6,7,
                  and are the ones this claim fingerprints",
    falsifier  = "the count law fails, or the count holds on a different set
                  of spectra",
    gate       = "scripts/ci/zd_fiber_spectra_witness_gate.sh",
    verdict_token = "SPECTRA_COUNT_IS_3_TIMES_2_POW_N_MINUS_5",
    witness    = "705d0afdf8e830756f5d58eed9e6a11c7681d9e2e3a29ce7054ea67edc385757",
    ...
}
```

The witness is a SHA-256 over the spectra themselves, sorted — the labelling,
not its size. The gate ships with a *perturbed twin* that applies the
count-preserving flip R15 measured and R16 explained. The twin exits 0, reports
3/6/12 distinct spectra — the count law holds of the perturbed object exactly
as of the real one — and emits the **same** verdict token. Only the witness
differs: emitted `e9f935cbab6f09fe…` against declared `705d0afd…`, and the compile is
refused with `CLAIM_WITNESS_MISMATCH` [13, §1.3]. We re-ran both gates while
writing this paper (2026-07-28): the real gate emits witness `705d0afdf8e83075…`
in 3.4 s wall-clock; the perturbed twin emits the identical token and witness
`e9f935cbab6f09fe…`. The proposition is preserved; the evidence is replaced; only
the witness records it.

The bound proposition covers $n = 5, 6, 7$ and says so: the $n = 8$ computation
on which the anomaly was first found costs ~86 s and would hit the executor's
30 s per-gate cap — a scoping decision recorded in the claim itself. Because
$n = 8$ is the level where the anomaly was discovered, the exclusion deserves
an argument rather than an apology. Three points. First, *the exclusion is
inside the proposition*: the claim asserts the count law at $n = 5, 6, 7$ and
nothing beyond it, so no $n = 8$ behaviour can pass silently — there is no
claim at that level to violate, and a future $n = 8$ claim would be a new
binding with its own witness, not a silent extension of this one. Second,
*the phenomenon is level-agnostic*: the flip is an exhibited element of
$\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ at every level $n = 5, 6, 7, 8$
(§2.4), so the bound levels already bind the same group element, acting by the
same mechanism, that the $n = 8$ anomaly exhibited; $n = 8$ would add a larger
instance of the same measured fact, not a new fact. Third, *the cap is a
budget, not a boundary of the method*: the 30 s limit is an executor constant,
and the tempting alternative — precompute the $n = 8$ spectra once and
fingerprint the cache — would quietly change the evidence function from
*computed at this build* to *read from a file*, weakening exactly the property
the witness exists to bind. Binding the levels the build can afford to
recompute, and declaring the boundary in the proposition, is the sound
scoping; raising the budget is deferred engineering (§5), recorded here rather
than disguised.

This is, to our knowledge, the first claim in any compiler whose build is
conditioned on a fingerprint of its evidence. It is also, plainly, **one claim
of roughly 295** in its corpus [13, §3]; the deployment question is §4.3, not a
euphemism here.

---

## 4. Evaluation

We ask three questions: does witness binding catch real errors, what does it
cost, and where does it stop.

### 4.1 Does it catch real errors?

The motivating error was not synthetic. The flip was discovered as an
unexplained anomaly (rung R14: a perturbation that killed the contract's
verdict at levels 4–7 and survived at 8), characterised as count-preserving
with controls (R15), explained as partition-preserving (R16), and reduced to a
single open equivariance lemma (R19). R14's kill pattern and §2.4's
count-preservation at $n = 5, 6, 7, 8$ do not conflict, because they are
statements about different propositions: R14's dying contracts were the
spectral-classifier contracts, whose proposition is the spectrum *as a
complete invariant* — a different, finer proposition than the count law —
while R15's count-preservation is measured against the coarser cardinality
proposition of §2.4, which the flip preserves at every level tested. Only then
was the mechanism built (R17) and the real claim bound (R18). The error class
the mechanism catches is one the corpus actually produced, on its own
load-bearing contract, at the boundary of its own analysis — the contract
checks its levels jointly and the tower is recursive, so a count-preserving
flip below the top level has a second chance to be caught higher up, and the
blind spot sits at $n = 8$ precisely because that is the only level with
nowhere higher to look [15, §1.2].

Against the corpus's *history*, we claim nothing: rung R4 measured the
historical arms at zero for the earlier grades of binding, and nothing here
re-runs that [13, §3]. The three audited self-corrections in the repository's
history were interpretive — claim and check wrong together — and witness
binding would have caught none of them, by the same scope limit that bounds
every grade (§2.5). The honest statement is: witness binding catches a failure
class that is (i) real, (ii) exhibited on a production contract, and (iii) not
the class that has historically damaged this corpus.

The demonstration is live rather than archival: the perturbed twin gate is kept
in the repository and is part of the R17 gate's compile arm, so "the mechanism
still discriminates the real tower from its twin" is re-measurable on demand.

### 4.2 What does it cost?

Three components, measured or bounded:

- **Fingerprinting.** A SHA-256 over an already-computed object; in the
  production gate the hash is over the enumerated spectra and costs nothing
  measurable against the computation it fingerprints (3.4 s wall-clock for the
  whole gate, dominated by spectrum enumeration). Witness binding never adds
  asymptotic cost: the evidence exists before it is fingerprinted.
- **Comparison.** One string equality per claim, after the token decision.
  Negligible.
- **The real cost is the check itself, and it is serial.** The executor runs
  gates one subprocess at a time; a build pays the sum of its claims' gates.
  Rung R1 measured 15 gates ≈ 30 s, and five of twenty sampled gates exceed
  the 30 s per-gate budget and cannot be bound at all [17, §3]. Witness
  binding inherits this unchanged — the production claim's gate fits in 3.4 s
  precisely because its $n = 8$ arm was excluded.

### 4.3 What are the limitations?

Each of these is recorded in the rung specs and survives into the paper.

- **Opt-in, and almost nothing is bound.** One claim of ~295 declares a
  witness. The mechanism's value at corpus scale is unmeasured, not negative.
- **Shared misinterpretation stands** (§2.5). A witness binds *which* evidence;
  it cannot say the evidence is well-founded.
- **The module-closure wall is gone, and it was not witness binding that
  removed it.** At R1 claims executed only in the main source file, so a library
  whose scientific premise had been refuted passed silently into every dependent
  build. R29 gave claim verification a walk over the module closure
  (`MODULE_CLOSURE_PASSES` [17, §2]). Because a witness is checked inside the
  per-claim loop that walk now wraps, witnesses are checked wherever claims are:
  under `--verify-claims`, anywhere in a build's transitive import closure. The
  distinction worth keeping is about *credit*, not scope — nothing in the witness
  mechanism widened anything; it inherited a wider domain from a change made
  elsewhere, and would still run only on the main file had that change not
  landed. The limitation this bullet was really recording therefore stands
  untouched and is the first one in this list: **one claim of ~295 declares a
  witness**, so what the wider domain currently reaches is one claim.
- **The capture path is fixed, and the race is real but degrades fail-closed.**
  The executor captures every gate's output at one container-wide path, so two
  concurrent `--verify-claims` compiles can interleave and a build can read
  another build's capture. The per-process fix died on a hard constraint: the
  self-hosted compiler SIGSEGVs on runtime string building in this code path
  [16, §3.1], so a `/tmp/…<pid>.out` path cannot be constructed at all, and the
  correct common case was preferred to a broken compiler [16, §5; 13, §3].
  "Unresolved" alone overstates the hazard, so we bound it precisely. Exit
  statuses are never read from the capture — each build waits on its own gate
  subprocess — so exit-code gating is immune; only token and witness
  *extraction* reads the capture. A clobbered capture is almost always
  fail-closed: it lacks this claim's token (`CLAIM_TOKEN_ABSENT`) or carries
  the wrong one (`CLAIM_TOKEN_MISMATCH`), and the build is refused. A false
  *accept* requires the concurrent build's gate to emit exactly this claim's
  declared token and witness — for a witness-bound claim a 256-bit SHA-256
  equality, and even for token-only claims it requires both compiles to share a
  gate-output convention and a declared token, which in practice means two
  compiles of the *same* source, where both runs pose the same question to the
  same gates and a swapped capture answers it correctly. The residual hazard —
  concurrent compiles of *different* sources with coinciding token conventions —
  is real but narrow, and this workspace's parallel agents are the population
  at risk. The operational mitigation is serialisation: an `flock` around
  `--verify-claims` invocations, or separate containers. The designed
  engineering fix — a fixed-path lock file taken with `O_EXCL` before the claim
  loop, which needs no string construction and so avoids the segfault that
  killed the PID variant — is recorded as future work rather than shipped:
  shipping it means rebuilding the compiler and re-certifying R17's behaviour
  receipt, and the window above did not justify that cost on the day.
  Unresolved, but bounded.
- **The fingerprint is authored.** Nothing computes it. An author who declares
  the fingerprint of the wrong evidence binds the wrong evidence, exactly.
- **Single corpus.** The case study is one repository, one contract, one
  exhibited group element. Proposition 2.9 and Corollary 2.13 say the class is
  general; our evidence that it *occurs* at scale is one measured family.

### 4.4 Threat model

Witness binding is a guard, and a guard is only as honest as its statement of
who it guards against.

**Who is the adversary?** The principal adversary is not a person; it is
*drift* — the world, the pipeline, or the author's own later edit replacing the
evidence under a still-true proposition. Concretely: a dependency update that
changes an enumeration order or a rounding; a refactor of the check that
silently swaps which objects get counted; an upstream dataset regenerating with
the same cardinality and different contents. This adversary has no goals and
exploits no code, and it is the adversary the corpus actually produces (§4.1).
The active adversary is bounded by trust already placed elsewhere: a claim's
gate runs as a subprocess of the compiler, so whoever controls the gate
controls the exit code, the verdict token, and the emitted witness alike. An
attacker who can edit the gate can emit the declared fingerprint without
computing anything, and no mechanism at this layer can prevent that — the gate
is the trusted computing base. Nor is the gate the whole of it: the base
includes the executor, the shell that runs the gates, the string-scraping
extraction convention, the shared capture path, and — because Sounio is
self-hosted — a previous incarnation of the very compiler doing the checking,
the bootstrap circularity of Thompson's trusting-trust lecture, which no audit
of the current compiler's source closes [25]. What the fingerprint adds against an active
adversary is collision resistance and nothing more: an attacker who can move
the evidence but *not* the gate must find a SHA-256 collision with the declared
fingerprint to pass. Against drift that strength is ample; against a motivated
forger the mechanism is exactly as strong as SHA-256, and we claim no more
(§2.4, caveat 1).

**What the fingerprint protects against.** (i) Evidence replacement under a
preserved proposition — the paper's subject, elements of
$\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ realised by accident. (ii) Silent
regeneration of computed objects by a changed pipeline, including
nondeterminism: a nondeterministic gate mismatches against itself on
recompilation, surfacing its own nondeterminism as a build failure rather than
letting it through. (iii) Partial edits — a claim whose prose or proposition
was adjusted to stay true while its evidence base moved underneath.

**What it does not protect against.** (i) Shared misinterpretation: claim and
check wrong together agree on a witness (§2.5). (ii) A compromised or lying
gate (above). (iii) Every claim that declares no witness — the mechanism is
opt-in, and ~294 of ~295 claims in the corpus are outside the model. (iv) The
capture-path race (§4.3): concurrent compiles can read each other's captures;
the failure degrades fail-closed except in the narrow window analysed there,
and the mitigation is serialisation. (v) Joint rollback of evidence *and*
fingerprint: the manifest is versioned, so a moved fingerprint is visible in
review, but nothing stops an author updating claim and witness in one commit —
which is why the protocol below treats a fingerprint update as a review event.

**False-positive protocol.** A `CLAIM_WITNESS_MISMATCH` has exactly two
readings. On unchanged intent it is the alarm the mechanism exists to raise:
the evidence moved, and the build stops until a human finds out why. On
intended change it is the mechanism's maintenance cost, and the response is a
*witness update*: re-run the gate standalone to confirm the new evidence is
deterministic and is the evidence meant, record the new fingerprint in the
claim in the same commit as the change that moved the evidence, and let review
see the fingerprint's diff beside the code's diff — a witness update is a
review event of the same gravity as editing the claim itself. The cost is real:
every legitimate evidence migration pays it, and §5's witness schemas (typed,
diffable witnesses) are the planned relief, because today the mismatch says
*that* the evidence moved, not *how*.

---

## 5. Discussion

**When witness binding is necessary.** Whenever a claim's proposition
aggregates over computed objects — counts, cardinalities of classifications,
existence statements, optima reported without their argmins — and the build's
correctness depends on *which* objects, not just how many. Scientific computing
supplies these claims wholesale; so do learned-artifact pipelines (a model that
reports the same metric on different weights), data-dependent code generation,
and any contract of the form "exactly $N$ equivalence classes", which
Corollary 2.13 singles out. The stronger the classification theorem, the
coarser its token.

**When it is overkill.** When the proposition already *is* the evidence —
boolean properties of the program text, type-correctness, absence of a flagged
pattern — the stabiliser and the invariance group coincide on the reachable
states, Theorem 2.12's strictness condition fails, and token binding (or a type
system) is already exact. Witness binding buys nothing where $p$ does not
factor through a coarser $\pi$. It is also the wrong tool against the failure
mode this corpus actually exhibits: shared misinterpretation needs
independently authored evidence, not finer fingerprints [19, §3].

**The group is the design tool.** The useful habit the theory suggests is to
ask of any verified claim: *what is $\mathrm{Inv}(p)$?* If the answer includes
transformations that would alarm the claim's author, the claim is a candidate
for a witness. If the author cannot say what preserves $p$, that is itself
diagnostic — the flip was found because a perturbation survived that should not
have, and the survival was informative precisely because the contract had
controls.

**Future directions.** (i) Closing the equivariance lemma [18, §2] — why
marking $P$ and $Q$ does not refine the spectrum partition — which would
promote the case study's central measurement to a theorem. (ii) Module-closure
propagation of claims, the line's main engineering obstacle since R1.
(iii) Empirical lockfiles: recorded witness sets with replay and staleness
policies, reconciling witness-bound builds with reproducibility (the line's
RQ2 [19, §4]). (iv) Witness schemas: today the fingerprint is an opaque hash;
a typed witness (a canonical serialisation of the evidence) would let the
compiler diff *how* the evidence moved, not just that it did. (v) A corpus
study: what fraction of an empirical codebase's claims admit non-trivial
$\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$, and at what authoring cost.
(vi) The deferred engineering: raising the executor's 30 s per-gate budget (or
a resumable gate protocol) so the production claim's $n = 8$ arm can be bound
without weakening its evidence function (§3.4), and the fixed-path `O_EXCL`
serialisation of the capture path designed in §4.3 — both are compiler rebuilds
with receipt re-certification, and belong in one batch.

---

## 6. Related work

**Proof-carrying code and certifying compilers.** PCC attaches to a binary a
proof that it satisfies a safety policy, checked by the consumer [1, 7];
certifying compilers generate such proofs automatically [4]. The evidence
concerns the *program*; the checker validates a *proof*. Witness binding
attaches evidence about the *world the program's claims describe*, and the
compiler validates *identity*. The mechanisms are complementary: PCC answers
"is this artifact safe?", witness binding answers "is this the evidence this
claim meant?".

**Certified compilation and translation validation.** CompCert carries a
machine-checked proof that compilation preserves semantics [2]; translation
validation checks each run of a compiler against its input [5]. Both bind
artifact to source. Our binding is orthogonal in direction: it binds artifact
to *computation about the world*, and it is deliberately non-hermetic — the
same source can fail to compile on a different day because the world changed,
which the line treats as the feature, not the bug [19, §4].

**Certifying algorithms.** A certifying algorithm returns, beside its answer,
a witness from which a checker can verify correctness more cheaply than
recomputation — the line of work that descends from Blum and Kannan's program
result checking [26] and that McConnell et al. survey [3]. This is the closest
neighbour and the contrast is sharp:
there the witness *establishes* the proposition (the checker re-derives truth
from it); here the witness *identifies* the evidence (the builder compares
fingerprints). A certifying checker's witness is sound by construction; ours is
exact about identity and agnostic about soundness (§2.5). The two compose: a
certifying gate whose witness is both checkable and fingerprinted would give
identity *and* soundness.

**Reproducible builds and software provenance.** Reproducible-builds practice
[6] and supply-chain frameworks such as SLSA and in-toto [10, 11] bind
artifacts to their inputs by hash — the same primitive as witness binding —
towards hermeticity: the build must not depend on the uncontrolled world.
Witness-based compilation inverts the goal: the build *must* depend on the
claimed face of the world, and refuse when that face is replaced. The
reconciliation — reproducibility *relative to a witness set* — is the
empirical-lockfile design deferred by the line (§5).

**Hash-pinned fetching: Nix FODs, Bazel, go.sum.** Three deployed systems use
the declared-hash primitive, and the comparison localises what is new here. A
Nix *fixed-output derivation* declares the hash of a derivation's output in
advance, and the build fails if the realised output differs [21]; Bazel's
repository rules pin fetched archives (`http_archive(sha256=…)` and
`repository_ctx.download(sha256=…)`) [22]; Go's `go.sum` records module-content
hashes and refuses a module whose contents changed under a fixed version [23].
All three bind a build to the identity of *fetched or produced inputs*, in
service of hermeticity and supply-chain integrity — the pin exists so that
everything else can be sandboxed. Two differences mark witness binding. First,
*referent*: the fingerprint is of evidence for a proposition, computed fresh by
re-running the check at each compile — not of a stored artifact fetched once —
and the check must still run and the proposition must still hold, so witness
binding binds $p$ and $w$ jointly where the pins bind $w$ alone (a `go.sum`
entry says nothing about what the module *does*). Second, *direction*: pins
exist to make the build independent of the world — the same name must forever
yield the same bytes, and a mismatch is a supply-chain incident — where witness
binding exists to keep the build dependent on the claimed face of the world,
and a mismatch is the intended signal. `go.sum` is the closest in spirit: a
`go.sum` mismatch *is* "the world moved under a fixed name". But its names are
version strings, not propositions, and its enforcement lives in the toolchain's
fetch path, not in a compiler's verdict on a claim about the world.

**Metamorphic testing.** Metamorphic testing is the field that asks, of a
program whose correctness has no test oracle, *which transformations of the
input must preserve — or predictably change — the output*, and turns the
answers (metamorphic relations) into follow-up test cases [24]. This is exactly
the group question of §2 read in the opposite direction. Metamorphic testing
*exploits* $\mathrm{Inv}(p)$ to manufacture new tests from old ones where no
oracle exists; witness binding treats
$\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ as the region where an oracle that
*does* exist — the proposition — is provably insufficient, and binds the
witness so that transformations in that region are refused rather than
exploited. The two compose: a metamorphic relation a gate is expected to
satisfy is a candidate witness, and §2.5's open problem — the group is
characterised, not computed — is the metamorphic tester's familiar situation
that useful relations are found, not enumerated. The difference of setting
remains: metamorphic relations are properties of a program's input–output
behaviour, checked by a test runner after the fact; a witness is an identity
statement about a claim's evidence, checked by the compiler before the
artifact exists.

**N-version programming and recovery blocks.** Avizienis's N-version systems
tolerate faults by independent reimplementation and voting [8]; the line's own
measurements (rungs R6–R13 of the programme draft [12]) bound how independent
"independently authored" checks in one corpus actually are. Witness binding is
not redundancy: it adds no second opinion, it sharpens the first one's
*resolution*.

**Abstract interpretation.** The formalism of §2 is, in the language of
abstract interpretation, a statement about kernels: $p$ is an abstraction of
$w$, $\mathrm{Inv}(p)$ is the symmetry group of the abstract domain, and
witness binding is refinement of the domain [9]. We chose the group-theoretic
presentation because the blind spot *is* a group, and because "what preserves
the proposition?" is the question a claim author can actually be asked to
answer.

**Snapshot testing and golden masters.** Golden-master testing binds a build
to literal output. Witness binding differs in what is frozen: not the output
text but a semantic fingerprint chosen by the claim's author, declared beside
the proposition it grounds, and enforced by the compiler rather than the test
runner.

**Self-falsifying compilation.** The immediate context is the repository's own
line: R0's drift/misinterpretation distinction and scope-limit proposition
[19], R2's verdict-token binding [16], R1's binding of real gates and the
module-closure wall [17], and the R14–R19 arc this paper reports [20, 15, 14,
13, 18]. The line's programme draft [12] reports the whole arc; this paper
extracts and generalises its R15–R18 segment, whose content — the theory of
witness blindness and the mechanism that closes it — is independent of the
programme's other results.

---

## 7. Conclusion

A verifier bound to a proposition is blind to exactly the proposition's
invariance group; for the aggregate propositions by which scientific software
actually states its claims, that group contains transformations an author would
call errors — measured, in our case, as a single sign flip that replaced every
spectrum while their count held at 24. The repair, stated plainly, *is* a
finer proposition: $p_w(s) = [\,f(w(s)) = h\,]$ is a predicate on states, so
on this paper's own definitions (Definition 2.3) witness binding is token
binding at the refined proposition that names the evidence's fingerprint —
Theorem 2.12's strict refinement says exactly this, and we do not pretend
otherwise. What is new is not the logic but the convention. For an empirical
claim the refinement is *canonical* — the proposition determines the evidence
it factors through, so there is a designated finer proposition rather than an
arbitrary one — and its reference value is one a human cannot author unaided,
because the fingerprint is computed, not intuited: the check computes it, the
claim declares it, and the compiler refuses the build when the grounds move
beneath a true verdict. The contribution is the engineering discipline of
binding to the witness — the fingerprint declared beside the proposition it
grounds, versioned with it, reviewed with it, and enforced before the artifact
exists — not a new grade of predicate.
Soundness and completeness come cheap (Theorem 2.11 is one line once the
fingerprint is idealised); the honesty costs are elsewhere — the witness cannot
say the evidence is well-founded, and almost nothing in the corpus is bound
yet. The proposition is not the evidence. Compilation can now tell the
difference; the remaining work is making it matter at scale.

---

## 8. References

[1] G. C. Necula. Proof-carrying code. In *Proc. POPL*, 1997.

[2] X. Leroy. Formal verification of a realistic compiler. *Communications of
the ACM*, 52(7):107–115, 2009.

[3] R. M. McConnell, K. Mehlhorn, S. Näher, and P. Schweitzer. Certifying
algorithms. *Computer Science Review*, 5(2):119–161, 2011.

[4] G. C. Necula and P. Lee. The design and implementation of a certifying
compiler. In *Proc. PLDI*, 1998.

[5] A. Pnueli, M. Siegel, and E. Singerman. Translation validation. In
*Proc. TACAS*, 1998.

[6] C. Lamb and S. Zacchiroli. Reproducible builds: increasing the integrity
of software supply chains. *IEEE Software*, 39(2):62–70, 2022.

[7] A. W. Appel. Foundational proof-carrying code. In *Proc. LICS*, 2001.

[8] A. Avizienis. The N-version approach to fault-tolerant software. *IEEE
Transactions on Software Engineering*, SE-11(12):1491–1501, 1985.

[9] P. Cousot and R. Cousot. Abstract interpretation: a unified lattice model
for static analysis of programs by construction or approximation of fixpoints.
In *Proc. POPL*, 1977.

[10] SLSA: Supply-chain Levels for Software Artifacts. https://slsa.dev
(accessed 2026-07-28).

[11] S. Torres-Arias et al. in-toto: providing farm-to-table guarantees for
bits and bytes. In *Proc. USENIX Security*, 2019.

[12] Sounio self-falsifying compilation line. *Where Did the Evidence Come
From? Compile-Time Claims, Their Limits, and Measuring Whether a Corpus
Corroborates Itself.* Draft, `docs/papers/oopsla2027/paper.md`, 2026.

[13] Sounio self-falsifying compilation line, rung R17. *Witness binding, in
the compiler.* `docs/research/self_falsifying_compilation_line_r17_2026-07-28.md`,
2026. (Rung R18 — the first production claim bound to a witness — is §1.3
thereof.)

[14] Sounio self-falsifying compilation line, rung R16. *The invariance group,
identified: partition-preserving, not merely count-preserving.*
`docs/research/self_falsifying_compilation_line_r16_2026-07-28.md`, 2026.

[15] Sounio self-falsifying compilation line, rung R15. *A verdict token is
blind to whatever preserves the truth of its proposition.*
`docs/research/self_falsifying_compilation_line_r15_2026-07-28.md`, 2026.

[16] Sounio self-falsifying compilation line, rung R2. *Verdict-token binding:
the compiler now checks the proposition, not just the exit code.*
`docs/research/self_falsifying_compilation_line_r2_2026-07-26.md`, 2026.

[17] Sounio self-falsifying compilation line, rung R1. *Binding the corpus: 15
real gates bound, and the module-closure wall measured.*
`docs/research/self_falsifying_compilation_line_r1_2026-07-26.md`, 2026.

[18] Sounio self-falsifying compilation line, rung R19. *R16's locality
derived, and what is left reduced to one lemma.*
`docs/research/self_falsifying_compilation_line_r19_2026-07-28.md`, 2026.

[19] Sounio self-falsifying compilation line, rung R0. *The substrate is live,
the corpus was unbound, and the failures it must catch are interpretive.*
`docs/research/self_falsifying_compilation_line_2026-07-26.md`, 2026.

[20] Sounio self-falsifying compilation line, rung R14. *The perturbation that
survived only at the boundary.*
`docs/research/self_falsifying_compilation_line_r14_2026-07-27.md`, 2026.

[21] E. Dolstra, M. de Jonge, and E. Visser. Nix: a safe and policy-free system
for software deployment. In *Proc. LISA*, 2004. (Fixed-output derivations:
`outputHash` / `fetchurl`, Nix and Nixpkgs manuals.)

[22] Bazel. *Repository rules and `repository_ctx.download` (`sha256`)*. Bazel
documentation, https://bazel.build (accessed 2026-07-28).

[23] The Go Authors. *Go Modules Reference: `go.sum` and the checksum
database.* https://go.dev/ref/mod (accessed 2026-07-28).

[24] T. Y. Chen, F.-C. Kuo, H. Liu, P.-L. Poon, D. Towey, T. H. Tse, and
Z. Q. Zhou. Metamorphic testing: a review of challenges and opportunities.
*ACM Computing Surveys*, 51(1):4:1–4:27, 2018.

[25] K. Thompson. Reflections on trusting trust. *Communications of the ACM*,
27(8):761–763, 1984.

[26] M. Blum and S. Kannan. Designing programs that check their work.
*Journal of the ACM*, 42(1):269–291, 1995.

[27] R. Pollack. How to believe a machine-checked proof. BRICS Report Series
RS-97-18, University of Aarhus, 1997. (Also in *Twenty Five Years of
Constructive Type Theory*, Oxford Logic Guides 36, Oxford University Press,
1998.)

---

## 9. AI disclosure

This paper was drafted under human direction (2026-07-28), consistent with
GAIDeT-ICMJE 2025. The theory of §2 was developed against the rung specs it
cites; the elementary proofs (Propositions 2.5, 2.7, 2.9; Theorems 2.8, 2.11,
2.12) were checked by hand and then reviewed by orthogonal LLM providers under
the repository's offload policy (`math-review` for §2, multi-provider raw
review for the full draft), with outcomes logged in `.claude/llm_offload_log.md`.
Every empirical figure cites the rung that measured it, and the companion gate
`scripts/ci/witness_based_compilation_paper_gate.sh` fails if a cited verdict
token drifts from its spec, if the witness fingerprints quoted here drift from
the claim in the manifest, or if the measured/derived status distinctions of
§2.3 are erased. The real and perturbed witness gates were re-run on
2026-07-28 in the drafting session; the fingerprints in §3.4 are from those
runs. A peer-review round on 2026-07-28 produced five revisions (the setwise
stabiliser of Proposition 2.9, the $n = 8$ exclusion argument, the threat
model of §4.4, the prior-art engagements of §6, and the bounded analysis of
the capture-path race in §4.3); the revised Proposition 2.9 was re-reviewed
under the same math-review offload, and its one flag is addressed in the text
and logged with the rest. A re-review the same day produced three further
corrections: the §4.1 reconciliation of R14's kill pattern with §2.4's
count-preservation (different propositions, not a contradiction), the §7
concession that witness binding is formally token binding at a refined
proposition — the novelty being the convention, not the logic — and the
Thompson, Blum–Kannan, and Pollack citations. A second math-review offload over those corrections (xai, zai)
confirmed all three and caught one pre-existing imprecision: Theorem 2.12's
strictness condition now requires the witness to move *at a reachable state*,
since membership in $\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ alone moves
the witness only somewhere. No clinical content.
