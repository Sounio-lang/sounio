<!-- docs:meta
topic_id: repo.docs.reviews.witness-paper-reviewer-5-rereview
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.reviews.witness-paper-reviewer-5-rereview
-->

# Re-Review: "Witness-Based Compilation: When the Verdict Is Right but the Evidence Is Wrong"

**Reviewer:** 5 (skeptical / devil's advocate), re-review of the 2026-07-28 corrected draft
**Recommendation:** Weak reject
**Confidence:** 4/5

---

## Verification performed for this re-review

Unlike the first round, this time I executed the artifact:

- `bash scripts/ci/witness_based_compilation_paper_gate.sh` — **passes**
  (`WITNESS_BASED_COMPILATION_PAPER_GATE_OK`; four cited rung tokens bound,
  witness fingerprints pinned, ten load-bearing figures checked, plus a new
  `W5_PEER_REVIEW_FIXES` arm that guards the survival of this review round's
  fixes).
- `scripts/ci/zd_fiber_spectra_witness_gate.sh` (with the repo `.venv` on
  `PATH` — the gates fail with `ModuleNotFoundError: numpy` under the system
  Python, a portability wart worth noting) — emits
  `SPECTRA_COUNT_IS_3_TIMES_2_POW_N_MINUS_5` and witness
  `705d0afd…385757`, matching both the paper (§3.4) and the manifest
  (`examples/epistemic/rupture_claims_verified.sio:188`).
- `scripts/ci/zd_fiber_spectra_witness_perturbed_gate.sh` — exits 0, emits the
  **identical verdict token** and witness `e9f935cb…19424`, exactly as §3.4
  claims. The proposition-is-preserved/evidence-is-replaced demonstration is
  real and reproducible.
- Witness extraction/comparison exists in
  `self-hosted/compiler/claim_executor.sio` (`ce_extract_witness`,
  `ce_witness_outcome`, `CLAIM_WITNESS_MISMATCH`/`ABSENT` paths at lines
  ~213–591), with the single shared scan body the paper claims.
- I re-checked the arithmetic of the revised Proposition 2.9 by hand.

The empirical core of the paper checks out. What follows is about whether the
fixes address the fundamental concerns, not whether the demo works — it does.

---

## Disposition of the five previous concerns

### 1. Proposition 2.9 (setwise vs pointwise stabiliser) — FIXED, correctly

The revised statement is right. For set-valued evidence $w(s) \subseteq X$,
$|X| = M$, $|w(s)| = N$: the condition $w \circ \sigma = w$ asks that the set
*value* be preserved, so the stabiliser in $\mathrm{Sym}(X)$ is the setwise
stabiliser of order $N!\,(M-N)!$; the pointwise stabiliser (order $(M-N)!$) is
the proper subgroup appropriate only to *ordered* evidence. The orbit count
$\binom{M}{N} - 1$ and the consistency check
$|\mathrm{Sym}(X)|/|\mathrm{Stab}| = \binom{M}{N}$ both verify. The prose even
diagnoses the earlier error precisely (conflating the witness value with the
group elements acting on its members, shrinking the stabiliser by $N!$). This
is the model fix: not patched over, but explained. No residual issue.

### 2. The n=8 exclusion — ADDRESSED, with a genuine argument (partial residual sting)

§3.4 now gives three arguments rather than an apology: (i) the exclusion is
*inside the proposition* — the claim asserts $n = 5,6,7$ and nothing beyond, so
no $n=8$ behaviour can pass silently; (ii) the exhibited flip is an element of
$\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ at every level $n=5,6,7,8$, so the
bound levels bind the same group element acting by the same mechanism; (iii)
the obvious workaround — precompute the $n=8$ spectra and fingerprint the cache
— would silently change the evidence function from *computed at this build* to
*read from a file*, weakening exactly the property the witness exists to bind.

Point (iii) is genuinely good and shows the authors understand their own
framework. Point (i) is sound. Point (ii), however, leans on the equivariance
lemma that is **still open** (§2.3, R19): "the same mechanism at every level"
is measured, not proved, and the paper says so. And the residual sting from my
first review stands in reduced form: the abstract's headline numbers (126 of
128 fibres, 24 spectra, the set wholly exchanged) are $n=8$ measurements, while
the deployed claim fingerprints $n=5,6,7$. The deployed artifact guards smaller
instances of the phenomenon, not the exhibited one. The argument makes this
defensible; it does not make it disappear.

### 3. Threat model — FIXED

§4.4 is a real threat model, not a paragraph of reassurance. It names the
primary adversary (drift, not a person), places the gate inside the TCB with
the correct conclusion (an attacker who controls the gate controls everything;
no mechanism at this layer can help), bounds the active-adversary strength
exactly (collision resistance of SHA-256 and nothing more), enumerates what is
and is not protected (including the capture-path race and joint rollback of
evidence + fingerprint in one commit), and — answering my Q4 directly —
specifies a false-positive protocol: a witness update is a review event of the
same gravity as editing the claim, recorded in the same commit as the change
that moved the evidence. The habituation concern from my first review (W5) is
acknowledged as the mechanism's maintenance cost rather than waved away, with
witness schemas (§5, item iv) as the planned relief. Adequate.

### 4. Prior art — MOSTLY FIXED (one notable gap remains)

§6 now engages everything I named: Nix fixed-output derivations, Bazel's
`download(sha256=…)`, `go.sum` (each with the two-axis differentiation —
*referent*: recomputed evidence for a proposition, binding $p$ and $w$ jointly
where pins bind $w$ alone; *direction*: anti-hermetic on purpose), and
metamorphic testing, with the correct framing: metamorphic testing *exploits*
$\mathrm{Inv}(p)$ where no oracle exists; witness binding delimits where an
oracle that exists is insufficient. The metamorphic paragraph is the best
related-work writing in the paper and concedes the right things (§2.5's "the
group is characterised, not computed" is the metamorphic tester's familiar
situation).

Two residuals. First, the build-systems-à-la-carte / self-adjusting computation
literature (Mokhov et al.; Acar et al.) is *still* absent, and it is the
literature whose foundational primitive is "hash a computed value to decide
whether downstream artifacts may be emitted." The "direction" axis does not
clear this: anti-hermeticity is a design-goal difference, and an empirical
lockfile (their own future-work item iii) is exactly the incremental-build
primitive pointed at non-source inputs. Second, the "binds $p$ and $w$
jointly" delta is thinner than presented — a Nix FOD whose build script runs a
check before producing output also binds both in practice. The novelty claim
("first production claim whose build is bound to a cryptographic hash of its
evidence") remains unfalsifiable, and "production" remains a claim inside the
authors' own research compiler. The engagement is now honest and specific; the
delta is still small.

### 5. Concurrency — ADDRESSED honestly; not fixed in code

The §4.3 analysis is precise and, as far as I can check, correct: exit statuses
are never read from the capture (each build waits on its own subprocess), so
exit-code gating is immune; only token/witness extraction reads the capture; a
clobbered capture degrades fail-closed (`CLAIM_TOKEN_ABSENT`/`MISMATCH`)
except when a concurrent build's gate emits exactly this claim's declared token
and witness — a 256-bit equality for witness-bound claims. The residual hazard
(concurrent compiles of different sources with coinciding token conventions) is
named, the population at risk is named (this workspace's own parallel agents),
and the designed fix (a fixed-path `O_EXCL` lock file needing no string
construction, thus dodging the segfault that killed the PID-path variant) is
recorded as deferred with a reason (compiler rebuild + receipt
re-certification). One glib spot: for token-only claims the paper argues a
swapped capture "answers it correctly" when both compiles share source — but
two compiles of the same source at different commits is precisely the CI race,
and §4.4 itself raises nondeterministic gates, which would break that argument.
Narrow, but the paper overstates its own bound by one sentence. Shipping a
verification-path race in a paper about verification rigour remains an
uncomfortable look; the analysis, however, is now worthy of the venue.

---

## The fundamental concern: is this still a specification-granularity problem dressed as theory?

Partially. The revision sharpened the paper's best answer to my first-review
Q1 (what does this add over "declare the full evidence set and compare"?):
**the theory is a diagnostic, not a mechanism.** §5's "the group is the design
tool" — ask of any verified claim *what is $\mathrm{Inv}(p)$?*; if the answer
contains transformations that would alarm the author, bind a witness; if the
author cannot say what preserves $p$, that is itself diagnostic — is a real,
if small, conceptual contribution: a characterisation of *which* propositions
need witness binding (aggregate ones, per Prop 2.9 and Cor 2.13) and which do
not (per Theorem 2.12's strictness condition and §5's "when it is overkill").
That is more than "hash the evidence." It is not much more.

The deciding weaknesses from the first review are structural and survive every
edit:

- **The theory remains below the venue's floor where it is not a restatement.**
  Propositions 2.5 and 2.7 (bijections preserving a predicate form a group) are
  exercises. Theorem 2.8 is the problem statement with a group wrapper; the
  paper says "deliberately elementary — a scoping statement," which is honest
  and also the point: a scoping statement is a remark, not a theory section.
  Theorem 2.11 is one line *after* idealising the hash as injective — the only
  step with content is assumed away. Theorem 2.12's converse still conjures a
  transposition to witness strictness, i.e., strictness fails iff the two
  verifiers disagree somewhere on reachable states — the definition restated.
  Corollary 2.13 is an aphorism. The abstract-interpretation framing in §6
  ($p$ is an abstraction of $w$; witness binding is domain refinement) remains
  the correct home for all of it, and in that framing §2 is a paragraph.
- **The evaluation is still one self-produced family.** One repository, one
  contract, one exhibited group element, found by the authors, catching a
  failure class that — by their own R0/R4 accounting — has never actually
  damaged the corpus, while the class that did (shared misinterpretation) is
  provably out of reach. The case study's most surprising quantitative claim
  (partition preserved, every spectrum moved) rests on measurement at
  $n \in \{5,6,7\}$ plus an **open** equivariance lemma. One claim of ~295 is
  bound. Nothing here changed, because nothing here could be changed by
  revision; it is what the work currently is.
- **The group formalism still only covers bijections.** Definition 2.4 requires
  invertible transformations of a fixed state set. The drift the threat model
  actually worries about — dependency updates, dataset regeneration, refactors
  — is not bijective; it is arbitrary, often many-to-one. The *mechanism*
  catches non-bijective drift (a hash compares values, not orbits), so
  practice is broader than theory — but then the theory is not what licenses
  the mechanism, and the paper never says how the group characterisation
  degrades, or even that it does. One paragraph would fix this; it is not
  present.

## New minor concerns introduced by the revision

1. **Canonicalisation is still under-specified.** §3.4 now says the hash is
   "over the spectra themselves, sorted," which answers the ordering half of my
   first-round canonicalisation point. The encoding half — float formatting,
   serialisation, platform independence of the hashed bytes — is undiscussed.
   A witness that mismatches across platforms for identical evidence would be
   the mechanism's first false positive in the wild. Witness schemas (§5, iv)
   gesture at it.
2. **Gate portability.** The production and perturbed gates crash under a
   bare system Python (no numpy), exiting via the shell's `set -euo pipefail`
   in a way my run reported as RC=0 from the *pipeline* tail. A gate whose
   environment failure mode is not a hard, loud non-zero is exactly the kind of
   surface/behaviour gap the paper itself warns about (§3.2). Worth one
   hardening pass before camera-ready of anything.
3. **The paper gate grew a `W5_PEER_REVIEW_FIXES` arm.** The consistency
   apparatus now also polices the survival of this review round's fixes. I
   noted last time that consistency with one's own specs is not falsification;
   extending the apparatus to the review process itself is more of the same
   theatre, however internally consistent.

## What the revision got right (credit where due)

- Every fix is *load-bearing*, not cosmetic: the setwise correction changes
  the arithmetic and the prose owns the earlier error; the n=8 argument's
  third point (caching would weaken the evidence function) is a genuinely
  framework-internal insight; the threat model's false-positive protocol is
  operational, not rhetorical.
- I could reproduce the paper's central demonstration end-to-end: real gate
  and perturbed twin emit identical tokens and different witnesses, matching
  the manifest to the hex digit. Few PLDI submissions' artifacts survive a
  skeptical reviewer actually running them.
- The measured/derived/open status discipline (§2.3) is exemplary and the
  revision preserved it under edit, with a gate arm ensuring it stays
  preserved.

## Overall

**Recommendation: Weak reject.** All five concrete concerns from the first
round are addressed — four fixed, one (concurrency) honestly bounded and
deferred — and the artifact verifies under execution, which moved me off a
firm reject. What did not move: the theory is correct and almost content-free,
the evaluation is a single self-inflicted family whose flagship deployment
does not cover the motivating level, and the mechanism remains a well-placed
hash pin whose delta over `go.sum`/FODs plus one metamorphic-testing
reframing is real but small. At PLDI/OOPSLA/ICFP main track this is still a
specification-granularity observation with an excellent case study, now
exceptionally honest about its boundaries. At a workshop, or as a short
experience report with §2 recast as a one-paragraph abstraction-refinement
remark, I would accept it without hesitation. The path to main track is
unchanged from my first review: a corpus study showing
$\mathrm{Inv}(p)\setminus\mathrm{Stab}(w)$ is non-trivial beyond the authors'
own contract, one external case, and a theory section that earns its notation.

**Confidence: 4/5.** I verified the revised mathematics by hand and executed
the paper gate, both witness gates, and inspected the executor and manifest.
I have not independently re-derived the R15/R16 measurements (126/128,
partition preservation) from first principles — those rest on the rungs'
own scripts, which I ran only at the gate level — and my novelty judgement
still relies on the related-work sections rather than a fresh literature
search of the incremental-computation corpus.
