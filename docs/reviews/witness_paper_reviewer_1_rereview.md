<!-- docs:meta
topic_id: repo.docs.reviews.witness-paper-reviewer-1-rereview
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.reviews.witness-paper-reviewer-1-rereview
-->

# Re-Review — "Witness-Based Compilation: When the Verdict Is Right but the Evidence Is Wrong"

**Reviewer:** 1 (PL theory), second round
**Paper:** `docs/papers/witness_based_compilation_2026-07-28.md` (revised draft, 2026-07-28)
**Scope of this round:** verify the five mandated fixes — (1) Proposition 2.9 setwise stabiliser, (2) the n = 8 exclusion, (3) the missing threat model, (4) the prior-art gap (metamorphic testing, Nix FODs, Bazel `download(sha256)`, go.sum), (5) the capture-path race — and re-assess soundness of the theory as a whole. I re-checked every proof in §2 line by line in the revised text.

---

## Verdict on the five mandated concerns

### 1. Proposition 2.9 (setwise stabiliser) — FIXED, and the fix is correct

The revised statement (§2.3) now says: a permutation of the ground set $X$ lies in $\mathrm{Stab}(w)$ iff it maps the produced subset to itself *as a set*; the stabiliser is the **setwise** stabiliser of $w(s)$ in $\mathrm{Sym}(X)$, order $N!(M-N)!$; the pointwise stabiliser (order $(M-N)!$) is a proper subgroup whenever $N \ge 2$; and the number of evidence values indistinguishable from the true one under $p$ is $\binom{M}{N} - 1$.

I verified the arithmetic:

- Setwise stabiliser of an $N$-subset of an $M$-set: $\mathrm{Sym}(N) \times \mathrm{Sym}(M-N)$, order $N!(M-N)!$. ✓
- Pointwise stabiliser: order $(M-N)!$; proper subgroup of the setwise one iff $N! > 1$, i.e. $N \ge 2$ (edge cases $N \in \{0,1\}$ and $N = M$ behave as stated). ✓
- Orbit size $= M! / (N!(M-N)!) = \binom{M}{N}$, so $\binom{M}{N} - 1$ other evidence values in the orbit. ✓
- The ratio claimed in the following paragraph — the gap between $\mathrm{Inv}(p) \supseteq \mathrm{Sym}(X)$ and the stabiliser is a factor of $\binom{M}{N}$ — matches the orbit count. ✓
- The sentence connecting the setwise reading to the implementation (the fingerprint is a hash over the *sorted* enumeration, so an internal relabelling "moves nothing — not even the fingerprint") closes the loop between theory and artifact that the wrong version had left open. This was the live part of the original error and it is now right.

The paragraph narrating the earlier draft's mistake ("conflating the witness *value* (a set) with the group elements that act on its members… shrinking the stabiliser by a factor of $N!$") is an unusual thing to print in a paper, but it is accurate, and in this repository's audit-trail culture it is defensible.

**One residual imprecision, minor.** Definition 2.6 defines $\mathrm{Stab}(w)$ *globally* — $w \circ \sigma = w$ as functions on all of $S$. Proposition 2.9's characterisation ("iff it maps the produced subset to itself") is the stabiliser of the single *value* $w(s)$. The two coincide only if the image of $w$ on the states under discussion is that one subset; with a rich image (which the paper's own phenomenon requires — the whole point is that $w$ varies across states), the global stabiliser is the intersection of the setwise stabilisers of all attained subsets, generically much smaller, often trivial. The fix the paper needed is the valuewise reading, and the text comes within one clause of saying so ("the witness is a single set value") — but Definition 2.6 and Proposition 2.9 still formally quantify over different things. One sentence — "throughout, stabilisers and orbits are read at the reference evidence value; the global stabiliser of Def. 2.6 is the intersection over attained values" — would close it. Not a blocker; the mathematics the paper uses is the valuewise reading and it is correct.

### 2. The n = 8 exclusion — FIXED (addressed with an argument, not an apology)

§3.4 now gives three reasons: (i) the exclusion is *inside the proposition* — the claim asserts the count law at $n = 5,6,7$ only, so no $n = 8$ behaviour can pass silently; a future $n = 8$ claim is a new binding, not a silent extension. (ii) The phenomenon is level-agnostic — the flip is an exhibited element of $\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ at every level $n = 5,6,7,8$ (§2.4, citing R15), so the bound levels already bind the same group element acting by the same mechanism. (iii) The 30 s cap is a budget, not a boundary of the method — and the tempting workaround (precompute the $n = 8$ spectra, fingerprint the cache) would change the evidence function from *computed at this build* to *read from a file*, weakening exactly what the witness exists to bind.

Point (iii) is the load-bearing one and it is genuinely correct: a cached fingerprint binds the cache, not the computation. Point (i) is sound scoping. Point (ii) is fair given the exhibited element. The residual discomfort — a *production* claim's scope is being set by an executor constant, not by the science — is disclosed verbatim ("recorded here rather than disguised"), and the deferred fix is itemised in §5(vi). I accept this as adequate.

### 3. Threat model — FIXED (new §4.4, and it is a real one)

The new section identifies the principal adversary as drift rather than a person; bounds the active adversary by the gate-as-TCB argument (whoever controls the gate controls exit code, token, and witness alike — no mechanism at this layer can do better); enumerates what the fingerprint protects against (evidence replacement under a preserved proposition; silent regeneration including nondeterminism surfacing as build failure; partial edits) and what it does not (shared misinterpretation; a lying gate; the ~294 unbound claims; the capture race; joint rollback of evidence and fingerprint, with the mitigation that a fingerprint update is a review event). The false-positive protocol (a `CLAIM_WITNESS_MISMATCH` has exactly two readings, and the intended-change arm pays a real maintenance cost) is the part most threat-model sections skip; this one doesn't.

**Minor technical slip.** §4.4 (and §2.4 caveat 1) say the mechanism "is exactly as strong as SHA-256" up to "collision resistance". Against a *declared, fixed* fingerprint $h$, the property an active adversary must break is **second-preimage resistance**, not collision resistance — the adversary cannot choose both inputs. The distinction matters quantitatively ($2^{256}$ vs $2^{128}$ work factors) and the paper is otherwise careful about exactly this kind of quantifier. One word in each location.

### 4. Prior art — FIXED (the engagement is now substantive)

§6 gains a dedicated "Hash-pinned fetching" subsection covering Nix fixed-output derivations, Bazel's `repository_ctx.download(sha256=…)`, and go.sum, with citations [21]–[23], and a metamorphic-testing subsection with the Chen et al. survey [24]. The FOD paragraph makes the two distinctions that survive scrutiny: *referent* (the fingerprint is of evidence for a proposition, recomputed by re-running the check at each compile, so $p$ and $w$ are bound jointly — a `go.sum` entry binds bytes and says nothing about what they *do*) and *direction* (pins exist to make the build independent of the world; witness binding exists to keep it dependent). The metamorphic paragraph correctly reads the group question in the opposite direction (exploit $\mathrm{Inv}(p)$ where no oracle exists vs. delimit where an existing oracle is insufficient) and concedes the shared open problem — the group is found, not enumerated. The novelty claims in the abstract and §3.4 are now hedged ("to our knowledge"). The one-line FOD rebuttal I predicted in round 1 no longer lands.

Not carried over from round 1 (it was a specific comment, not a numbered weakness): the proof-relevance/BHK connection — "the proposition is not the evidence" is the founding distinction of constructive type theory — is still absent. For an ICFP audience this remains the most likely source of a hostile first question, and one paragraph would pre-empt it. Suggestion, not a condition.

### 5. Capture-path race — FIXED (bounded honestly, with a designed fix)

§4.3 now gives the analysis the first draft lacked: exit statuses are never read from the capture (each build waits on its own gate subprocess), so exit-code gating is immune; only token and witness *extraction* reads the capture; a clobbered capture is almost always fail-closed (`CLAIM_TOKEN_ABSENT` / `CLAIM_TOKEN_MISMATCH`); a false accept against a witness-bound claim requires a 256-bit SHA-256 equality; against a token-only claim it requires both compiles to share a gate-output convention and a declared token, which in practice means two compiles of the same source posing the same question — answered correctly by either capture. The residual hazard (concurrent compiles of *different* sources with coinciding token conventions) is named, the operational mitigation (flock, separate containers) is stated, and the designed fix (fixed-path `O_EXCL` lock needing no string construction, hence avoiding the SIGSEGV that killed the PID-path variant) is recorded as future work with the reason it was not shipped (compiler rebuild plus behaviour-receipt re-certification). The fail-closed analysis is correct as far as it goes, and "Unresolved, but bounded" is the right register.

---

## Standing concerns from round 1 not addressed in this revision

These were weaknesses 1–3 of my first review; the revision did not attempt them, and they remain the gap between this paper and a clean accept.

1. **The theory is still set theory over an unstructured state space.** No operational semantics, no program model, and — still — no definition of *reachable*, a term doing load-bearing work in Theorem 2.12 and again in §4.2. Everything proved is correct, but nothing in §2 uses anything about programs. My round-1 condition (ii) — develop the semantics or reposition — is unmet.
2. **The group framing remains vacuous at this level of generality.** On an unstructured $S$, transpositions always exist, so blindness to $\mathrm{Inv}(p)$ coincides with blindness to the fibres of $p$, and Theorem 2.12's strictness condition collapses to "two reachable states with equal $p$ and different $w$ exist". The group becomes a tool only where $S$ has structure — and the Cayley–Dickson case has plenty (the R16 partition-preservation result is exactly structure), but §2 never imports it into the formalism. As written, the group theory is a vocabulary; the paper's own case study shows it could have been more.
3. **The world-evidence / gate-evidence identification is still unmodelled.** Definition 2.2 makes $w$ a function of the state; §3.4 fingerprints what the gate *emitted*. §4.4's observation that a nondeterministic gate mismatches against itself and surfaces as a build failure is an honest partial mitigation in prose, but there is still no observation channel in the formalism, so Theorem 2.11's soundness is about the idealised $w$, not the implemented mechanism. Round-1 condition (iv) is half-met.

## Is the theory now sound?

Yes, with the two caveats recorded above. Propositions 2.5, 2.7, 2.9 and Theorems 2.8, 2.11, 2.12 are all correct as elementary mathematics (2.9 now included; 2.12's transposition converse remains the best formal work in the paper). The measured/derived/open status discipline of §2.3 is intact and, unusually, gate-enforced. What the theory *is* — a scoping theorem over an unstructured state space — is smaller than what the word "theory" in the abstract suggests, but nothing in it is wrong.

## Overall recommendation

**Weak accept.**

All five mandated concerns are fixed, and the two that required mathematics (Prop 2.9) and security reasoning (threat model, race analysis) are fixed *correctly*, not cosmetically — I checked the stabiliser arithmetic independently and it is right. The revision converts the paper's two factual vulnerabilities (a false quantifier in its only quantitative proposition; an exposed novelty claim) into disclosed, bounded limitations, which is exactly what a revision should do.

It is weak accept rather than accept because the three structural objections from round 1 stand: the formalism has no semantics and no reachability, the group framing buys nothing over the fibre partition at this generality, and the gate-as-observation-channel gap means the proved theorem is one idealisation away from the shipped mechanism. Against that, the phenomenon is real and exhibited on a production contract with controls, the mechanism is implemented and live-tested, the reporting discipline remains the best I have seen in an empirical PL-adjacent draft, and the paper's claims are now calibrated to its evidence. For a theory-track reading my round-1 conditions (ii) and (iv) are unmet and I would hold at weak reject; read as the systems/experience-plus-scoping-theorem contribution the revised text in fact is, it clears the bar. On balance: weak accept, with the Prop 2.9 valuewise-reading sentence and the second-preimage correction requested as minor revisions.

**Confidence:** 4/5 on the theory assessment (every proof in §2 re-checked line by line; the remaining flags are certain but minor). 3/5 on novelty — the hash-pinning engagement is now adequate within my knowledge, but I still cannot rule out a closer precedent in the build-systems literature.

---

*Review notes for the parent agent: this re-review was authored directly at `docs/reviews/witness_paper_reviewer_1_rereview.md`. I read the full revised paper (968 lines) and my round-1 review (`docs/reviews/witness_paper_reviewer_1.md`); I did not re-run the witness gates or the companion paper gate, so the empirical claims are taken on the paper's citations, as in round 1. No files other than this review were modified; no commits were made.*
