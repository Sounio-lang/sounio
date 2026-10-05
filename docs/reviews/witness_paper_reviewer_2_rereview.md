<!-- docs:meta
topic_id: repo.docs.reviews.witness-paper-reviewer-2-rereview
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.reviews.witness-paper-reviewer-2-rereview
-->

# Re-review: "Witness-Based Compilation: When the Verdict Is Right but the Evidence Is Wrong"

**Reviewer:** 2 (compiler engineering)
**Round:** Re-review of corrected draft `docs/papers/witness_based_compilation_2026-07-28.md`
**Date:** 2026-07-28
**First-round review:** `docs/reviews/witness_paper_reviewer_2.md`

## Summary of re-review

The authors addressed all five of my first-round concerns, and — unusually —
each fix is verifiable in the repository rather than asserted in prose. I
spot-checked the revised Proposition 2.9 by hand, confirmed the witness
fingerprint in §3.4 matches the bound claim in
`examples/epistemic/rupture_claims_verified.sio:188`
(`705d0afdf8e83075…385757`), and confirmed the capture-path race the paper
describes is real in the shipped executor
(`self-hosted/compiler/claim_executor.sio:138-139` hardcodes
`/tmp/sounio_claim_gate_capture.out`, with the comment at lines 128–131
recording that the PID-based variant was abandoned). The paper does not
overclaim about what I could check.

## Issue-by-issue disposition

### 1. Prop 2.9 setwise stabiliser — FIXED, and fixed correctly

This was my hardest first-round objection and the revision gets the group
theory right:

- Stabiliser of set-valued evidence under the induced $\mathrm{Sym}(X)$ action
  is now correctly identified as the **setwise** stabiliser: permute within
  the produced $N$-subset and within its complement independently, order
  $N!\,(M-N)!$. The pointwise stabiliser (order $(M-N)!$) is explicitly
  identified as a proper subgroup for $N \ge 2$. Both orders check out:
  $|\mathrm{Sym}(X)| / \binom{M}{N} = N!(M-N)!$.
- The orbit count is now consistent: $\binom{M}{N} - 1$ indistinguishable
  evidence values, matching $M! / (N!(M-N)!) = \binom{M}{N}$. The first draft
  had the stabiliser too small by $N!$ and the orbit inflated by the same
  factor; both sides of that error are repaired, and the ratios are now
  internally coherent.
- The authors also do the right rhetorical thing: instead of silently
  patching, they explain *why* the pointwise reading was wrong (conflating
  the witness value, a set, with the group elements acting on its members)
  and note the pointwise reading would be correct only for ordered/tuple
  evidence. That is the correct scoping of the correction, not an apology.

One residual nit (not a defect): the statement "the invariance group contains
… all of $\mathrm{Sym}(X)$" is the standard abuse for "contains an isomorphic
copy of $\mathrm{Sym}(X)$ acting via the induced action on $S$" — $\mathrm{Inv}(p)$
is formally a group of bijections on $S$, not on $X$. Any PLDI reader will
read it as intended; a stickler could ask for one clause. I am not asking.

### 2. n = 8 exclusion — ADDRESSED, with an argument rather than an apology

§3.4 now gives a three-point defence: (i) the exclusion is *inside the
proposition* — the claim asserts $n \in \{5,6,7\}$ and nothing beyond, so no
$n=8$ behaviour can pass silently; (ii) the phenomenon is level-agnostic — the
flip is an exhibited element of $\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ at
$n=5,6,7,8$ alike, so the bound levels bind the same group element by the same
mechanism; (iii) the 30 s cap is a budget, not a method boundary, and the
tempting alternative (fingerprinting a precomputed $n=8$ cache) would change
the evidence function from "computed at this build" to "read from a file",
weakening exactly what the witness binds.

Point (iii) is the load-bearing one and it is correct — binding a cache is
hash-pinned fetching, not witness binding, and the authors show they
understand the difference in §6. Point (i) is sound as stated. Point (ii) is
the weakest: "the same group element acts at every level" is *measured*, not
proved (the equivariance lemma is open, and §2.3 says so honestly). The
argument stands, but it inherits the measured/derived boundary the paper
itself draws. Acceptable.

### 3. Threat model — ADDRESSED, adequate for the venue

§4.4 is a real threat model, not a paragraph of boilerplate:

- Correct identification of the principal adversary as *drift* (non-agent:
  dependency updates, check refactors, upstream regeneration) rather than a
  person — this matches the corpus's actual failure history and avoids the
  usual security-theatre framing.
- Correct placement of the TCB boundary: the gate runs as a compiler
  subprocess, so a compromised gate controls exit code, token, and witness
  alike; no mechanism at this layer can do better, and the paper says so.
- Correct reduction against an active adversary: the fingerprint adds exactly
  SHA-256 collision resistance and nothing more, stated without inflation.
- The false-positive protocol (witness update as a review event, fingerprint
  diff beside code diff) is operationally concrete and acknowledges the
  maintenance cost of every legitimate evidence migration.

What it does not protect against is enumerated ((i)–(v)) and matches the
theory's own limits. I have no further requirement here.

### 4. Prior art — ADDRESSED, and the positioning is accurate

§1.3 gives the compact positioning; §6 now has a dedicated subsection on
hash-pinned fetching (Nix FODs [21], Bazel `download(sha256=…)` [22],
`go.sum` [23]) and one on metamorphic testing [24]. Both comparisons get the
essential distinctions right:

- Versus the pins: *referent* (freshly recomputed evidence for a proposition,
  with $p$ and $w$ bound jointly, vs. a stored artifact fetched once) and
  *direction* (pins buy hermeticity; witness binding buys intentional
  world-dependence, and a mismatch is signal rather than a supply-chain
  incident). The `go.sum` acknowledgement — "a `go.sum` mismatch *is* 'the
  world moved under a fixed name', but its names are version strings, not
  propositions" — is exactly the right concession, and it localises the
  novelty to enforcement inside a compiler's verdict on a claim.
- Versus metamorphic testing: same group, opposite direction — MT exploits
  $\mathrm{Inv}(p)$ where no oracle exists; witness binding binds against
  $\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$ where an oracle exists but is
  provably coarse. The composition remark (a metamorphic relation as a
  candidate witness) is correct and genuinely useful.

The earlier draft's silence here was a hole; the repair closes it without
overclaiming distance from neighbours.

### 5. Concurrency — NOT FIXED IN CODE; bounded in analysis. Acceptable, with a stated condition.

This is the one issue where the authors chose analysis over repair, and I
weighed it carefully.

What they shipped instead of a fix, §4.3: exit statuses are never read from
the capture (each build waits on its own gate subprocess), so exit-code
gating is immune; only token/witness *extraction* reads the shared path. A
clobbered capture is almost always fail-closed (`CLAIM_TOKEN_ABSENT` /
`CLAIM_TOKEN_MISMATCH`). A false *accept* requires the concurrent gate to
emit exactly this claim's declared token and witness — a 256-bit equality for
witness-bound claims; for token-only claims it effectively requires two
compiles of the same source, where a swapped capture answers the same
question correctly. The residual hazard (concurrent compiles of *different*
sources sharing token conventions) is named, the at-risk population is named
(this workspace's own parallel agents), and an operational mitigation
(`flock`, separate containers) is given. The designed fix — a fixed-path
`O_EXCL` lock file needing no string construction, sidestepping the
SIGSEGV-on-string-building constraint that killed the PID variant — is
recorded as deferred future work with an honest reason (compiler rebuild plus
behaviour-receipt re-certification).

I verified the premises I could: the executor does hardcode the shared
capture path, and the source comment records the failed PID variant,
corroborating the constraint story. The fail-closed analysis is logically
sound given those premises.

My judgment: for a research prototype making a conceptual contribution, a
precisely bounded known race with a designed fix and an operational
mitigation is acceptable — barely. What moves me from "reject until fixed" to
"accept" is that (a) the analysis degrades in the safe direction for every
realistic interleaving, (b) the false-accept window for the mechanism the
paper is actually about (witness-bound claims) requires a 256-bit collision
with the declared fingerprint, and (c) the paper undermines its own thesis in
no way — it treats the race as an instance of the phenomenon it studies and
discloses it in the limitations, not the fine print. **Condition (strongly
worded, not blocking):** the `O_EXCL` serialisation must land before any
deployment claim broader than this corpus, and the camera-ready should keep
the sentence "this workspace's parallel agents are the population at risk" —
it is the most honest sentence in the section.

## Remaining concerns (none blocking)

1. **Evaluation breadth.** The mechanism is bound to exactly one claim of
   ~295. The paper is disarmingly honest about this ("one claim of roughly
   295… the deployment question is §4.3, not a euphemism here"), and the
   theory (Prop 2.9, Cor 2.13) argues the class is general — but the evidence
   that the class *occurs* at scale is one measured family. The §5 corpus
   study (what fraction of claims admit non-trivial
   $\mathrm{Inv}(p) \setminus \mathrm{Stab}(w)$) is the right next
   measurement. For this paper's claims as stated, one exhibited instance
   plus a characterisation theorem suffices; a reader wanting a systems
   evaluation will not find one, and should not expect one from what this
   paper sets out to be.
2. **Measured premises carry real weight.** The case study's central fact —
   partition preservation under the flip — is measured at $n=5,6,7$ and
   reduced to, but not proven by, one equivariance lemma. The paper's
   measured/derived separation is exemplary, but a reviewer should note the
   theory in §§2.2–2.4 is elementary (the authors say so: "deliberately
   elementary") and the paper's weight rests on the mechanism plus the
   measured phenomenon. If the equivariance lemma were refuted tomorrow, the
   theory stands but the motivating case weakens to "exhibited at three
   levels, unexplained". The paper survives that; it is still worth saying.
3. **Minor presentational nit.** §2.3's "contains all of $\mathrm{Sym}(X)$"
   abuse (see issue 1) could gain one clause ("via the induced action on
   $S$"). Not required.

## Verification performed for this re-review

- Hand-checked the group orders in revised Prop 2.9 (setwise $N!(M-N)!$,
  pointwise $(M-N)!$, orbit $\binom{M}{N}$, ratio coherence).
- Hand-checked Theorem 2.12's strictness converse (the transposition of $S$
  swapping $s, s_0$ with equal $p$-values preserves $p$ pointwise and moves
  $w$ — correct).
- Confirmed the §3.4 fingerprint `705d0afdf8e83075…385757` matches the bound
  claim in `examples/epistemic/rupture_claims_verified.sio:188`.
- Confirmed the cited gates and companion gate exist:
  `scripts/ci/zd_fiber_spectra_witness_gate.sh`,
  `scripts/ci/witness_based_compilation_paper_gate.sh`.
- Confirmed the capture-path race premise in
  `self-hosted/compiler/claim_executor.sio:128-139` (fixed
  `/tmp/sounio_claim_gate_capture.out`; comment records the abandoned
  PID-based path).
- Did not re-run the gates or the paper gate myself; I take the 2026-07-28
  re-run reported in §3.4 and §9 at face value, with the companion gate
  existing as the re-measurable check.

## Recommendation

**Weak accept** (trending accept).

The first-round draft had a real mathematical error, a missing threat model,
a prior-art hole, and an unanalysed race. The corrected draft fixes the
mathematics correctly (not cosmetically), adds an adequate threat model,
positions against the exact prior art I named with accurate distinctions,
defends the $n=8$ scoping with an argument, and replaces the silent race with
a bounded analysis plus a designed fix. One issue (concurrency) is resolved
by analysis rather than repair; I accept that trade for a research prototype
given the fail-closed degradation, and I have stated the condition under
which it stops being acceptable. The paper's remaining weaknesses — single
corpus, measured (not proven) central phenomenon, elementary theory — are
disclosed in the text with a discipline most venues would benefit from, and
none of them contradict the claims as scoped.

**Confidence:** 4/5. I verified the revised mathematics by hand and the key
repository cross-references directly; I did not independently re-execute the
measured rungs, relying on the disclosed re-run and the companion gate's
existence.
