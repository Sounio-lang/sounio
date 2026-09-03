<!-- docs:meta
topic_id: repo.docs.reviews.witness-paper-reviewer-4-rereview
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.reviews.witness-paper-reviewer-4-rereview
-->

# Re-Review: "Witness-Based Compilation: When the Verdict Is Right but the Evidence Is Wrong"

**Reviewer:** 4 (Systems / software supply chain)
**Round:** 2 (re-review of corrected draft, `docs/papers/witness_based_compilation_2026-07-28.md`)
**Venue:** PLDI/OOPSLA/ICFP-style review
**Recommendation:** **Accept** (low end; see remaining concerns)
**Confidence:** 4/5 — I re-verified the load-bearing artefacts and re-executed both
gates in the repository this round.

---

## Verification performed for this re-review

Before scoring the revision I re-checked the artefact, not just the prose:

- The claim `zd_fiber_spectra_count_law_holds` in
  `examples/epistemic/rupture_claims_verified.sio:181-190` declares
  `witness = "705d0afdf8e830756f5d58eed9e6a11c7681d9e2e3a29ce7054ea67edc385757"`,
  matching §3.4 exactly.
- I **re-ran both gates** (`scripts/ci/zd_fiber_spectra_witness_gate.sh` and its
  perturbed twin, using the repo's `.venv` python — the ambient `python3` lacks
  numpy, see remaining concern R4). Real gate: exit 0, counts 3/6/12 at
  n=5,6,7, witness `705d0afdf8e83075…`. Perturbed twin: **exit 0, identical
  verdict token `SPECTRA_COUNT_IS_3_TIMES_2_POW_N_MINUS_5`, identical counts
  3/6/12, witness `e9f935cbab6f09fe…`** — exactly the discrimination §3.4
  claims. The central empirical result is live and independently reproducible.
- All cited rung documents (R0–R2, R14–R19) exist at the cited paths, and the
  companion gate `scripts/ci/witness_based_compilation_paper_gate.sh` exists.
- I cross-read R14 §2.1 and R15 §§1.1–1.2 against the paper's §4.1 and §2.4 to
  check a suspected inconsistency (see N1 below).

---

## Disposition of the five previous concerns

### 1. Proposition 2.9 (pointwise vs setwise stabiliser) — **FIXED**

The revised §2.3 now states the correct group: the stabiliser of a set-valued
witness under permutation-induced transformations is the **setwise** stabiliser
of the produced subset in Sym(X), order `N!(M−N)!`, with the pointwise
stabiliser (order `(M−N)!`) explicitly identified as a proper subgroup for
`N ≥ 2`. The orbit count `C(M,N) − 1` is now consistent with the group orders
(`M! / (N!(M−N)!) = C(M,N)` — the first draft's numbers were internally
incoherent; these are not). The paragraph candidly diagnoses the earlier error
(conflating the witness *value* with the group action on its members) and adds a
correct observation I had not asked for: relabelling *internal* to the produced
set moves nothing, not even the fingerprint, because the implementation hashes
the sorted enumeration. That last point closes a loophole a careful reader
would otherwise have raised against Theorem 2.11's applicability to set
evidence. The fix is mathematically right and better motivated than the
original. No residual issue.

### 2. The n=8 exclusion — **ADDRESSED; one framing reservation remains**

§3.4 now gives an argument rather than an apology, as demanded. Of its three
points, two carry real weight: (a) the exclusion is *inside the proposition* —
the claim asserts the count law at n=5,6,7 and nothing beyond it, so no n=8
behaviour can pass silently under this claim; and (b) the precompute-cache
alternative would quietly change the evidence function from *computed at this
build* to *read from a file*, weakening exactly what the witness binds. Point
(b) is the strongest systems argument in the revision and I accept it
fully — it converts my objection into a demonstration that the authors
understand the mechanism's own theory. Point (c) ("the bound levels already
bind the same group element") is true per R15's table (σ(H/2, H+H/2) preserves
the count at n=5,6,7,8 with generic-flip controls) but is the weakest leg, and
the paper knows it: §4.1 itself argues the blind spot "sits at n=8 precisely
because that is the only level with nowhere higher to look", i.e. n=8 is the
scientifically interesting level, and the production claim binds the levels
where the original contract already enjoyed cross-level protection.

**Residual (minor):** the abstract and Contribution 4 still lead with the n=8
story (24 spectra, the anomaly's level) and say "we apply it to the motivating
case"; only §3.4 discloses that the bound claim covers n=5,6,7. One parenthesis
in the abstract — "(bound at n=5,6,7; the n=8 arm exceeds the gate budget,
§3.4)" — would close this. Shepherding item, not a rejection ground: the
disclosure is in the paper, prominently argued, and honest.

### 3. Threat model — **FIXED; this is the best part of the revision**

The new §4.4 is a real threat model, not boilerplate. It names the principal
adversary (drift, not a person — correct for this mechanism), places the gate
squarely in the TCB ("whoever controls the gate controls the exit code, the
verdict token, and the emitted witness alike… no mechanism at this layer can
prevent that"), and bounds the active-adversary strength to exactly collision
resistance ("exactly as strong as SHA-256, and we claim no more"). The
protects/does-not-protect lists are exhaustive and cross-referenced. Two of my
first-round concerns are substantially answered here: the trust-model
distinction (the witness channel has no authenticity property; the gate is
trusted) is now stated, and the rotation/lifecycle question gets the
false-positive protocol — a witness update is a review event, recorded in the
same commit as the change that moved the evidence, "of the same gravity as
editing the claim itself". The protocol honestly admits its residual hole
(joint update of claim and witness in one commit is unstoppable; version
control makes it visible). That is the right answer at this layer; lockfiles
solved it the same way.

The nondeterminism point (a nondeterministic gate mismatches against itself on
recompilation, surfacing as a build failure) is a genuinely nice consequence I
had not appreciated in round 1.

### 4. Prior-art engagement (metamorphic testing, Nix FODs, Bazel `download(sha256)`, go.sum) — **FIXED**

The added §1.3 positioning and the three new §6 paragraphs are substantive, not
name-dropping:

- **Hash-pinned fetching (Nix FODs [21], Bazel [22], go.sum [23]).** The
  two-axis distinction is correct and precise. *Referent*: the fingerprint is of
  evidence recomputed by re-running the check at each compile, not of an
  artifact fetched once — and witness binding binds p and w jointly where the
  pins bind w alone ("a `go.sum` entry says nothing about what the module
  *does*"). *Direction*: pins serve hermeticity (a mismatch is a supply-chain
  incident); witness binding serves intentional world-dependence (a mismatch is
  the intended signal). The `go.sum` clause — "its names are version strings,
  not propositions" — is exactly the localisation I asked for. With this in
  place, the "first claim in any compiler whose build is conditioned on a
  fingerprint of its evidence" phrasing is defensible under its hedge.
- **Metamorphic testing [24].** Correctly identified as the same group question
  read in the opposite direction: MT *exploits* Inv(p) to manufacture tests
  where no oracle exists; witness binding treats Inv(p) ∖ Stab(w) as the region
  where an existing oracle is provably insufficient. The compositional remark
  (a metamorphic relation a gate is expected to satisfy is a candidate witness)
  and the honest parallel (both fields find relations rather than enumerate the
  group) are well judged.

**Residual (minor):** in-toto [11] still gets one clause. A witness-bound claim
is expressible as an in-toto link with product constraints; saying so would
strengthen, not weaken, the novelty claim. Rebuilders / non-reproducible-build
literature is still absent. Not blocking — the mandated four are done well.

### 5. Concurrency safety — **ADDRESSED with a bounded analysis; fix designed but not shipped**

The §4.3 bullet is now a precise hazard analysis rather than an admission. The
load-bearing facts: exit statuses are never read from the capture (each build
waits on its own gate subprocess), so exit-code gating is immune; only token and
witness *extraction* reads the capture. A clobbered capture is fail-closed in
almost all cases (`CLAIM_TOKEN_ABSENT` / `CLAIM_TOKEN_MISMATCH`). A false
*accept* requires a concurrent build's gate to emit exactly this claim's
declared token and witness — a 256-bit equality for witness-bound claims — or,
for token-only claims, two compiles sharing gate-output conventions, which in
practice means the same source answering the same question. The residual window
(different sources, coinciding token conventions, this workspace's parallel
agents) is named and the mitigation (flock, separate containers) is operational
today. The designed fix — a fixed-path lock file taken with `O_EXCL`, avoiding
the runtime-string-construction path that SIGSEGVs the self-hosted compiler —
is recorded with its deferral rationale (compiler rebuild plus R17 behaviour-
receipt re-certification).

This is a fair, quantified response and I accept the analysis. Two harsh notes
stand. First, the deferral rationale — "the window above did not justify that
cost on the day" — sits oddly in a paper whose thesis is receipts over
intentions; the verification path of the reference compiler remains
demonstrably racy by design. I do not demand the fix for acceptance (the
analysis bounds the hazard and the mechanism is opt-in), but the O_EXCL lock is
cheap relative to the paper's claims and its absence will be the first thing a
systems PC member cites. Second, my round-1 point about interaction with build-
system semantics — remote caching, CAS-addressed actions, rebuilders that
*should* disagree on different days — remains untouched anywhere in the paper;
§5's "empirical lockfiles" gestures at it without naming the problem.

---

## New issues found in the corrected draft

**N1 (minor, clarity).** §4.1's compressed parenthetical — "rung R14: a
perturbation that killed the contract's verdict at levels 4–7 and survived at
8" — appears to contradict §2.4's claim that σ(H/2, H+H/2) is an exhibited
element of Inv(p) ∖ Stab(w) "at every level n = 5, 6, 7, 8". I checked the rung
documents: there is no actual inconsistency. R15 §1.2 explains that a flip
*aimed* at level k also perturbs every deeper level computed through it (the
wrapper intercepts recursive calls), and the R14 contract checked n=6,7,8
together — so a count-preserving flip aimed low betrays itself higher up, and
only the level-8 aim survives. But this reconciliation lives entirely in R15
§1.2; a reader of the paper alone will see a contradiction between two of its
own claims. One sentence in §4.1 carrying the recursion mechanism would fix it.

**N2 (nit).** Proposition 2.9 is now correct but still stated as prose. Given
that its first-draft error was precisely a formalisation error, a displayed
formal statement (Stab(w) ∩ Sym(X)-induced maps = Sym(w(s)) × Sym(X ∖ w(s)))
would inoculate against regression. Optional.

---

## Remaining concerns carried over from round 1 (not among the mandated five)

- **R1 — The witness still dies at the compiler boundary.** The emitted ELF
  carries nothing; no SBOM annotation, no in-toto predicate, no build manifest.
  §5's future directions do not list it. For a paper about binding artifacts to
  evidence, producing artifacts that carry no evidence binding remains the
  largest systems gap. Camera-ready should at least name it as future work.
- **R2 — Canonicalisation is still hand-waved.** "SHA-256 over the spectra,
  sorted" pins one source of nondeterminism; serialisation format, float
  precision, and platform/library drift in the gate's own toolchain are
  unaddressed. Witness schemas (§5) are the planned relief, but cross-machine
  stability is a *systems* property of the mechanism as deployed, not a
  semantic nicety.
- **R3 — The discriminating baseline experiment is still not run.** Nothing
  exhibits a case where input-pinning or output-freezing gives the wrong answer
  and witness binding gives the right one; a sceptical reader can still
  approximate the demonstrated protection with `sha256sum` in CI plus a test
  runner. The compiler-enforcement difference (no ELF emitted) is genuine; the
  paper asserts rather than shows that it matters.
- **R4 — Artefact portability nit (new this round, from verification).** The
  gates require the repo's `.venv` python (numpy); under the ambient `python3`
  the gate crashes with `ModuleNotFoundError`. With `set -euo pipefail` this
  fails closed (nonzero exit), which is correct behaviour — but it is a live
  instance of R2: the witness's stability depends on a toolchain the paper does
  not pin or discuss.

None of R1–R4 blocks acceptance; all belong in a camera-ready or the
empirical-lockfile follow-up.

---

## Overall recommendation: **Accept**

All five mandated concerns are addressed — three fully (Prop 2.9, threat model,
prior art), two with residual notes that do not rise to rejection grounds (n=8
framing; concurrency fix designed but unshipped). The revision is substantive,
not cosmetic: §4.4 is a model of honest scoping, the setwise/pointwise repair
is mathematically correct and better argued than the original, and the
hash-pinning comparison finally localises the novelty precisely. I independently
re-executed the central experiment this round — the real gate and its perturbed
twin emit identical verdict tokens and counts with different witnesses, exactly
as claimed — so the paper's load-bearing empirical claim is not archival
assertion but re-measurable fact.

What keeps this at the low end of accept rather than higher is unchanged in
kind, though diminished in degree: the theory is elementary (the authors say
so), the deployment is one claim of ~295, and the systems content still stops
short of artifact-carried attestation and build-system integration. As a
theory-plus-mechanism paper with a genuinely unusual, reproducible empirical
hook and exemplary self-discipline (the paper gate), it clears the bar.

**Shepherding items (camera-ready):** (i) abstract parenthesis disclosing the
n=5,6,7 scope of the bound claim; (ii) one sentence in §4.1 carrying R15's
recursion mechanism to defuse the apparent contradiction with §2.4 (N1);
(iii) name the artifact-carried-witness direction (SBOM / in-toto predicate) in
§5 (R1); (iv) one clause positioning witness-bound claims against in-toto link
semantics.

**Confidence:** 4/5. I verified the revision's textual claims against the
repository and re-ran the load-bearing gates myself; I did not rebuild the
compiler or re-execute the W1–W4 probes against `claim_executor.sio`, and I
continue to take the Cayley–Dickson mathematics on the cited rung evidence.
