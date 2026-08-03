<!-- docs:meta
topic_id: repo.docs.reviews.witness-paper-reviewer-3-rereview
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.reviews.witness-paper-reviewer-3-rereview
-->

# Re-Review: "Witness-Based Compilation: When the Verdict Is Right but the Evidence Is Wrong"

**Reviewer:** 3 (Verification)
**Paper under review:** `docs/papers/witness_based_compilation_2026-07-28.md` (revised draft)
**Prior review:** `docs/reviews/witness_paper_reviewer_3.md` (Weak reject)
**Date of re-review:** 2026-07-28

## Evidence basis this round

Stronger than last round. This time I did not take the behavioural claims on the
paper's say-so:

- **Re-ran the companion paper gate** (`scripts/ci/witness_based_compilation_paper_gate.sh`):
  all five arms pass (`W1_TOKENS_BOUND`, `W2_WITNESS_PINNED`, `W3_FIGURES_PINNED`,
  `W4_HONESTY_MARKERS`, `W5_PEER_REVIEW_FIXES`), verdict
  `DRAFT_TOKEN_BOUND__WITNESS_PINNED__LIMITS_STATED`, rc 0.
- **Re-ran both production gates live** (with the repo venv on `PATH`; the ambient
  `python3` lacks numpy and the gates fail with an import error otherwise — a
  reproducibility wrinkle worth one line in the gate header, since the executor
  runs gates with an empty envp and the CI environment must be the one that has
  numpy). The real gate emits verdict `SPECTRA_COUNT_IS_3_TIMES_2_POW_N_MINUS_5`
  and witness `705d0afd…385757`, byte-identical to the manifest
  (`examples/epistemic/rupture_claims_verified.sio:188`) and to §3.4. The
  perturbed twin emits the **same token** and witness `e9f935cb…019424`,
  matching §3.4's quoted prefix. The paper's central empirical claim — the
  proposition preserved, the evidence replaced, only the witness recording it —
  is reproduced, not asserted.
- **Re-read the executor**: fixed capture path `/tmp/sounio_claim_gate_capture.out`
  (`claim_executor.sio:139`), parent-side `O_TRUNC` pre-create (`:337`), parent
  `wait4`s its own child pid (`:369,386`) — the load-bearing premise of §4.3's
  race analysis ("exit statuses are never read from the capture") is true in the
  source. `ce_extract_after` (`:185–206`) confirmed last-occurrence-wins; token
  and witness readers share it (`:210,227`) as §3.2 claims; witness comparison
  sequenced after the token decision via fresh variables (`:267–279`).
- **Read the rungs behind the two contested narratives**: R14 §2.1 and R15
  §§1.1–1.2 (see issue 2 below).

---

## Verdict on the five concerns from my first review

### 1. Proposition 2.9 (setwise vs. pointwise stabiliser) — **Fixed**

The revision is correct, and the correction is now load-bearing in the right
places. The stabiliser of set-valued evidence under the induced Sym(X) action is
the *setwise* stabiliser of the produced subset, order N!(M−N)!; the pointwise
stabiliser (order (M−N)!) is identified as a proper subgroup for N ≥ 2; the
indistinguishable-evidence count C(M,N)−1 and the gap factor
M! / (N!(M−N)!) = C(M,N) are mutually consistent. I re-checked the arithmetic;
it is right. The added paragraph explaining *why* the earlier draft shrank the
stabiliser by N! (conflating the witness value with the group elements acting
on its members) is exactly the kind of erratum-in-place a theory paper should
keep, and the closing remark — that the pointwise reading would be correct only
for *ordered* evidence — shows the distinction is understood, not patched.

One residual imprecision, non-blocking: Def. 2.6 defines Stab(w) as stabilising
the *function* w on all of S, while Prop. 2.9's characterisation quantifies at
the produced state ("maps the produced subset to itself as a set"). For
S = 2^X these coincide; in general the setwise stabiliser is the image of
Stab(w) restricted to the reference state's orbit. The sentence "the witness is
a single set value, and w ∘ σ = w asks that value to be preserved" makes the
intended reading clear enough that I will not hold the paper to it.

### 2. The n = 8 exclusion — **Adequately addressed, one self-contradiction left standing**

The three-point argument in §3.4 is a real argument, not an apology. Point one
(the exclusion is *inside* the proposition, so nothing at n = 8 can pass
silently) is the correct scoping move. Point three is the one that matters most
and is right: precomputing the n = 8 spectra and fingerprinting the cache would
change the evidence function from *computed at this build* to *read from a
file*, weakening exactly what the witness binds. Declining that trade and
saying so is the sound choice. Point two (the bound levels already bind the
same exhibited group element by the same mechanism) is supported by R15's table
(σ(H/2, H+H/2) count-preserving at n = 5, 6, 7, 8 with generic-flip controls
changing the count — I re-read it).

**However: Question 1 of my first review is not answered in the text.** §4.1
still narrates the discovery as "a perturbation that killed the contract's
verdict at levels 4–7 and survived at 8", while §2.4 states the flip is "an
exhibited element of Inv(p) \ Stab(w) at every level n = 5, 6, 7, 8". A reader
of the paper alone cannot reconcile these: if the flip preserves the count at
5, 6, 7, how did it *kill* verdicts there? I dug into the rungs; the
reconciliation exists and is interesting — R14's dying/surviving contracts were
the spectral-classifier contracts (a different proposition than the count law),
and R15 §1.2 explains the level structure: the contract checks levels jointly
and the tower is recursive, so a count-preserving flip aimed below the top
level "has a second chance to be caught" higher up; only at the boundary is
there nowhere higher to look. One or two sentences in §4.1 naming *which*
contract's verdict died at 4–7, and citing R15 §1.2's joint-level explanation,
would close this. As written, the paper's discovery narrative contradicts its
own §2.4 — in a revision whose §3.4 defence of the n = 8 exclusion leans
directly on §2.4's "every level" claim, so the tension is now load-bearing
rather than incidental.

### 3. Threat model — **Fixed**

§4.4 does what I asked and does it well. The principal adversary is named as
drift rather than a person; the active adversary is correctly bounded by the
observation that the gate is the TCB (whoever controls the gate controls exit
code, token, and emitted witness alike, and no mechanism at this layer can
prevent that); the strength against an evidence-moving-but-gate-respecting
attacker is stated as exactly collision resistance and no more. The
protects/does-not-protect lists are accurate against the implementation I read,
and the false-positive protocol (witness update as a review event, fingerprint
diff beside code diff) is the right operational answer to "what happens on
intended change". My Weakness 4 is resolved.

Two residuals, non-blocking:

- The TCB is named as "the gate", but the enforcement chain also includes the
  executor, the shell, the string-scraping extraction, the shared capture path,
  and — because Sounio is self-hosted — a previous incarnation of the very
  compiler being verified. Thompson's "Reflections on Trusting Trust" remains
  uncited and the bootstrap circularity unacknowledged (first-review Weakness 1
  stands in part). One sentence and one citation would do.
- My Weakness 3 — the gap between "the evidence the verdict was computed from"
  and "the object the gate fingerprints" (a gate that hashes a stale cache
  passes witness binding while the evidence moved) — is covered only obliquely
  under "a compromised or lying gate". The realistic case is the *honest but
  buggy* gate, not the lying one. The factoring p = π ∘ w is assumed, never
  checked; §2.5 should name this next to shared misinterpretation.

### 4. Prior art (metamorphic testing, Nix FODs, Bazel, go.sum) — **Fixed**

The new §6 engagements are substantive, not citation-stuffing. The hash-pinned
fetching paragraph gets both differences right and in the right order:
*referent* (the fingerprint is of evidence recomputed fresh at each compile,
for a proposition that must still hold — pins bind w alone, witness binding
binds p and w jointly) and *direction* (pins buy hermeticity; witness binding
buys deliberate world-dependence). The go.sum comparison is fair — "a go.sum
mismatch *is* 'the world moved under a fixed name', but its names are version
strings, not propositions" is precise. The metamorphic-testing paragraph
correctly identifies the shared group question and the opposite direction of
use (exploiting Inv(p) to manufacture tests vs. binding against
Inv(p) \ Stab(w) where an oracle exists but is insufficient), and the
"relations are found, not enumerated" connection to §2.5's open problem is a
genuine insight, not a courtesy citation.

Still missing from my first review's list: Blum & Kannan (program result
checking, 1995) under certifying algorithms — the most conspicuous absence for
a verification audience — and Pollack ("How to Believe a Machine-Checked
Proof", 1997) for the R0 scope limit and the checker-belief regress, plus
Thompson above. Non-blocking, but §6's credibility with this community depends
on exactly these names.

### 5. Concurrency (fixed capture path) — **Addressed, with one overclaim**

The §4.3 analysis is the right shape and its premises check out in source:
exit statuses come from `wait4` on the build's own child, so exit-code gating
is immune; only token/witness extraction reads the shared capture; a clobbered
capture lacks the token or carries the wrong one and the build is refused —
fail-closed in almost all interleavings. The false-accept window is correctly
narrowed to "the concurrent build's gate emits exactly this claim's declared
token and witness", and the 256-bit-equality argument for witness-bound claims
is sound. The mitigation (flock serialisation) and the designed fix (a
fixed-path `O_EXCL` lock, which sidesteps the string-building segfault that
killed the PID variant) are reasonable, and the deferral rationale (compiler
rebuild + receipt re-certification) is disclosed honestly.

The overclaim: for token-only claims the paper says two concurrent compiles of
the *same* source make "a swapped capture answer it correctly". That holds only
if the gate is deterministic. Under a nondeterministic gate — which §4.4 itself
cites as something witness binding exists to surface — compile A's gate can
exit 0 while emitting a wrong or missing witness, compile B's gate emits the
declared one, A reads B's capture, and A false-accepts: precisely the failure
class the mechanism exists to catch, let through by the race. The window is
narrow and flock closes it, but "answers it correctly" should be qualified with
"provided the gate is deterministic". Also unmentioned: the fixed path is
world-writable `/tmp`, so the clobbering writer need not be another compile —
any process on a shared machine can truncate it. Still fail-closed in nearly
all cases, but the analysis should say "concurrent compiles *or any other
writer*", since it costs one clause.

---

## Status of first-review items outside the five assigned concerns

- **Weakness 2 (witness binding is token binding at the refined proposition
  p_w(s) = [f(w(s)) = h]) — NOT addressed, and §7 now repeats the error.**
  On the paper's own Def. 2.3 a verifier is a predicate on S; witness binding
  is then token binding at evidence-identity-up-to-f. "The repair is not a
  finer proposition but a bound witness" is rhetorically clean and formally
  false. What is genuinely new — empirical claims come with a canonical
  refinement whose reference value a human cannot author unaided, so the check
  computes it — survives the concession intact. I do not understand why the
  authors protect this sentence; conceding the collapse in §2.4 would *strengthen*
  the paper for a verification audience, because it relocates the contribution
  from "a new grade of binding" (which reviewers will dispute) to "the right
  refinement, with the reference computed by the check" (which is defensible
  and still interesting). This remains my main theoretical complaint.
- **Last-occurrence-wins extraction** — confirmed in source; still disclosed
  but not hardened. Under §4.4's drift adversary this is now mostly scoped out
  (a spurious `_WITNESS` line from a dependency is drift and fails noisily in
  the common case), but "refuse on multiple distinct witness lines" remains a
  cheap hardening for a mechanism whose thesis is distrust of surfaces.
- **Near-miss claim fields** (`witnness` silently degrading to W4 opt-out) —
  unaddressed. An unknown-field warning is cheap; the mechanism is opt-in, so a
  typo is a silent opt-out.
- **Weakness 6 (necessity unestablished; one claim of ~295)** — unchanged, and
  the paper now says so itself plainly in §4.1 and §4.3. I accept the honest
  scoping; it caps the score rather than blocking.

---

## What changed my mind, and what did not

Changed: the math error is fixed and the fix is understood, not patched; the
threat model exists and is honest about strength; the prior-art gap is closed
with real engagement; the race is bounded with verified premises; and the
central experiment is now reproduced by my own hand, not trusted. The
companion gate passing on my machine — including its `W5_PEER_REVIEW_FIXES`
arm, which exists to stop the fixes being edited away — is a discipline I have
not seen another paper attempt, and it is directly responsive to how this
paper's claims could otherwise rot.

Not changed: the theoretical contribution is still mispriced (the §7 collapse
denial); the deployed witness still guards n = 5, 6, 7 while the anomaly lived
at n = 8 — the new argument makes this defensible rather than damning, but the
§4.1/§2.4 self-contradiction must go; necessity at corpus scale is still
unmeasured, which keeps this short of a clear accept for a main track.

## Overall recommendation: **Weak accept**

Up from weak reject. Four of five concerns are fixed; the fifth (n = 8) is
adequately argued with one residual contradiction. The remaining asks are
small and specific, and I would make them conditions for the camera-ready
rather than grounds for another round:

1. §4.1: one or two sentences resolving "killed the contract's verdict at
   levels 4–7" against §2.4's "at every level n = 5, 6, 7, 8" — name the
   spectral-classifier contracts and cite R15 §1.2's joint-level explanation.
2. §2.4/§7: concede that witness binding is token binding at the refined
   proposition p_w, and state the real residue (canonical refinement +
   check-computed reference). Delete or repair "not a finer proposition".
3. §6: cite Thompson (trusting trust), Blum & Kannan (result checking), and
   Pollack (believing machine-checked proofs).
4. §4.3: qualify the same-source race claim with gate determinism, and widen
   "concurrent compiles" to "any writer of the shared /tmp path".
5. Gate header: note the numpy interpreter requirement (gates run under an
   empty envp; the wrong `python3` fails with an import error, as it did for
   me).

## Confidence: **4 / 5**

Higher than last round on evidence: I re-ran the paper gate and both witness
gates and re-read the executor's race-relevant code paths. Still one notch off
because I did not rebuild the compiler itself (the W1–W4 probe table in §3.3 is
taken on the rung's receipt, not re-executed), and because the necessity
judgement still rests on how common coarse aggregate claims are across
empirical codebases — which neither the paper nor I measured.
