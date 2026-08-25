# Adversarial Review: Loom Journal Authority Quorum

Review the attached Sounio/OCaml implementation diff as a hostile scientific
and systems reviewer. Focus on whether the implementation and documentation
support this narrow claim:

> In configured quorum mode, Loom refuses pre-spawn continuity unless at least
> two pairwise-distinct configured journal principals provide valid signatures
> over the same checkpoint-bound event history.

Look specifically for BLOCKER or MAJOR findings in:

1. quorum bypasses, role collapse, duplicate-principal admission, or failure to
   bind signatures to the same event/checkpoint;
2. replay or generation-transition paths that can create a successor with only
   one valid journal share;
3. a treatment/control experiment that does not isolate the Sounio predicate
   `journal_quorum_is_satisfied`;
4. claims that overstate multi-signature authorization as threshold-signature
   cryptography, Byzantine consensus, semantic truth, organizational
   independence, or hardware-rooted custody;
5. compatibility regressions in the preserved legacy single-authority mode.

The intended boundary is explicit: this is 2-of-3 independent-key
authorization against one-key custody failure and offline one-key history
rewrite. It does not stop two structurally honest daemons from signing a
semantically false but monotonic event requested by the workload, and a member
that misses an event cannot automatically rejoin.

Return a severity-ordered review. If there is no BLOCKER or MAJOR, say so
explicitly and identify residual risks.

## Patch

The patch follows this prompt in the submitted review artifact.
