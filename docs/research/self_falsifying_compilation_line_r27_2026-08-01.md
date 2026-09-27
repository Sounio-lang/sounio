<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r27-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r27-2026-08-01
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R27 — declared alive, never checked

**Date:** 2026-08-01
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `CLAIM_LIVENESS_DEFINED__DECLARED_ALIVE_IS_UNCHECKED__1_OF_16_BOUND`
**Parents:** `self_falsifying_compilation_line_r22_2026-07-29.md` (a field shaped like a measurement that was a literal), `self_falsifying_compilation_line_r1_2026-07-26.md` (the module-closure wall), `self_falsifying_compilation_line_r2_2026-07-26.md` (token binding), `self_falsifying_compilation_line_r17_2026-07-28.md` (witness binding)
**Harness:** `scripts/research/self_falsifying_compilation_line_r27_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r27_gate.sh`

---

## 1. Result

R22 found a field shaped like a measurement that was a literal, enforced by a
gate. R23 found `validated_by` was path ownership. R25 found research authority
was a path default. Each of those lived in document front-matter — the
governance surface, not the compiler.

This rung finds the same shape inside the compiler's own claim ontology.

> **Every claim in the production manifest declares `verdict = Verdict::Alive`.
> The executor never checks it. Its only read of the `verdict` field scans the
> slice for the substring "archived"; the token `Alive` does not occur in the
> executor at all. Aliveness is asserted by every claim and tested by none. One
> claim of sixteen binds anything beyond an exit code, and an entire compiler
> lane emits ELFs without consulting the verifier.**

Verdict: `SELF_FALSIFYING_R27_VERDICT CLAIM_LIVENESS_DEFINED__DECLARED_ALIVE_IS_UNCHECKED__1_OF_16_BOUND`

**No counts in the headline, and R1 is the reason.** R1's headline carried a
measured count and went stale the moment a claim was added to the manifest it
counted. The figures below are corpus figures; they live in §3 with the date
they were measured, and the contract re-measures them on every run.

## 2. Where it comes from

```
self-hosted/compiler/claim_executor.sio:452   if ce_name_eq_str(f.name, "verdict")
self-hosted/compiler/claim_executor.sio:455       if ce_slice_is_archived(src, vs, ve)
```

That is the whole of it. The executor decodes the `verdict` field, asks whether
the slice mentions "archived", and discards it. There is no comparison against
`Alive`, no comparison against anything else, and no other read of the field.

A claim declaring itself alive therefore costs the compiler nothing and buys the
reader nothing. The word is load-bearing in the prose and inert in the machine.

## 3. Verified, and how

Corpus figures measured 2026-08-01 at `03b7caf1a`; the contract re-measures them.

| clause | | |
|---|---|---|
| `A1_ALIVE_IS_UNCHECKED` | 16 claims, 16 declare `Verdict::Alive`; one read of `verdict` (:452), one use of it (:455, the archived scan); `Alive` occurs 0 times in executor code | the finding |
| `A2_PROMISE_SCOPE_IS_NARROWER_THAN_THE_MECHANISM` | `main.sio` calls the verifier at 3 sites; `lean_single.sio` emits ELFs from 3 functions and calls the verifier **0** times | a whole lane outside the promise |
| `A3_BINDINGS_ARE_RARE` | 1 of 16 claims declares a binding beyond the exit code — `zd_fiber_spectra_count_law_holds`, with all three (`verdict_token`, `witness`, `provenance`). The other 15 are EXIT_ONLY | per R2/R15, exit-code gating is not content checking |
| `A4_ANCHORING_CHANGES_THE_CENSUS` | naive substring counts total 12; comment-stripped and field-anchored, 3 | the instrument check |

**A4 exists because the naive census is wrong in the flattering direction.** A
bare `grep -c witness` over the manifest returns 8, and `provenance` returns 3 —
because the manifest *discusses* witnesses and provenance in comment blocks. An
unanchored count would have reported far more binding than exists and turned
this rung's finding into its opposite. The clause measures both ways and fails
if they ever agree, because agreement would mean the anchoring stopped working.

**Cost: ~0.4 s, and no build.** This rung invokes no compiler and takes no
global build lock. Every clause is a static census of checked-in source.

## 4. What the ELF promises, stated so it can be falsified

The mechanism verifies claims **in the source file it was given**, on the lanes
that call the verifier, to the depth each claim declares. Written out:

> If `souc` emits an ELF from source `T` with `--verify-claims`, then for every
> claim in `T`'s own registry the declared gate ran to completion and its exit
> code was zero — and, for the claims that declare one, the verdict token,
> witness or provenance matched. It promises nothing about claims in `T`'s
> imports (R1), nothing about a lane that never calls the verifier (A2 names
> `lean_single`), nothing about whether a claim declaring itself `Alive` is
> alive (A1), and nothing about whether the gate's evidence is well-founded
> (R0).

Three of those four exclusions were already known separately. Writing them into
one sentence is what makes the promise checkable prose rather than an
impression, and it is what the referees' "no semantics, no reachability"
objection was asking for.

## 5. What this is NOT

- **Not a claim that any claim is false.** Nothing here shows a single claim in
  the manifest is untrue. The finding is that the compiler does not test the
  field that says so — the gap is between *asserted* and *checked*, not between
  *asserted* and *true*.
- **Not a refutability test.** Establishing that a claim would go red under a
  perturbation of its input needs compiler runs with a null control (R13 §5.1),
  and this rung budgets none. `EXIT_ONLY` here means "declares no binding", not
  "proven inert".
- **Not a compiler change.** No `.sio` file is touched. Making `Alive` mean
  something — or removing it — is a decision about the claim ontology, and
  belongs to a rung that argues for one.
- **Not a statement about `lean_single`'s correctness.** A2 says only that the
  lane emits without verifying, which makes it out of scope for the promise. It
  is not a defect in that lane; it is a boundary of this one.

## 6. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r27_contract.py
bash scripts/ci/self_falsifying_compilation_line_r27_gate.sh
```

Needs only Python and the checked-in sources. Leaves the working tree unchanged.

## 7. AI disclosure

Finding, contract, gate and spec drafted under human direction (2026-08-01). The
rung was selected by a multi-agent planning workflow whose designs were then run
past adversarial refuters; every number in §3 was independently re-measured by
hand before it was written here, and the refuters' corrections to the census
method (comment-stripping, field anchoring, excluding the function definition
from the call-site count) are why A4 exists. No clinical content.
GAIDeT-ICMJE 2025.
