<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r26-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r26-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R26 — the never-committed oracle, reconstructed; the orbit theorem's verifier runs in-tree for the first time

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ORACLE_RECONSTRUCTED__ORBIT_VERIFIER_RUNS_IN_TREE`
**Parents:** `self_falsifying_compilation_line_r20_2026-07-28.md` (found the dangling dependency), `self_falsifying_compilation_line_r21_2026-07-28.md` (proved a theorem resting on the one this oracle's absence kept from running)
**Harness:** `scripts/research/self_falsifying_compilation_line_r26_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r26_gate.sh`

---

## 1. Result

R20 found `cd_tower_automorphism_oracle.py` absent from every branch in the
repository's history (`git log --all` empty), yet loaded by the orbit theorem's
verifier via `exec_module` with no fallback. R21 then proved a theorem that
**rests on** that orbit theorem — whose own verification script could not execute
in any checkout, because its oracle had never existed anywhere.

> **The oracle is reconstructed from the verifier's own proof, validated rather
> than assumed, and the orbit theorem's verifier now runs end to end for the
> first time in any checkout of this repository — reproducing the predicted
> orbit structure at n = 4, 5, 6, 7.**

Verdict: `SELF_FALSIFYING_R26_VERDICT ORACLE_RECONSTRUCTED__ORBIT_VERIFIER_RUNS_IN_TREE`.

```
n=4: orbits {7: 1}          stab 24  fixed []                   OK
n=5: orbits {1: 1, 7: 2}    stab 24  fixed [8]                  OK
n=6: orbits {1: 3, 7: 4}    stab 24  fixed [8, 16, 24]          OK
n=7: orbits {1: 7, 7: 8}    stab 24  fixed [8,16,24,32,40,48,56] OK
```

## 2. The reconstruction, and the subtlety that validates it

The oracle exposes two functions the verifier consumes:

- `sweep_autos(n)` — the valid index-maps at level n as image arrays of length
  2ⁿ. At n = 4 these are the block-form maps `[[g, 0], [0, 1]]`, g ∈ GL(3,2) on
  the octonion bits {0,1,2}, seam bit 3 fixed.
- `orbits_on(M, elems)` — the partition of `elems` under the group given
  explicitly by the image arrays M.

Built from the verifier's own PROOF (block-form freezing + GL(3,2)-transitivity),
then **validated, because a reconstruction assumed is worth nothing**:

> The verifier's docstring says "168 valid maps = GL(3,2) = Aut(octonions)". That
> is imprecise for the **signed** table: only **21** of the 168 GL(3,2) linear
> maps preserve `cds` with a fixed sign convention. The correct fact — checked in
> `_self_check` and clause W1 — is that all 168 are valid **permutation parts**:
> for each g there **exists** a sign vector ε making `e_i ↦ ε_i e_{g·i}` an
> algebra automorphism, i.e. the discrepancy cocycle
> `δ(i,j) = cds(g·i, g·j) · cds(i,j)` is a coboundary `ε_i ε_j ε_{i⊕j}`. Solving
> that F₂ system succeeds for all 168 and fails for none. Orbits on fibers depend
> only on the permutation part (ε does not move the label L = lo ⊕ hi), which is
> exactly why the orbit theorem is stated for the permutation part — and why this
> is the memory's "our 168 is the permutation part; the signed group grows".

So the reconstruction is not a guess dressed as GL(3,2): it is the 168
permutation parts, each independently certified to admit a sign completion, and
they generate precisely the orbit multiset the theorem predicts.

## 3. What this closes, and what it does not

- **Closes the deepest dangling dependency of the arc.** R20's audit went from 6
  absent hard-dependencies to 2, and **neither of the remaining two is this
  line's** — both are the foreign fMRI lane (`extract_fmriprep_roi_timeseries.py`,
  `extract_trained_roi_associator.sio`). The orbit theorem R21 rests on is now
  verifiable in every checkout.
- **Not a new proof of the orbit theorem.** The theorem was already proven ∀n in
  the verifier's docstring (block-form freezing, unconditional per Kirshtein 2012
  Thm 41, + GL(3,2) transitivity). This makes its *machine verification* runnable;
  it does not re-derive the mathematics.
- **Not recovery.** The file never existed in git; this is reconstruction from
  the proof, not `git show` from a branch. Independence is deliberate: the oracle
  writes `cds` from the recursion rather than importing it, because an imported
  dependency is the failure that made the reconstruction necessary.
- **Not a claim about the foreign lanes.** Their two absent artifacts are
  recorded, not fixed here.

## 4. Reproduce

```bash
python3 scripts/research/cd_tower_automorphism_oracle.py       # self-check
python3 scripts/research/cd_tower_auto_action_on_zd_fibers.py  # the verifier, now runnable
python3 scripts/research/self_falsifying_compilation_line_r26_contract.py
bash scripts/ci/self_falsifying_compilation_line_r26_gate.sh
```

## 5. AI disclosure

Oracle, contract, gate and spec drafted under human direction (2026-07-31). The
oracle was reconstructed from the verifier's proof and its 168-vs-21 distinction
was derived by hand and then checked mechanically. The orbit-structure figures
are produced by the verifier itself, now that it can run. No clinical content.
GAIDeT-ICMJE 2025.
