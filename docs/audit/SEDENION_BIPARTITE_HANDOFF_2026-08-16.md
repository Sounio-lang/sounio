<!-- docs:meta
topic_id: repo.docs.audit.sedenion-bipartite-handoff-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-bipartite-handoff-2026-08-16
-->

# Handoff — `formal/lean4/SounioSedenionBipartite.lean`

- **Date:** 2026-08-16
- **Agent / lane:** `cursor-3` / `sorry-sedenion-bipartite`
- **Worktree:** `/workspace/.wt/cursor-3`
- **Branch:** `lane/cursor-3/formal-sedenion-core-patterns-20260817` (cherry-picked off
  clean `origin/main` at `c66014fda9`; the work was authored on the stale checkpoint
  branch `lane/cursor-3/20260814`, which is 627 commits ahead of main and not a
  landing candidate)
- **Base commit:** `965b2d3226` (file byte-identical to `origin/main` at start)
- **Toolchain:** Lean 4.33.0 (`d8b18978322de05a8f3dba51ef03cf5461676c17`), no Mathlib, no Lake

## Result

**All four original `sorry` are closed. The file is `sorry`-free.** It now meets
the repository's Mathlib-free, no-`sorry` contract, with no `native_decide`.

This halves the repo's real `sorry` debt. The founder's audit found eight real
`sorry` tactics across 137 Lean files, four of them here. Re-measuring now
(comments and docstrings stripped) leaves **four**, none in this file:

```
EpistemicEffects.lean: 1
SounioDeGreyChi5Real.lean: 1
SounioGradedModal.lean: 2
TOTAL real sorry tactics: 4
```

> **Re-measured 2026-08-27 against `origin/main` (146 Lean files in `formal/lean4`):
> `TOTAL real sorry tactics: 0`.** All four listed above were closed on `main` after
> this handoff was written — `SounioGradedModal.lean`'s two by #1772 (`1bb2db46fc`).
> The block above is the 2026-08-16 measurement and is kept as the record of what was
> true then; it is not the current state.

## 0. Reconciling the two `sorry` readings — they never conflicted

The founder measured four `sorry` at lines **48, 65, 74, 81**. This agent at
one point reported warnings at **583** and **599**. Both were correct; they
described **different states of the same file**.

```
$ git show HEAD:formal/lean4/SounioSedenionBipartite.lean | grep -n sorry
48:noncomputable def twistedNormSq (d : SedVec) (p : ZDPrim) : ℤ := sorry
65:  sorry
74:  sorry
81:  sorry

$ git show origin/main:... | sha256sum  ==  git show HEAD:... | sha256sum  → IDENTICAL
```

`HEAD` and `origin/main` were byte-identical for this file, so the founder's
measurement and this agent's starting point were the same bytes. 583/599 was
the working tree mid-edit. Two details worth recording so nobody re-derives
them: Lean reports the **declaration header** line rather than the `sorry`
token, and the line numbers moved repeatedly as header notes were added. All of
it is moot now — there are no `sorry` warnings left to number.

## 1. What `twistedNormSq` was settled on, and why

Line 48 was an **unimplemented definition**, not an unproved lemma, so the
three lemmas under it rested on something with no computational content.

```lean
def twCoeff (d : SedVec) (p : ZDPrim) (j : Nat) : Int :=
  d (j ^^^ p.lo) * sedSigma (j ^^^ p.lo) p.lo
    + primSign p * (d (j ^^^ p.hi) * sedSigma (j ^^^ p.hi) p.hi)

def twistedNormSq (d : SedVec) (p : ZDPrim) : Int :=
  (List.range 16).foldl (fun acc j => acc + twCoeff d p j * twCoeff d p j) 0
```

Justification, in order of authority:

1. **It is the right Cayley–Dickson product.** For a basis element,
   `(x · e_k)_j = x_{j⊕k} · σ(j⊕k, k)`. A primitive surgery element is
   `p = e_lo ± e_hi`, contributing exactly two such terms, the second carrying
   `primSign p`. `twistedNormSq` is `Σ_j (d·p)_j²`.
2. **It matches the sibling file already in the repo** — `twCoeff` / `twNormSq`
   in `SounioErdosUnitDistance.lean` have the same shape. Deliberately the same
   definition, not a new one.
3. **It matches Appendix C** of `papers/sedenion-chromatic-gap/paper.md`.
4. `σ` is `cdSigma … 4`, the same recursion as `SounioSeamFlip.cdSigma`,
   `SounioCayleyDickson.cdSigma`, and the compiler's `cd_sigma_ct`.

`noncomputable` was dropped and `ℤ` replaced by `Int` to stay Mathlib-free.

## 2. Status of the four original `sorry`s

| Original line | Declaration | Status |
|---|---|---|
| 48 | `twistedNormSq` | **CLOSED** — implemented (§1) |
| 65 | Thm 3.1 `k_odd_no_odd_cycle` | **CLOSED** — proved |
| 74 | Thm 3.2 `k_even_no_edges` | **CLOSED** — proved |
| 81 | `sedenion_zd_surgery_bipartite` | **CLOSED** — proved |

### Theorem 3.1 — odd K forbids an odd closed chain

Double-counting parity. Per coordinate, a vanishing sum of `0/±1` values has an
even number of nonzero terms (`unit_sum_zero_hits_even`); summing over the 16
coordinates preserves evenness; the same total counted the other way is `n·K`;
`n` and `K` both odd make it odd. `omega` closes it.

About 25 supporting list/arithmetic lemmas had to be built from scratch
(`iSum`, `nSum`, `foldl_add_split`, `filter_length_nSum`, …) for want of Mathlib.

### Theorem 3.2 — even K admits no edge

```
‖d·p‖² = 2K + 2·cross          so an edge forces  K + cross = 1
cross ≡ coinCount  (mod 2)     each term is 0 or ±1, nonzero iff coincidence
coinCount is even              fixed-point-free involution  j ↦ j⊕(lo⊕hi)
```

`t = lo ⊕ hi` is nonzero because `lo ≠ hi`, and `j ↦ j⊕t` swaps precisely the
two coordinates the coincidence predicate reads, so it preserves that
predicate. Sorting the coincidence set by `j < j⊕t` versus `j⊕t < j` splits it
into two halves that `nSum_reindex` shows are equinumerous, giving `2·N₀`.

Sorting by the comparison rather than by a bit position avoids needing any
lowest-set-bit machinery. Core Lean's `Nat.xor_self` / `xor_assoc` / `xor_comm`
suffice, via the derived `xor_undo` and `xor_mid`.

### Main theorem — no odd cycle

Three parts: `k_even_no_edges` forces every edge's `K` odd;
`mixed_odd_no_odd_cycle` handles heterogeneous odd `K_i`; and
`closed_walk_coordSum_zero` supplies the telescoping.

The mixed-K case needed **no new algebra**. Theorem 3.1's double count never
uses anything about the `K_i` beyond `Σ_i K_i` being odd, so restating it with
a `Ks : Nat → Nat` was enough. This was flagged by the Grok review, which
correctly said the docstring had overstated the remaining gap.

`Classical.choose` extracts the per-edge `K` from the hypothesis's existential,
after totalising it (`∀ i, ∃ K, i < n → isKDiff K (ds i)`) so that a plain
`if … then … else` on the decidable `i < n` suffices. That is the only source
of `Classical.choice` in the assembly.

## 3. Reproducing this exactly

```bash
export ELAN_HOME=/workspace/.tmp/elan
export PATH="$ELAN_HOME/bin:$PATH"
cd /workspace/.wt/cursor-3/formal/lean4
lean --version        # Lean (version 4.33.0, ... d8b18978322de05a8f3dba51ef03cf5461676c17)
lean SounioSedenionBipartite.lean; echo "exit=$?"
```

Observed, ~1.6 s wall: `exit=0`, no errors, **no `declaration uses sorry`
warning**. One cosmetic linter note remains, that the statement's `hn` binder
name is not referenced inside the statement — it is in the original signature
and is used in the proof.

Do not use `elan default stable`. `stable` is a moving ref and has drifted
under this project once (pinned by `7cd35ba73c`, moved by `220fe6c388`). The
pin is `leanprover/lean4:v4.33.0`. `/workspace/sounio/formal/lean4/lean-toolchain`
still reads `stable` on its stale research branch; `origin/main` carries
`v4.33.0` and is the one to trust.

### Axiom audit

Append `#print axioms …` before `end Sounio.SedenionBipartite` and re-run. All
ten headline results — `sedenion_zd_surgery_bipartite`, `k_odd_no_odd_cycle`,
`k_even_no_edges`, `mixed_odd_no_odd_cycle`, `closed_walk_coordSum_zero`,
`telescope`, `coinCount_even`, `cross_mod_two`, `twistedNormSq_expand`,
`perm_range_xor` — return only:

```
[propext, Classical.choice, Quot.sound]
```

**No `sorryAx`. No `native_decide` axioms.** Several results
(`k_odd_no_odd_cycle`, `mixed_odd_no_odd_cycle`, `telescope`,
`closed_walk_coordSum_zero`, `twistedNormSq_expand`) do not even need
`Classical.choice`.

## 4. What did NOT work — read this before extending the file

**(a) The `whnf` timeout, and how it was beaten.** This was the central
obstruction and it is worth understanding rather than rediscovering. The first
attempt at `sumA_sq_eq_K` tried to transport `perm_range_xor` across a sum by
rewriting *inside* a concrete `List.range 16`. The elaborator unfolded the
range into a 16-element literal and died on a deterministic `whnf` timeout.

The fix is to never let the list be destructed. `iSum_perm` proves permutation
invariance by induction on the **`List.Perm` derivation** — its four
constructors `nil`/`cons`/`swap`/`trans` — so the proof only ever sees
`iSum_cons` and never the range's contents. `iSum_map` and `iSum_reindex` then
transport along `perm_range_xor` with the list opaque throughout. The whole
file elaborates in about 1.6 s.

**Generalise this if you hit the same wall elsewhere:** prove the structural
lemma about an arbitrary list, then instantiate at `List.range 16`. Rewriting
under a concrete range is what explodes.

**(b) `native_decide` on `perm_range_xor` — used, then removed.** The first
working `perm_range_xor` discharged its 16 cases with `native_decide`. It
compiled, but `#print axioms` exposed 16
`perm_range_xor._native.native_decide.ax_1_*` axioms plus `Classical.choice`.
Each case is a *closed* instance, so this was not the forbidden move of
`native_decide`-ing a universally quantified theorem, but it still put the
compiler in the trusted base for a decision the kernel can make. Plain `decide`
handles each 16-element `List.Perm` instance in well under a second. **Try
`decide` first for finite case sweeps here.**

**(c) Mathlib tactics that do not exist in core.** `set` and `by_contra` are
Mathlib, not core, and both failed as `unknown tactic`. Replace `set` with
top-level `def`s (that is why `coinLo` / `coinHi` exist as definitions rather
than local bindings) and `by_contra` with an explicit `omega`-generated case
split. Similarly there is no `push_cast`; `omega` handles `Nat → Int` casts
directly, which is what `iSum_ofNat` relies on.

**(d) `dite` needs a `Decidable` instance.** Choosing the per-edge `K` with
`if h : ∃ K, …` fails to synthesise. Totalise the existential first, then
branch on the decidable `i < n`.

**(e) `simp` is not enough for the list layer.** `iSum_range_succ`,
`nplus_nminus_eq_nz`, `iSum_eq_plus_minus`, `double_count_hits` and
`nSum_supp_const` all resisted `simp` and needed explicit `calc` / `rw` /
induction, plus two accumulator lemmas (`foldl_add_init`, `foldl_add_init_nat`)
to move the `foldl` seed out of the way. Without Mathlib the `foldl` seed is the
recurring obstacle.

**(f) `rcases` alternatives must bind uniform names.** Writing
`rcases h with h0 | h1 | h1` then referring to `h0` and `h1` in a combined
`simp` fails in every branch that did not bind that name. Use one name.

**(g) A `native_decide` sweep over even `K ∈ {2,4,6}` was refused throughout.**
It would only re-check the Sounio census and would swap a `sorry` for a subtler
gap on a `∀K` statement. It also turned out to be unnecessary.

**(h) A rotation-permutation of `List.range n` was never needed.** The obvious
route to the telescoping identity is to show `i ↦ (i+1) % n` permutes
`List.range n`, which for general `n` is real work. Splitting the last step off
with `iSum_range_succ` turns the modular index into a plain one on
`List.range m`, where `Nat.mod_eq_of_lt` applies pointwise. Cheaper, and it
avoids a general-`n` permutation proof entirely.

## 5. Honest scope — what this does NOT establish

The proof is complete. **The statement is narrower than the paper's claim**, in
two ways, both stated in the file header and neither lifted here:

- **(a) Monochromatic in `p`.** The theorem fixes one `ZDPrim` and forbids odd
  cycles for that surgery alone. Bipartiteness of the **union** graph over all
  84 primitives does **not** follow — a union of bipartite graphs need not be
  bipartite. If the intended unit-distance graph admits any prim on any edge,
  that is a separate open statement.
- **(b) `isKDiff` edges only.** The edge hypothesis is
  `zdEdge ∧ ∃ K, isKDiff K (u − v)`, i.e. differences with unit coordinates and
  an exact support. The full `zdEdge` graph over unrestricted `SedVec` is not
  covered. Both 3.1 and 3.2 assume `isKDiff`, so this restricts the graph, not
  just the proof.

The declaration name `sedenion_zd_surgery_bipartite` reads broader than either
of these. A NAME WARNING now sits directly above it telling readers to read the
binders. **Do not cite this file for χ=2 of the full unit-distance graph.**

Conversely — and this correction matters — the census sign-reduction debt
recorded earlier is the **census's** problem, not the proofs'. The census
enumerates unsigned supports × 84 prims while `cross` depends on the *relative*
signs of the live coordinates, so the census rows do not by themselves cover
every `isKDiff`. But Theorem 3.2 and the main theorem quantify over **all** sign
patterns allowed by `isKDiff`, so they are strictly **stronger** than the census
rows rather than dependent on them. Earlier wording in this handoff invited the
opposite reading and has been corrected.

### LLM-offload math reviews (mandatory, all 2026-08-16)

Four `bin/llm-offload -t math-review -p xai` runs against Grok 4.5, one per
milestone. Raw output in `/tmp/llm-offload-{S3pon4,WVrwhO,0YV2OZ,0tNgK0}/`, all
logged in `.claude/llm_offload_log.md`. The final run found **no arithmetic
FAIL**. The reviews earned their keep three times:

1. **The monochromatic-`p` gap** — genuinely new, not a restatement of the
   known `sorry`s. It changes what the main theorem *means*, and it is why
   scope note (a) exists.
2. **The `isKDiff` narrowing** — caught as a silent weakening of the edge
   relation; scope note (b).
3. **The mixed-K lead** — the review pointed out that the docstring overstated
   the remaining gap, since the hit-count parity works unchanged for
   heterogeneous odd `K_i`. That materially lowered the cost of the final
   assembly.

## 6. Next actions for whoever picks this up

1. **Lift narrowing (a).** Bipartiteness of the union over all 84 primitives is
   the actual paper claim and does not follow from the per-`p` result. This is
   the substantive open mathematics, not a formalisation chore.
2. **Lift narrowing (b).** Decide whether a `zdEdge` with `‖d·p‖² = 2` forces
   `isKDiff` on the difference. If it does, prove it and drop the hypothesis;
   if it does not, the graph in the paper needs restating.
3. **Settle the census sign reduction**, or retire the census rows in favour of
   Theorem 3.2, which is strictly stronger over `isKDiff`.
4. ~~Four real `sorry` remain elsewhere in `formal/lean4`: `EpistemicEffects.lean`
   (1), `SounioDeGreyChi5Real.lean` (1), `SounioGradedModal.lean` (2).~~
   **Done — re-measured 2026-08-27: zero real `sorry` tactics remain anywhere in
   `formal/lean4`.** The `iSum_perm` / `iSum_reindex` pattern in §4(a) remains
   reusable for any future proof that hits the same `whnf` wall.
