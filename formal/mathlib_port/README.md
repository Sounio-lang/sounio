# Mathlib port staging area

This directory holds Mathlib-dependent variants of files from `formal/`,
kept separate so the main `formal/` Lake project stays self-contained
(no Mathlib dependency, matching its existing `lean-toolchain` and
build time).

## `OctonionAlgebra_proved.lean`

Same statements as `../OctonionAlgebra.lean`, but every identity that was
previously an `axiom` (distributivity, alternative laws, flexibility,
the three Moufang identities, scalar-multiplication bilinearity, conjugate
anti-multiplicativity, `x·conj(x) = |x|²·e0`, and norm multiplicativity —
the Degen eight-square identity) is now a real `theorem`, discharged by
the `ring` tactic. Verified to compile with `exit=0` against
`leanprover-community/mathlib4` (master, pinned via `lake exe cache get`)
on 2026-09-22.

**Bug found and fixed in the process.** The `octMul` formula in the original
axiom-based file had incorrect signs on the `e4..e7` cross terms in
components `e1, e2, e3, e5, e6` (5 of 8): it agreed with the canonical
implementation (`../../stdlib/algebra/octonion.sio`, mirrored in
`../../scripts/research/ossm_168_dryrun/octonion.py`) on every basis
product `e_i * e_j` that the file's own `basis_*` spot-checks exercised,
but did **not** satisfy the alternative law in general. Minimal
counterexample: for `x = e4+e5, y = e2`,
`x·(x·y) = (0,0,2,0,0,0,0,0)` but `(x·x)·y = (0,0,-2,0,0,0,0,0)`.
Found by attempting to actually discharge the axioms with `ring` here
(it failed on the buggy formula), confirmed by comparing against both a
from-scratch symbolic Cayley-Dickson doubling of quaternions and the
canonical `octonion.py`. `../OctonionAlgebra.lean` has since been fixed
to use the same sign convention as `octonion.py`/`octonion.sio` (still as
axioms there, to keep that file Mathlib-free); this file has the same fix
plus the real proofs.

**The published 168-theorem result was not affected**: the associator
count in `../FanoLabellingOrbits.lean` (`fano_automorphism_group_card`)
is a pure combinatorial statement about the Fano plane's automorphism
group, proved by `native_decide`. That file's own header states it is
"deliberately self-contained (no `import`, no Mathlib, no `OctonionAlgebra`
dependency)" — so this is not merely "the proof doesn't call `octMul`",
there is no import path between the two files at all, and the bug could
not have propagated into that proof through any dependency chain. The bug
was confined to this standalone Lean formalization of the octonion algebra
itself.

**Reviewed 2026-09-22** via `bin/llm-offload -t math-review -p xai` (Grok
4.5, run from `/workspace/sounio` since this local checkout has no
`~/.sounio-keys.env`). Findings: counterexample and fix both confirmed by
independent symbolic expansion; `NonUnitalNonAssocRing` (not `Ring`) and
`StarRing`'s anti-multiplicative `star_mul` order both confirmed correct.
One `[FAIL]` (this file's docstring claimed the instances in §20 hadn't
been built yet — fixed) and a few `[TIGHTENABLE]` notes folded into
"Next step" below (a non-associativity witness for general `R`, a
`One`/`NonAssocRing` instance given the two-sided unit is already proved).

## `OctonionGeneral.lean`

Generalizes `Octonion` from `Int` (in `OctonionAlgebra_proved.lean` above) to
an arbitrary `[CommRing R]`, with the *fixed* standard-octonion coefficients
(not yet the fully parametrized `OctonionAlgebra R a₁ ... a₇` — see "Next
step" below). All 15 identities are proved via `ring`, generalized from `Int`
to `R`, same as above.

Also adds the algebraic instances:

- `AddCommGroup (Octonion R)` — built directly from the `oct_add_*` lemmas
  (not via `Function.Injective.addCommGroup` transferred along an `Equiv` to
  a product type, the way `Mathlib.Algebra.Quaternion`'s `equivProd` does it:
  that transfer lemma needs `Add`/`Neg`/`Zero`/`SMul ℕ`/`SMul ℤ` instances on
  the target to already exist, which is circular here since those are exactly
  what's being constructed. `nsmul := nsmulRec`, `zsmul := zsmulRec` are the
  standard generic bootstrapping definitions for exactly this situation).
- `NonUnitalNonAssocRing (Octonion R)` — **not** `Ring`, since octonions are
  non-associative (`Mathlib.Algebra.Quaternion`'s `instRing` can require
  `mul_assoc` because quaternions *are* associative; octonions aren't, so
  `NonUnitalNonAssocRing` — `AddCommGroup` + `Mul` + distributivity +
  `zero_mul`/`mul_zero`, no associativity or unit required — is the correct
  target class).
- `StarRing (Octonion R)` — conjugation as `star`, reusing
  `oct_conj_involution` / `oct_conj_antimultiplicative` / a direct proof of
  `star_add`.

Sanity-checked (§21 in the file) that generic Mathlib lemmas apply out of the
box on `Octonion ℤ` through these instances alone — `add_comm`, `star_star`,
`star_mul`, `zero_mul` — none of which are proved in this file.

## To rebuild these files

From a Lean 4.30.0-rc2 project with Mathlib as a dependency:

```bash
lake env lean OctonionAlgebra_proved.lean
lake env lean OctonionGeneral.lean
```

## Next step (not done here)

1. Generalize further from fixed coefficients to a `QuaternionAlgebra`-style
   `OctonionAlgebra R a b c d` (the `a b c` matching `QuaternionAlgebra R a b c`'s
   own quaternion-part coefficients, `d` for the Cayley-Dickson doubling
   parameter) — see `../../ZULIP_DRAFT_octonions.md` for why the *generic
   doubling functor* approach (`CayleyDickson (CayleyDickson (CayleyDickson R))`)
   is avoided instead (stalled in a 2023 Lorentz Center workshop attempt on
   Mathlib's associative-biased typeclass hierarchy); the explicit-8-field-struct
   approach here and in `Mathlib/Algebra/Quaternion.lean` sidesteps that.
2. `IsAlternative` / non-associative-algebra-specific instances (Moufang loop
   structure, the alternative law as a typeclass) if Mathlib has or wants one —
   not checked yet.
3. `DivisionRing`-adjacent instances for `R` an ordered field (norm
   multiplicativity, already proved generically, is the key ingredient), the
   way `Quaternion.lean`'s `instDivisionRing` does for `ℍ[R]`.
4. A concrete non-associativity witness for `OctonionGeneral.lean` itself
   (flagged by the 2026-09-22 math-review as tightenable): `OctonionAlgebra_proved.lean`
   / `../OctonionAlgebra.lean` already have `oct_nonassociative` on `Int`; the
   general-`R` file only has the alternative/Moufang laws, which hold, and
   never states non-associativity holds in general (it doesn't, for
   nontrivial `R` — same `e1, e2, e4` witness, generalized).
5. A `One (Octonion R)` / `NonAssocRing (Octonion R)` instance: `oct_mul_one`
   / `oct_one_mul` are already proved, so `e0` is a genuine two-sided unit —
   just not yet wired into a typeclass instance.
