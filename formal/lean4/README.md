# formal/lean4/ — Lean 4 proof tree

## The contract

**No Mathlib. No `sorry`.** Declared in dozens of file headers and enforced
by review. `lakefile.lean` has no Mathlib dependency; `lean-toolchain` is
pinned to `leanprover/lean4:v4.33.0` (aligning with origin/main — do not
unpin to `stable`, it is a moving ref that has broken main before).

Build a single file (preferred — full `lake build` is heavy):

```
export ELAN_HOME=/workspace/.tmp/elan
export PATH="$ELAN_HOME/bin:$PATH"
lean <File>.lean        # exit 0 + no "declaration uses sorry" warning = done
```

## Read this before writing tactics: what is Mathlib-only

The eight failures cursor-3 hit (2026-08-17) were not bad proofs — they were
Mathlib tactics that do not exist in core Lean. Every entry below was
**verified by compiling** against core 4.33.0 on 2026-08-17: the positive
replacements are `TACTICS_CORE_WITNESSES.lean` in this directory (exit 0),
and each Mathlib-only item was observed to fail with `unknown tactic` /
`unknown constant` in a throwaway probe.

| You reached for (Mathlib-only) | Error | Core-only replacement |
|---|---|---|
| `by_contra h` | unknown tactic | `cases Nat.lt_or_ge a b with \| inl … \| inr …` for Nat order goals; `rcases Int.le_total 0 x` for Int; `omega` kills the bad branch |
| `set x := e with hx` | unknown tactic | `have hx : x = e := rfl` (or `show`) + `rw [hx]` |
| `push_cast` | unknown tactic | `simp` — the cast lemmas are in core's simp set (but are **not** addressable constants) |
| `nlinarith` | unknown tactic | named monotonicity lemmas (`Nat.mul_le_mul_left/right`, `Nat.pow_le_pow_right`) then `omega` |
| `positivity` | unknown tactic | case-split (`rcases Int.le_total`) + `Int.mul_nonneg` etc. |
| `ring` | unknown tactic | `rw [Int.add_mul, Int.mul_add, …, Int.mul_comm]` then `omega`, keeping every product an omega atom |
| `field_simp` | unknown tactic | manual field lemmas; on Nat/Int rarely needed |
| `norm_num` | unknown tactic | `decide` for closed numerics |

Also **not addressable in core** (use `simp`, not `simp only [name]`):
`Nat.cast_add`, `Int.ofNat_add` — `unknown constant`.

These **are core** — don't replace them: `rcases`, `obtain`, `rintro`,
`by_cases`, `exfalso`, `omega`, `simp`, `simp only`, `split`, `decide`,
`cases … with`, `induction … with`, `constructor`.

Two gotchas that cost real time even after you know the table:

1. **`omega` is linear.** `a*b + a*b` and `2*a*b` are different worlds to
   it (`2*a*b` parses as `(2*a)*b` — nonlinear). Keep products as atoms:
   write `a*b + a*b`, or establish `2*(a*b) = a*b + a*b` first.
2. **`rw` auto-closes goals with `rfl`** — a trailing `decide` after a
   `rw` that already finished gives "No goals to be solved".

The witnesses for every row live in `TACTICS_CORE_WITNESSES.lean`, each
annotated with which shipped kernel-clean file uses the same pattern
(`EpistemicEffects.lean`, `SounioGradedModal.lean`). If you hit a new
Mathlib-only tactic, add a row + a witness there first, then link it here.

## Layout

- `README.md` (this file) — start here
- `TACTICS_CORE_WITNESSES.lean` — compile-verified tactic table witnesses
- `lakefile.lean` — target list; new proof files should be registered
- Proof modules: see the lakefile and `../README.md` for the Phase 8 map
  (ElfLinker, TypeChecker) and the newer lanes' file headers
