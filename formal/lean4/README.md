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
| `nlinarith` | unknown tactic | named monotonicity lemmas (`Nat.mul_le_mul_left/right`, `Nat.pow_le_pow_right`) then `omega` |
| `linarith` | unknown tactic | `omega` |
| `push_neg` | unknown tactic | `Classical.byContradiction`, or push the negation by hand |
| `tauto` | unknown tactic | `rcases` on the decidable split, then `omega`/`decide` |
| `positivity` | unknown tactic | case-split (`rcases Int.le_total`) + `Int.mul_nonneg` etc. |
| `ring` | unknown tactic | `rw [Int.add_mul, Int.mul_add, …, Int.mul_comm]` then `omega`, keeping every product an omega atom |
| `field_simp` | unknown tactic | manual field lemmas; on Nat/Int rarely needed |
| `norm_num` | unknown tactic | `decide` for closed numerics |
| `interval_cases n` | unknown tactic | `match` on numeral patterns with `_ + k => omega` last |

Also **not addressable in core** (use `simp`, not `simp only [name]`):
`Nat.cast_add`, `Int.ofNat_add` — `unknown constant`.

> **Correction, 2026-08-17 — `push_cast` is NOT Mathlib-only.** An earlier revision of
> this table listed it as `unknown tactic`. That is wrong, and so is the same claim made
> independently from experience by `cursor-3`. `push_cast`, `norm_cast` and
> `exact_mod_cast` are all in Lean core and all run under 4.33.0. What *is* missing is
> the two lemma names in the line above, which is the real reason a `push_cast` session
> goes wrong. See `CORE_PROOF_PATTERNS.md` §11 for the witness. Prefer `omega` anyway:
> it crosses `Nat → Int` unaided, truncated subtraction included.

These **are core** — don't replace them: `rcases`, `obtain`, `rintro`,
`by_cases`, `exfalso`, `omega`, `simp`, `simp only`, `split`, `decide`,
`cases … with`, `induction … with`, `constructor`, `subst`, `generalize`,
`push_cast`, `norm_cast`, `exact_mod_cast`, and `exact?` (a *search* aid — it
prints `Try this:` and should never be left in committed code, since it
re-searches on every build).

Two gotchas that cost real time even after you know the table:

1. **`omega` is linear.** `a*b + a*b` and `2*a*b` are different worlds to
   it (`2*a*b` parses as `(2*a)*b` — nonlinear). Keep products as atoms:
   write `a*b + a*b`, or establish `2*(a*b) = a*b + a*b` first.
2. **`rw` auto-closes goals with `rfl`** — a trailing `decide` after a
   `rw` that already finished gives "No goals to be solved".
3. **`omega` is stronger than the `linarith` habit assumes.** It proves
   `min a b ≤ a`, `max a b ≥ b`, and truncated `Nat` subtraction in both
   directions (`a ≤ b → a - b = 0`, `b ≤ a → (a - b) + b = a`) natively.
   Reach for it before hunting a replacement for `linarith`.

### Triage an error before believing it

Three outcomes look similar in a terminal and mean completely different things.
Classifying them is the whole method behind this table:

| What you see | What it means | What to do |
|---|---|---|
| `unknown tactic` | the tactic is genuinely absent from core | use the replacement column |
| ran, `unsolved goals`, goal is **true** | a real limitation of the tactic | restructure or pick another tactic |
| ran, `unsolved goals`, goal is **false** | the tactic was **right** and your statement is wrong | fix the statement |

The third row is not hypothetical. `kimi-cli1` probed
`(a b : Int) (h : 0 ≤ a) : 0 ≤ a * b / 2 + 1` with `omega`, which refused, and its
first read was an `omega` limitation. It then checked the goal and found the goal
false: with `b` negative the expression is negative, so `omega`'s counterexample was
correct. A tactic that *refuses* looks identical to a tactic that *cannot*, and only
checking the goal separates them. `k_goal_is_false` in `CorePatternsWitnesses.lean`
proves the falsity constructively, depending on no axioms at all.

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
