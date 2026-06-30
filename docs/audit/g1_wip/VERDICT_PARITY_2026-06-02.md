<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.verdict-parity-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.verdict-parity-2026-06-02
-->

# Verdict-parity: E014 (usize index) + E003 (module-global mutation) — 2026-06-02

Follow-up to [CRASH_CLASS_ZERO](CRASH_CLASS_ZERO_2026-06-02.md). The 3 programs that
crash-fixed to a *divergent* rc=1 (modular rejects, canonical `bin/souc` accepts) were
root-caused. Two distinct, narrow divergences fixed; the third (E008) is the deep #1 lever
(separate work).

## E014 — `usize` not recognised as an integer

`x as usize` lowered to `ty_named("usize")` (kind `TyNamed`) because `checker_lower_named_type_mut`
/ `lower_named_type` had no `usize` case. Array-index checks (`idx_ty.kind != TyI64`) then
rejected it (E014 "array indices must be integers"). The oracle `lean_single` collapses every
integer width — i32/u32/i64/u64/usize — to a single integer type code, so the faithful fix is:

- new `name_is_usize` (compat.sio), and `usize → ty_i64()` in both named-type resolvers.

Mapping to `i64` (the codebase's canonical integer, and what `as usize` casts from) means the
existing `!= TyI64` index check passes with no change. Fixes `lean_mini_compiler`,
`lean_utils_self_host`, and every `arr[i as usize]` program.

## E003 — spurious immutable-binding error on module-global assignment

`GLOBAL = GLOBAL + 1` (module-level `var`) reported E003 "cannot modify an immutable binding"
because module globals are **not bound into the checker env** in the *mut path, so
`env.lookup_is_mutable(name)` returned its `false` default. Fix: gate E003 on
`env.has_binding(name) && !lookup_is_mutable(name)` — fire only for a name that IS bound (a
local) and immutable (a `let`). Unbound names are globals (mutable per the oracle) or genuinely
undefined (a different error class), never a local `let`. Preserves the true positive on local
`let` mutation (verified: local `let L; L = …` still E003, matching the oracle). Applied to
both the *mut `checker_check_assign_mutability_inplace` and by-value `check_assign_mutability`.

## Verification (mc_Q vs committed mc_P baseline, 847 examples @1GB)

`13 programs 1→0`, **0 non-crash→crash, 0 pass→fail, 0 crashes** (157 rc=0 / 690 rc=1), g1
gate PASS.

Oracle-vetted all 13 wins: **10 genuinely agree** with `bin/souc` (rc=0). **3 are net
false-passes** — and they were *already divergent* at baseline (modular rejected them for a
**wrong** reason; the oracle rejects them for a **different** reason the modular checker is
pre-existingly lenient about):

| program | baseline modular | oracle rejects for | after fix |
|---|---|---|---|
| `octonion_derivation_algebra` | spurious E003 | effect-not-declared | passes (effect-leniency unmasked) |
| `octonion_projective_plane` | spurious E003 | effect-not-declared | passes (effect-leniency unmasked) |
| `test_zstd` | E014 (+E001/E004) | unknown-identifier | passes (ident-leniency unmasked) |

The E014/E003 fixes are correct (they remove genuinely spurious errors — and `lean_utils_self_host`
*requires* the E003 fix). The false-passes stem from **separate pre-existing leniencies**
(effect-not-declared and unknown-identifier checking, both confirmed absent in the modular *mut
path) that removing the spurious error merely exposed — not new wrong logic. The effect-leniency
is the **same `current_effects` non-propagation** that drives E008/E170; the deep E008 fix
(propagate `current_return_type`/`current_effects`) is expected to close the 2 octonion
false-passes as a side effect.

**Follow-up (separate):** the modular *mut checker does not catch (a) effect-not-declared, (b)
unknown identifiers/methods — both pre-existing. Tracked with E008 (the #1 corpus lever).
