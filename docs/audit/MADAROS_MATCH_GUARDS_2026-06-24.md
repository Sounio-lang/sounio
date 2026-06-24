<!-- docs:meta
topic_id: repo.docs.audit.madaros-match-guards-2026-06-24
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-match-guards-2026-06-24
-->

# Madaros match guards — working (2026-06-24)

*Branch off `main`. `match n { x if x > 10 => … }` now evaluates the guard; previously the
guard was silently ignored and the first pattern-matching arm was always taken — a silent
wrong result.*

## Root cause — the guard was parsed and thrown away

`MatchArm` had no guard field (`{ pattern, body, span }`), and the parser
(`exprs.sio`) parsed the `if <cond>` then **discarded it**:
`let _guard_box = parser_take_expr_box()` (the `_` prefix). So no guard ever reached the
checker or lowering, and every guarded arm behaved as unguarded → the first arm whose
*pattern* matched won, ignoring the condition.

## Fix (4 layers)
1. **AST** (`ast.sio`): add `guard: Option<Box<Expr>>` to `MatchArm`.
2. **Parser** (`exprs.sio`): capture the guard into the arm (`guard: guard_opt`) instead of
   discarding it; the two desugar constructions (`while let`/`?`-style) pass `guard: None`.
3. **Lowering** (`lower.sio`): new `lower_arm_guard` — after the pattern matches and bindings
   are made, evaluate the guard and `branch_false(guard, l_skip)` so a false guard **falls
   through to the next arm**. Wildcard/binding arms now allocate an `l_skip` too (dead when
   there is no guard). Applied to all pattern kinds (wildcard, binding, Some/None/variant,
   int-literal, bool).
4. **Checker** (`check.sio`): type-check the guard with the pattern variables in scope.

## Verified (madaros from this source, exit codes are mod 256)
- `match n { x if x>10 => 100, x if x>0 => 50, _ => 0 }`: `classify(5) → 50` (was 100).
- 4-way chain `g`: `g(5)=3, g(50)=2, g(500)=1, g(0)=4` (correct fall-through across 3 guards).
- Guard on an enum pattern: `E::A if n>0 => 1` selects correctly.
- No-regression: guard-less `Option`/enum matches still `→ 42 / 23`; 53/90 run-pass =
  prebuilt main +6, 0 regressed; madaros self-builds.

## Honest scope
- The guard reads the bound pattern variables at the arm; standard semantics.
- The checker type-checks the guard expression but does not yet *enforce* it is `bool`
  (madaros leniency); a non-bool guard would still lower (the `branch_false` tests the value).
- Tuple/struct destructuring patterns with guards follow the same path but were not
  separately exercised here.

## AI disclosure
Fix by AI agent (Claude) under human direction; root found by reading the parser
(`_guard_box` discard) and the guard-less `MatchArm` struct. Every claim backed by a
re-runnable probe (with the mod-256 exit-code caveat).
