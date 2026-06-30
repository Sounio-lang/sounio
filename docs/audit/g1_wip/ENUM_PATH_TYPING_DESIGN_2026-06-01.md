<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.enum-path-typing-design-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.enum-path-typing-design-2026-06-01
-->

# Enum-path typing — design note + scoping (2026-06-01)

Output of the "enum-path typing session." No code was written: the session's real result is
the characterization below + a decision that belongs to the user. **No landable increment
exists** for this in isolation (see "Why there is no checkpoint").

## The problem

To enable `ItemEnum` net-clean, the modular checker must type enum-variant paths the way
canonical `bin/souc` does. Today `check_path_expr` types `Color::Red` as `ty_named("Color")`
(it copies only the path's HEAD segment via `checker_copy_string_list_to_name` = `seg.head`),
and `types_compatible` / `binary_result_type` have no enum rule, so the modular checker
rejects `i64 == Color::Red` (E004) — diverging from canonical, which accepts it.

## Canonical enum model (VERIFIED via bin/souc compile-oracle)

`bin/souc <src> <out>` compile success = valid in real Sounio (bin/souc has NO `--check`
flag — it lex-errors on it; use compile mode as the oracle). Probe battery
(`.dbg/g1corpus/sem/`):

| Program | Canonical | Implication |
|---|---|---|
| `fn f()->E { E::A }` | VALID | fieldless variant is compatible with its enum type `E` |
| `fn f()->i64 { E::A }` | VALID | …and with `i64` (assignment) |
| `let x:i64 = E::A` / `let x:E = E::A` | both VALID | i64↔enum assignment, both directions |
| `c:i64 == E::A` | VALID | i64 == enum compare |
| `E::A == E::B` | VALID | enum == enum compare |
| `E::A + 1` | **INVALID** | NOT int-arithmetic — so `E::A` is NOT simply `i64` |
| `match e { E::A=>.. }` | VALID | pattern match on enum |
| fielded `S::C{r:1}` construction | **INVALID** | separate gap (fielded enum ctor unsupported even in canonical) |

**Model:** a fieldless enum variant has a DISTINCT enum type (not `i64` — else `E::A + 1`
would be valid) that is compatible with `i64` and with its own enum **for assignment and
equality, but not arithmetic**. C-style.

## The implementation constraint

- `types_compatible(a, b)` (compat.sio:10) and `binary_result_type(op, l, r)` (compat.sio:739)
  are **pure** — they take only `TypeEntry`s, with no access to `(*c).enums`. So they cannot
  tell `ty_named("E")` (enum) from `ty_named("Point")` (struct). There is **no `TyEnum` kind**
  and **no enum marker** on `TypeEntry` today (enums and structs are both `TyNamed`).
- `check_binary_op_types` / `check_binary_with_operand_types` ARE Checker methods (have
  `self.enums`), so a helper `is_fieldless_enum(self, ty) -> bool` (`self.enums.find(ty.name)>=0`
  + all variants fieldless) is feasible **there** — the viable hook for the compare cases.
- Precedent for marking a `TyNamed`/base type exists: `unit_id`, `refinement_id`,
  `ontology_id`, `epistemic_meta_id` are spare ID fields already used this way
  (`lower_named_type`, compat.sio). So Option B can reuse a spare field instead of adding a
  new one — but still must set it at every enum-value-producing site and read it in the pure
  functions.

## Two implementation options

**Option A — minimal, DIVERGENT.** Handle only binary `==`/`!=` for `i64`-vs-fieldless-enum
in `check_binary_with_operand_types` (consults `self.enums`). Covers `enum_match`'s compare.
Low blast radius. But assignment cases (p2/p6: `i64 = E::A`, `f()->i64 { E::A }`) stay wrong
— deliberately incomplete vs canonical. Writing known-divergent semantics.

**Option B — structural, FAITHFUL.** Mark enum-ness on the `TypeEntry` (reuse a spare ID
field, e.g. set it when `check_path_expr`/`lower_named_type` produces an enum type), then add
i64↔enum rules to `types_compatible` (assignment) and `binary_result_type` (`==`/`!=`), and
keep arithmetic rejected. Matches canonical. **Corpus-wide blast radius** (every enum-value
construction site must set the marker; the pure functions are used everywhere).

## Why there is NO checkpoint (the disqualifier)

`enum_match` is the **only** corpus program any of this touches. Making it correct (rc=0)
requires BOTH:
1. the typing fix (this note), AND
2. ExprIf migration (codegen — its `Color::Red` is nested in `if c == Color::Red`, which
   bridges to by-value `check_if_expr` → by-value `check_path_expr` → the SRET crash).

Neither alone moves `enum_match` off rc=139:
- Typing fix + ItemEnum ON → `enum_match` still CRASHES (ExprIf) → cannot commit net-clean.
- Typing fix + ItemEnum OFF → enum types never produced → the fix is DEAD CODE.

So unlike Build A (`cb92d66a9`, net-positive and verifiable on its own — 80 rescues), there
is **no incremental landable unit**. Typing + ExprIf must land **together**, as one combined
push, verified against the canonical compile-oracle (not the lenient baseline). That is
multi-workstream work (TypeEntry marker + compat/binary changes + ExprIf `*mut` migration)
to flip ONE corpus program from 139 to 0.

## Decision for the user (this is a scope call, not a technical one)

This is justified only as **compiler completeness/correctness** (the self-hosted checker
should handle enums like canonical), NOT as measurable corpus wins (it is ~1 program, and
the checker is lenient on enums generally). Options:

- **(a) Commit to the full combined push** — TypeEntry enum marker (Option B) + compat/binary
  rules + ExprIf `*mut` migration, landed together, verified against canonical. A deliberate
  correctness investment; several builds, real blast radius, one combined commit.
- **(b) Shelve enum-enablement, bank this design** — and move to higher-yield work. The
  collector body (`ddc7a8b7e`) and the 80-rescue spine progress (`cb92d66a9`) already stand.
- **(c) Reconsider whether the modular checker needs enum-enablement near-term at all** — its
  only effect is making it stricter on the 1 enum program, and that requires matching
  canonical's C-style leniency to be correct.

Recommendation: (b) unless enum support is a near-term requirement for the dissertation /
conference pipeline. The verified canonical model above is the durable asset; re-enter via a
single combined typing+ExprIf push if (a) is chosen.

## OUTCOME (user chose (a)): ItemEnum ENABLED — landed `cef81349a`

The full combined push was executed and landed. Sequence:
- `102350faa` — ExprIf routed onto the *mut spine (split-and-bridge), +32 crash rescues, 0
  regressions (checkpoint, ItemEnum off).
- `cef81349a` — the combined unit: FULL *mut ExprIf handler (then-block via
  checker_check_opt_block_inplace + else via checker_check_expr_inplace, so `else if` chains
  route through *mut), binary `i64==enum` accept in check_binary_with_operand_types, and
  ItemEnum WIRED.

Result: `enum_match.sio` → **rc=0, canonical-faithful** (was a baseline lenient false-pass;
crashed when ItemEnum was naively enabled). **0 corpus regressions, 0 pass→fail**, g1 gate
PASS. 116 crashes→completions across the arc (Build A 80 + ExprIf 32 + this 4); spot-check vs
canonical = most faithful, a minority reflect the modular checker's PRE-EXISTING
incompleteness (now visible, not masked by the crash).

Probe battery: **6/8 faithful** to canonical (p1–p5, p8). Two divergences REMAIN (synthetic,
not in corpus — so 0 measured regressions — but real):

### Follow-ups (do NOT chain on low context — the cascade restarts here)
- **p7 `let x: E = E::A` — NEW latent crash introduced by enabling ItemEnum.** `let` with a
  TyNamed annotation + path init routes through `check_binding_init_expr` (check.sio:~12782),
  whose ExprPath special-case calls **by-value** `check_path_expr`; with the enum now
  collected it takes the enum-found branch → the 8MB-self SRET smash. Fix: a *mut binding-init
  path (route that init through checker_check_path_expr_inplace / the *mut spine). Absent from
  the corpus, so the sweep is clean — but it is a real reachable crash shipped by flipping
  ItemEnum on.
- **p6 `let x: i64 = E::A` — divergent rejection.** enum→i64 ASSIGNMENT compat is unhandled
  (only binary ==/!= was fixed). Fix: add the enum-int rule at the `types_compatible` call
  sites already catalogued — return (check.sio:2605), let-binding (check.sio:2875), fn-args —
  via a Checker-method helper (those sites have self.enums), OR the TypeEntry-marker route
  (124 literals) for a uniform fix. Canonical accepts p2/p6 (enum coerces to i64).

These are the next unbounded leg (p7 → *mut binding-init → its callees; p6 → every compat
call site). The verified canonical model + probe battery (`.dbg/g1corpus/sem/`) is the spec.
