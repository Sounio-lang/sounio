<!-- docs:meta
topic_id: repo.docs.audit.e043-rust-macro-design-dispatch-2026-07-12
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e043-rust-macro-design-dispatch-2026-07-12
-->

# Design dispatch — E043 `ident!(...)` Rust-macro rejection (needs a design decision, not a port)

**Filed:** 2026-07-12 · **Status:** OPEN (design, not a mechanical port) · **Protocol:** CLAUDE.md §8.

Follow-up to the guard-wiring arc (#798 → E213 #801, E216 #802, E040/E041 #808).
E040/E041 were clean ports because `let mut`/`&mut` use the dedicated `Mut`
keyword token, which has **no valid meaning** in Sounio. **E043 is not a clean
port** — it is a genuine language-design question, documented here so it is
decided deliberately rather than wired with a false-reject-prone heuristic.

## The gap

`println!("hi")` (and any `ident!(...)`) compiles **clean** under the default
`souc check` (Madaros v0.80.0) — it silently accepts invalid, Rust-shaped code.
The error catalog documents E043; the checker/parser never fires it. This is the
one Rust-compat case that accepts invalid code (E040/E041 already reject).

## Why it is NOT a clean port — the ambiguity

`!` is **both** a prefix and a postfix operator in the modular parser:

- **Prefix** `!b` — logical-not (`self.parse_unary(UnaryOp::OpNot)`, `self-hosted/parser/exprs.sio:208`).
- **Postfix** `x!` — carries postfix **precedence 13** (`self-hosted/parser/exprs.sio:115-122`); disambiguated from a line-leading prefix `!b` by `had_newline_before` + adjacency.

Therefore `ident!(...)` is **token-identical** to a valid Sounio expression:
`ident` → postfix-`!` → `(...)` (a parenthesised expression / call). A naive
"`Ident` `Bang` `LParen` ⇒ E043" rule would **false-reject** any real
`ident!(...)` postfix-`!`-then-parens usage. This is exactly the failure the
guard-wiring regression gate forbids.

## Evidence

- Prefix vs postfix `!`: `exprs.sio:208` (prefix) and `exprs.sio:115-122` (postfix, precedence 13).
- **Engine divergence:** the `lean_single` seed *does* reject macros — `tc_rust_macro` (`self-hosted/compiler/lean_single.sio:4295`, "Sounio does not use Rust macros (ident!())"). Madaros accepts them. So the two engines already disagree on `ident!(`; the seed treats it as a macro, the modular parser as postfix-`!`.
- **Corpus check (2026-07-12):** a sweep for a real postfix-`!` *operator* usage (excluding string/comment `!`) across `tests/run-pass/`, `stdlib/`, `examples/` found **none**. The ~345 raw `char!` hits are all inside string literals / comments (`"hi!"`, etc.). Postfix-`!` appears to be **parser machinery with no live use** in the tree.

## Design options

1. **Adjacency heuristic (lowest effort).** Fire E043 when `Ident`, `Bang`, `LParen`
   are all **immediately adjacent** (`PT_END[ident]==PT_START[bang]` and
   `PT_END[bang]==PT_START[lparen]`) — the exact shape Rust macros take. A
   deliberately-written postfix-`!`-then-call would still be accepted if it has
   any whitespace (`x! (y)`). Risk: a tight `x!(y)` postfix usage would be
   caught; the corpus check suggests that shape does not occur, but the
   regression sweep must confirm it.
2. **Retire postfix-`!` and reclaim `ident!` (cleanest, bigger decision).** If
   postfix-`!` is genuinely unused (the corpus check supports this), remove the
   postfix-`!` production and make `ident!` an unambiguous macro error — aligning
   Madaros with the `lean_single` seed. This is a **language change** and needs
   the maintainer's sign-off (it removes a syntactic form, even if vestigial).
3. **Semantic (highest effort).** Let it parse, then reject in the checker when
   postfix-`!` is applied to a non-postfix-`!`-able operand (e.g. a function
   name). Complex, and only catches a subset. Not recommended.

## Recommendation

Decide between **(1)** and **(2)**; do not ship a bare syntactic rule without one.
- If the maintainer confirms postfix-`!` is dead → **(2)** is the honest fix
  (remove the vestigial operator, reclaim `ident!` for a clear macro diagnostic).
- Otherwise → **(1)** the adjacency heuristic, gated hard on the regression sweep.

## Regression gate (blocks landing, either option)

Same as the guard-wiring dispatch (#798): baseline the full suite with
`JOBS=1 scripts/dev/run_sio_test_suite.sh` on `main`, implement, rebuild Madaros
under the pod-wide build lock, re-run, and land only on **zero new failures**.
Add a positive test (`ident!(...)` → E043) and, for option (1), an explicit
negative test proving whitespace-separated postfix-`!` (if kept) still parses.

## Implementation notes

- Insertion point (option 1): the postfix loop in `self-hosted/parser/exprs.sio`
  around the Bang handling (`exprs.sio:115-122`), or the call/primary site — wherever
  the `Ident`→`Bang` transition is first observed with the token spans available.
- Emit via the parser error path used by E040/E041: set `p.had_error = true`,
  `p.error_count += 1`, print `error[E043]: ...` + a help line (`use a regular
  function call instead of macro syntax`), then recover.
- The `#`/`#[` attribute case (**E042**) is the remaining Rust-compat item; it is
  a straightforward item-level detection (no operator ambiguity) and can be a
  small separate PR.
