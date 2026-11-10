<!-- docs:meta
topic_id: repo.docs.decisions.adr-023-checked-indexing-and-proof-erased-panic
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-023-checked-indexing-and-proof-erased-panic
-->

# ADR-023: Checked indexing, path-conditioned obligations and proof-erased `Panic`

**Status**: accepted (implementation pending)
**Date**: 2026-10-28
**Decided by**: the founder, in the 2026-10-28 decision session
**Builds on**: [ADR-014](adr-014-short-circuit-logical-operators.md) (`&&` and `||` short-circuit)

## Context

ADR-014 makes `i < n && a[i] == x` a real guard: `a[i]` is not evaluated when `i >= n`. It does not make the index *checked*. Measured on 2026-10-27 with the shipped compilers:

| program | Madaros | lean_single |
|---|---|---|
| `s.a[9]` on `a: [i64; 8]` next to `b: [i64; 8]` with `b[1] = 4242` | prints `64` | prints `4242` |
| `s.a[-1]` | prints `1` | prints `0` |

Neither engine checks a fixed-array index. An out-of-range read returns a neighbouring value and an out-of-range write corrupts one, silently. A far index segfaults. A function's signature says nothing about it: a body that indexes is not required to declare `Panic`, and today about 1,085 indexing functions in `stdlib/` and 1,053 in `self-hosted/` do not (heuristic count over single-line signatures, 2026-10-28).

The 2026-10-27 review of the native Arrow codec (#3198) showed the cost. Its decoders are correct only because every guard returns before it indexes. A 102,000-input fuzz campaign found no crash, but under silent out-of-range access a fuzzer cannot see a near miss, so the safety argument had to be made by reading each guard by hand.

## Decision

### 1. Indexing is checked

`a[i]` on a fixed-size array, a slice or a string evaluates `0 <= i < len(a)`. If the check fails, the program panics with the index, the length and the source span. On both engines. A check is removed only under §4.

### 2. The effect checker stays flow-insensitive

The effect row of an expression is the union of the rows of its subexpressions, whatever the evaluation order. In particular `effects(a && b) = effects(a) ∪ effects(b)` and `effects(if c then x else y) = effects(c) ∪ effects(x) ∪ effects(y)`. This is the row discipline of Koka and Eff. It keeps inference predictable and error messages local.

### 3. The verification layer is path-conditioned

Every checked index emits an **obligation** `0 <= i < len(a)` together with its **path condition**: the conjunction of the guards that dominate it. Short-circuiting contributes to path conditions exactly as weakest preconditions prescribe:

- `a && b`: `b` is evaluated under `a`.
- `a || b`: `b` is evaluated under `¬a`.
- `if c { x } else { y }`: `x` under `c`, `y` under `¬c`.
- an early `return` or `break` guarded by `c` puts the rest of the block under `¬c`.

These obligations are what the Lean and F\* exports receive. They are conditional, not unioned.

### 4. A proved obligation erases its check and its `Panic`

A small decision procedure inside Madaros discharges obligations of the form *linear integer constraints over locals, constants and `len(...)`* (difference constraints and intervals; no multiplication of two variables). When the path condition entails the obligation:

1. the runtime check is removed;
2. the `Panic` contributed by that index is removed from the function's effect row;
3. the compiler emits a **certificate** for that erasure: the obligation, the path condition and the derivation.

Lean re-checks every certificate in CI. A certificate that Lean rejects fails the build. Trust therefore rests on Lean's kernel, not on the decision procedure.

### 5. An undischarged obligation in a `Panic`-free function is a compile error

If a function's declared effects do not include `Panic` and one of its index obligations is not discharged, compilation fails. The diagnostic names the index, prints the obligation and the path condition under which it was not proved, and offers two repairs: add a guard that implies it, or declare `Panic`. A signature without `Panic` is therefore a guarantee that the function cannot panic on an index.

### 6. Migration: codemod, then ratchet

1. The decision procedure lands in report-only mode.
2. A codemod adds `Panic` to exactly the functions whose obligations it does not discharge, and to nothing else.
3. §5 becomes an error in the same release. From that day every signature is true.
4. A CI ratchet records the number of codemod-added `Panic` declarations. It may only go down: each proof that removes one is a strict improvement, and a change that adds one is refused unless the function genuinely needs it.

## Consequences

- Out-of-range indexing becomes a reported panic instead of silent corruption. Programs that read out of range today change behaviour; under ADR-018 this is `refusal` for a value that was wrong, and `semantics` where the program now stops.
- Proved indexing costs nothing at run time, and the absence of `Panic` in a signature becomes evidence rather than convention.
- The stdlib fuzz campaigns become meaningful for memory safety: every out-of-range access is now a visible panic.
- The decision procedure is part of the compiler's trusted surface only for performance; its soundness is enforced by Lean.

## Prior art

- **F\*** already behaves this way inside a dependently typed language: indexing is a precondition of a `Tot` computation, so a proved index stays total and an unproved one needs a weaker effect (weakest preconditions, Dijkstra monads).
- **Refinement types** discharge bounds obligations with SMT (Liquid Haskell, Dafny, ATS). Answer Refinement Modification (Kawamata, Unno, Sekiyama, Terauchi, POPL 2024) combines refinement types with algebraic effects and handlers.
- **Optimizers** remove bounds checks without a type-level trace (Rust/LLVM, the JVM).

What this ADR adds for Sounio is the combination. The language keeps a simple, flow-insensitive effect row for the programmer. An automatic, certificate-producing linear decider decides when `Panic` leaves that row. Lean checks each certificate independently. The programmer writes no proofs.

## Acceptance

1. `s.a[9]` and `s.a[-1]` above panic with index, length and span on both engines.
2. `fn f(a: &[i64; 8], i: i64) -> i64 { if i >= 0 && i < 8 { a[i as usize] } else { 0 } }` compiles **without** `Panic`, has no runtime check in the emitted code, and its certificate is accepted by Lean.
3. The same function without the guard is refused with the obligation `0 <= i < 8` and an empty path condition.
4. A certificate altered to drop a hypothesis is rejected by the Lean check and fails CI.
5. After the codemod, `stdlib/` and `self-hosted/` compile, the self-host fixed point holds, and the ratchet file records the codemod count.

## Implementation

Four cards under EPIC · Decisions owed: checked indexing on both engines (§1); obligations and path conditions in the IR (§3); the decision procedure, certificates and the Lean checker (§4); the codemod, §5 as an error, and the ratchet (§5–§6).
