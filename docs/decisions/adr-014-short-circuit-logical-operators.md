<!-- docs:meta
topic_id: repo.docs.decisions.adr-014-short-circuit-logical-operators
authority: repo_only
audience: users
last_validated: 2026-10-23
validated_by: tests/run-pass/logical_ops_short_circuit.sio
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-014-short-circuit-logical-operators
-->

# ADR-014: `&&` and `||` short-circuit

**Status**: accepted
**Date**: 2026-10-23

## Context

The two engines disagreed:

- **lean_single short-circuits.** `a && b` evaluates `b` only when `a` is true, and `a || b` evaluates `b` only when `a` is false.
- **Madaros evaluated both operands.** `lower_binary_expr_ref` lowered the left operand, then the right, and only then emitted the binop.

The specification recorded this as an open decision (EXEC-040, `docs/spec/2026-10/06-execution-memory-and-targets.md`).

The difference was not academic. Madaros's own source is written for short-circuiting, with guards such as

```sounio
n.len > 5 && n.buf[0] == 95 as i8
```

in `checker_name_is_destructure_temp`. In gen2, the Madaros-compiled Madaros, that guard read `n.buf` even when the length check had already failed. It trapped on a null array reference for any program with a `let` (#2966).

Measured on 2026-10-23 with `tests/run-pass/logical_ops_short_circuit.sio`:

| engine | call counter after the three checks | prints |
|---|---|---|
| lean_single | 1, 2, 4 | `SHORT_CIRCUIT_OK` |
| Madaros before this change | 2, 4, 6 | `101010` |

## Decision

`&&` and `||` short-circuit on every engine:

1. `a && b` evaluates `a`. If `a` is false, the result is false and `b` is not evaluated.
2. `a || b` evaluates `a`. If `a` is true, the result is true and `b` is not evaluated.
3. Evaluation is left to right, so a side effect in `a` happens before any side effect in `b`.

The founder chose this over two alternatives:
- **Documented eager evaluation.** This would have required rewriting every guard in the compiler as a nested `if`.
- **Short-circuiting only when the right operand could trap.** This rule would be harder to state and to teach.

## Implementation

**Madaros.** `lower_short_circuit_ref` in `self-hosted/ir/lower.sio` handles both operators, and `lower_binary_expr_ref` routes `&&` and `||` to it before any other operator handling. It emits:

```
res = lhs
&& : if !lhs goto end       || : if lhs goto end
res = rhs
end:
```

The result is a bool in `res`, as before.

**lean_single.** Already short-circuits. Unchanged.

## Consequences

- A guard such as `i < len && a[i] == x` is safe on both engines.
- Code that relied on Madaros evaluating the right operand for its side effect changes behaviour. None was found in the repository: the Madaros source assumed short-circuiting throughout.
- EXEC-040 is closed for `&&` and `||`. The rest of EXEC-040 stays open: compound lvalues, builtin special cases and kernel side-effect order.

## Grounded in

- `tests/run-pass/logical_ops_short_circuit.sio`. It runs in the Full Test Suite under lean_single.
- `scripts/ci/madaros_short_circuit_gate.sh` runs the same file under the current-source Madaros in the gen2-proof workflow. The Madaros built from main fails it with `101010`.
