<!-- docs:meta
topic_id: repo.docs.audit.fn-type-effect-clause-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.fn-type-effect-clause-census-2026-08-19
-->

# Function types and effect clauses — census and ratchet

## Answer

In live `.sio` source, **381** function types occur in parameter position.
**165 carry an effect clause; 216 do not.**

## Why this corrects an earlier number

An earlier reading in this session reported *"559 function types and not one
declares an effect"* and both halves were wrong.

- **559** was a raw line count over all `*.sio` including `archive/` and
  `bootstrap/` (66 lines) and including matches inside comments and string
  literals.
- **"not one declares an effect"** came from a pattern whose return-type
  character class excluded the shapes that actually occur. `filter8`
  (`benchmarks/sounio_bench/sb_003_higher_order_pipeline.sio:30`) refutes it in
  one line:

      fn filter8(arr: [i64; 8], pred: fn(i64) -> bool with Div, Panic) -> ...

  The parameter's type carries `with Div, Panic`. The syntax is supported and
  in use.

`self-hosted/parser/types.sio:717` documents the grammar directly:
`// Function type: fn(T, U) -> V with E1, E2`.

## What is *not* refuted

No **effect variable** occurs in live source. The single apparent instance,
`fn(T, U) -> V with E`, is that same comment. `SOUNIO-SPEC-06` §6.6's question
about abstracting over an argument's effects therefore stands unanswered by
practice.

## Instrument

`scripts/ci/fn_type_effect_ratchet_gate.sh`. Parameter position only, anchored on
the `:` that introduces the annotation; `//` comments and `"..."` literals
stripped before matching; `archive/`, `bootstrap/` and `*.sio.old` excluded.

Return position is deliberately **out of scope**: a function type in return
position shares its line with the enclosing declaration's own `with` clause, and
no line-local pattern separates them. An earlier revision of the gate counted the
outer clause as the type's and undercounted bare types by 29 (245 vs 216).

### Controls, both directions

| control | expectation |
|---|---|
| positive — a bare parameter function type | detected |
| negative 1 — a function *declaration* (`fn name(`) | not counted as a type |
| negative 2 — a function type inside a `//` comment | not counted |
| negative 3 — a parameter type that *does* carry effects | not counted as bare |
| negative 4 — a function type in *return* position | not counted |

The gate refuses to emit a number if its own controls fail.

## Ratchet

Frozen at **216**. A new bare parameter function type fails the gate; converting
one to carry effects passes and lowers the frozen count. This does not implement
`SOUNIO-SPEC-06` §6.0 — it stops the distance to that ruling from widening while
the ruling is unimplemented. Refusing all 216 today would refuse the repository;
refusing the 217th costs nothing.

## Claims forbidden

- That Sounio function types cannot carry effects. 165 of them do.
- That §6.0 is unimplemented in the surface syntax. What is unimplemented is
  *requiring* the clause, and abstracting over it.
