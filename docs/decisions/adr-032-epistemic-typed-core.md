# ADR-032: Sounio Core — an explicitly typed, epistemic core between the checker and the lowerer

- Status: **Proposed** (2026-11-25)
- Deciders: Demetrios Chiuratto Agourakis
- Supersedes: nothing. Interacts with ADR-016 (epistemic merge and provenance), ADR-017 (IR memory effects, draft), ADR-026 (phase arenas), ADR-030 (generation chain).

## Context

Madaros lowers the **untyped AST** straight to flat IR (`self-hosted/ir/lower.sio`, 53,859 lines). The checker (`self-hosted/check/check.sio`, 35,897 lines) computes types and then throws them away. The lowerer must rediscover them.

Measured on `main` 8178ea5070:

| What the lowerer re-derives | Measure |
|---|---|
| Expression-type predicates (`fn (opt_)expr_(result_)is_*_ref`) | 25 functions, 112 call sites |
| `lookup_local_struct_type` lookups | 72 |
| Struct field index | majority vote across every struct declaring the name (`lower_field_idx_majority`, `lower_field_idx_untyped`); E311 and `SOUNIO_STRICT_FIELD` exist to refuse the guesses |
| Checker → lowerer channel | `IrEpistemicSection.hyper_exprs`, keyed by **span**, 128 entries |

Defects of the last two days, all traced to this one gap:
- **#3739:** the span table does not reach the lowerer on the multi-module path. `x * 2.0` SIGSEGVed in any program that imports a module.
- **#3735:** an untyped `let k = Knowledge(..)` read the hash slot. The field vote tied between Knowledge's layout and a user struct.
- **Knowledge (x) Knowledge:** operands are classified syntactically, and a span-keyed backstop catches the misses.
- **Source identity (noise sets, Knightian sets):** dropped at calls and fields, because it lives in side tables and not in a type the lowerer consumes (ledger N-04, N-11).
- **Globals:** bound as unknown inside functions until fix/madaros-global-types-in-fns. That fix found 33 latent enum-into-i64 stores in the compiler itself.

`Expr` has no node identity, and nodes are passed by value, even on the in-place checker path. An annotation written on a node by the checker does not survive.

## Decision

Introduce **Sounio Core (SC)**. SC is an explicitly typed, epistemic core language. The checker **elaborates** the AST into SC. The lowerer consumes SC only and never re-derives a type.

1. **Every SC node carries its full type.** This is not an inferred hint; it is part of the node. The type is epistemic:
   `τ ::= base | struct S | enum E | array τ n | ref/excl τ | fn τ̄ → τ ! ε | Knowledge[τ; A; K; m] | unit-branded τ @ u | …`
   - A is the aleatory source set (today's `noise_set_id`).
   - K is the Knightian source set.
   - m is the ε mode (error bound / exact confidence / floor).
   - ε is the effect row.

   Source identity becomes a property of the type of each node, not a side table. Field access carries the resolved field index, and calls carry the resolved callee.
2. **Node identity.** The parser gives every expression a stable id (module index, local id). SC is serialised with each module's IR, so the multi-module path carries it by construction.
3. **Interprocedural sets by summary.** A function's SC signature includes a source summary: which parameter sets flow to the result, and which sources the body creates. A call site applies the summary. Recursion is fail closed: the set becomes unknown.
4. **Core Lint.** An independent SC re-checker runs after elaboration and after every SC→SC pass. Precedent: GHC Core Lint.
   - It is written separately from the checker, by the diversity rule of the project instructions.
   - A disagreement between checker and lint fails the build and is logged as an anomaly.
   - The lint works as a translation validator for each pass.
5. **Lean semantics.** `formal/lean4/core/` states SC's syntax, typing (including the set rules) and a semantics, first for a fragment. The target theorems:
   - (a) type soundness of SC;
   - (b) the E230/E314 refusal rules are sound with respect to the semantics: a precise bound is never given to a value with a non-empty Knightian set;
   - (c) the lowering preserves types for the fragment.

   The axioms are checked with `#print axioms`.

## Prior art and delta

- **GHC Core + Core Lint:** a typed core with an independent linter between passes. It has no epistemic types.
- **rustc HIR/THIR, Swift SIL, Lean elaborated terms:** typed elaboration before codegen. No uncertainty or provenance in the types.
- **Type-preserving compilation:** Morrisett et al. (TIL/TAL); Bowman et al. (e.g. "ANF preserves dependent types up to extensional equality", JFP 2022). Types are carried through compilation; the types are not epistemic.
- **Bračevac et al., graph IRs with precise effect dependencies.** Effects are in the IR, not measurement sources.
- **CompCert, CakeML:** verified compilers. Their correctness is with respect to a semantics without measurement uncertainty.

Delta, to be attacked before any claim (novelty rule 2; recorded as CANDIDATE):

> No compiler is known to carry measurement-uncertainty source sets, aleatory and Knightian, in the type of every node of its typed core, re-checked by an independent lint after every pass, with the refusal rules proved sound in Lean.

Classification: a new combination. Only-Sounio dependencies: epistemic types + Lean-checked semantics (+ the bootstrap, since SC also elaborates the compiler itself).

## Migration (each step has a prediction and an exit criterion)

1. **This ADR + an SC grammar and typing rules document**, plus a Lean fragment covering literals, `let`, arithmetic, Knowledge with A/K sets, and calls with summaries.
2. **Node ids in the parser; elaboration to SC alongside today's path.** No consumer yet. Exit: a gen2 fixed point holds, and the elaborated SC of the corpus passes Core Lint.
3. **The lowerer reads types from SC, one construct at a time:**
   - fields (retires the field vote and E311 guesses);
   - Knowledge arithmetic (retires syntactic operand classification and `hyper_exprs` op_kind 21/22);
   - calls (summaries; closes N-04's call-boundary loss);
   - globals and literals.

   Prediction for each construct: its re-typing predicates reach 0 call sites, with corpus results unchanged except the defects it fixes.
4. **Retire** the AST→IR path and the span-keyed channel.

Baseline to beat: 25 predicates / 112 call sites / 72 struct-type lookups / the field-vote functions, all to 0.

## Consequences

- One source of truth for types, from the checker onward; the guessing machinery is deleted rather than hardened.
- Source identity crosses calls and fields by construction, which unblocks N-04 and strengthens N-11.
- Cost: a new serialised artifact per module, and a second checker (the lint) to maintain. Both must enter ADR-026 phase arenas, and the gen2 memory budget (~56 GB today) is a constraint to measure, not assume.
- The lean_single seed does not get SC; its divergences are frozen by name (decision of 2026-11-25).

## Decisions left open

1. Whether SC is a tree (by-value nodes with ids) or a graph IR with explicit effect edges (Bračevac-style).
2. The lint's implementation language: Sounio, compiled by the compiler under test, vs a twin in an admitted non-Sounio language (diversity rule).
3. How much of the Lean fragment precedes step 3.
