<!-- docs:meta
topic_id: repo.docs.decisions.adr-032-epistemic-typed-core
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-032-epistemic-typed-core
-->

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
3. **Interprocedural sets by summary.** A function's SC signature includes a source summary: which parameter sets flow to the result, and which sources the body creates. A call site applies the summary. The set lattice and the instantiation rule are part of the contract, not of the encoding:
   - **Three states, not two.** `Known(∅)` (exact: no source), `Known(S)` with S non-empty, and `Unknown` (top). `Unknown` is never `Known(∅)`: it never satisfies a precise consumer (E230/E314 refuse it exactly as they refuse a non-empty K). Recursion and any summary the elaborator cannot compute are fail closed to `Unknown`.
   - **Instantiation.** A summary names three kinds of source: (i) *forwarded* parameter sets, replaced by the caller's argument sets, so repeated identity calls preserve one shared source; (ii) *named global* sources (a Knightian `⊥` with a declared name), shared by every call and every module, so the same name with the same bounds agrees and conflicting bounds refuse (E314); (iii) *fresh* sources the body creates, which get a new identity **per call**, so two calls never produce residues that cancel.
   - **Canonical identity.** A source id is canonical (declaring module path plus declaration name, or call-site path plus allocation site for fresh ones), not a module-local counter. Serialised modules carry these names, and a deterministic remap rebuilds ids on load, so reordering imports changes neither identity nor results.
   - **Projection.** A field access on an aggregate yields the set of that field, not the union over the aggregate. Construction records one set per field. This rule is stated before N-04/N-11 are claimed closed.

   The semantics already fixed by #3741 (`knight_set_id`, E314, its `epi006_knight_*` witnesses) is the reference; SC encodes it and does not compete with it.
4. **Core Lint.** An independent SC re-checker runs after elaboration and after every SC→SC pass. Precedent: GHC Core Lint.
   - It is written separately from the checker, by the diversity rule of the project instructions.
   - A disagreement between checker and lint fails the build and is logged as an anomaly.
   - **The lint checks well-formedness, not meaning.** It verifies types, source sets, effects and field indices. It is not a translation validator: a pass that rewrites `x + 1` to `x + 2` keeps every type and set and passes the lint. Semantic preservation is a separate obligation: a stated refinement relation between the pre- and post-pass terms (observable values, refusals, provenance and the relevant effects), proved in Lean for the fragment or checked per pass by a validator built for that relation. Discriminant: a type-correct, value-changing rewrite passes the lint and must fail any claimed semantic validator.
5. **Lean semantics.** `formal/lean4/core/` states SC's syntax, typing (including the set rules) and a semantics, first for a fragment. The target theorems:
   - (a) type soundness of SC;
   - (b) the E230/E314 refusal rules are sound with respect to the semantics: a precise bound is never given to a value with a non-empty Knightian set;
   - (c) the lowering preserves types for the fragment (type preservation);
   - (d) a separately stated theorem, semantic refinement, for the passes it covers: every observable behaviour of the output (values, refusals, provenance) is one of the input's. (c) does not imply (d), and no pass is called semantics-preserving on the strength of (c) or of Core Lint.

   Each theorem states the fragment it covers.

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

1. **This ADR + an SC grammar and typing rules document**, plus a Lean fragment covering literals, `let`, arithmetic, Knowledge with A/K sets, and calls with summaries. Step-1 obligations, each a witness program with its expected verdict written before the run (reuse #3741's witnesses where they exist):
   - repeated identity calls preserve one shared source;
   - two calls that create fresh residues do not cancel;
   - an imported named Knightian source with the same bounds agrees; with conflicting bounds it refuses (E314);
   - an `Unknown` result (recursion) does not satisfy a precise requirement;
   - reordering imports leaves every source identity and every result unchanged;
   - a field projection carries the field's set, not the aggregate's;
   - a deliberately type-correct, value-changing rewrite passes Core Lint (it is a well-formedness check) and fails the semantic-refinement check.
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

## Decisions taken (2026-11-25, by the founder)

1. **SC is a graph with explicit effect edges** (Bračevac-style), not a tree. Effects and source sets are edges, so Core Lint and the Lean semantics range over the same structure the lowerer reads.
2. **Core Lint is written in Sounio, with an independent OCaml twin.** By the diversity rule a Sounio lint is compiled by the thing it measures; the receipt is the agreement of both, and a disagreement is logged as an anomaly before either is called wrong.
3. **Theorems (a) type soundness and (b) soundness of the E230/E314 refusals are closed before step 3.** (c) type preservation and (d) semantic refinement proceed in parallel with step 3. `#print axioms` on each.
