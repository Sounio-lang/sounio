# Observer surface survey — where producer identity lives and dies in the stack

**Lineage:** butterfly #1 ("o observador no sistema de tipos") from the
2026-08-30 kimi-cli2 conversation; sibling work to the Pharos loss census
(`madaros_pass_loss_census_20260830.md`). This survey is descriptive: it maps
what exists. The normative question (should values carry *who* observed them)
remains a Garden-level butterfly.

**Evidence label:** `implemented` as a document; all findings are **static
readings of source**, nothing compiled or executed. Reading surface:
`/workspace/sounio` @ `lane/cursor-1/20260826` (`d031627a4fa6`).

## The one-paragraph answer

Observer/producer identity **never reaches the language's value semantics**.
It exists in two disjoint places: the *process layer* (coordination claims,
sealed research receipts, offload logs — hash-bound, enforced) and a few
*leaf stdlib structs* (`DataSource`, `SLSAProvenance`, epistemic-core
`Source`) that are plain library data, disconnected from the `Knowledge<T>`
machinery. Between those two layers, the compiler has **three dead sockets**:
slots built for exactly this information, wired to nothing.

## The three dead sockets

1. **`Provenance.source_id` — the socket that is always zero.**
   `self-hosted/check/epistemic.sio:225-228` defines
   `Provenance {kind: i64, source_id: i64}`. The constructor
   `provenance_new` always sets `source_id = 0` (:231-237);
   `provenance_with_source` (:240-245) — the constructor that would populate
   it — **has no call sites anywhere in the tree**.

2. **`ContestInfo.provenance_bundle_id` — the socket that is always −1.**
   `self-hosted/check/defs.sio:2364-2378` reserves the field; it is
   initialized to −1 (:2394) and only ever copied
   (`check/check.sio:2088, 8561, 20333`), never populated.

3. **`ir_measure` — the opcode that is never emitted.**
   The IR can represent measurement: `IrMeasure` (ir.sio:388) with constructor
   `ir_measure(dst, value_reg, knowledge_meta_id)` (ir.sio:3130-3151, comment:
   "uncertainty is type-level only"). The constructor **has no callers** —
   the opcode exists in the IR alphabet and is never written into a program.
   HLIR lowering of `measure(v, u)` (hlir/lower.sio:2579-2585) returns the
   value and drops the uncertainty *before* the IR; the checker path
   (check.sio:22180-22188) type-checks only the value argument.

## The stack-layer map

| Layer | What exists for observer identity | Fate |
| --- | --- | --- |
| Source type `Knowledge[T, ε, δ, Φ]` | `provenance` field (ast.sio:496), a bare **category** enum `AstProvenanceKind`: Derived/Source/Computed/Literature/Measured/Input (ast.sio:452-459) — no payload naming who/what | Categories parse; probes note the annotation is "silent" |
| Checker | `KnowledgeMeta {epsilon, validity, provenance}` (epistemic.sio:480-484); dead `source_id` slot; EpistemicComplete gate seeds confidence **by constructor kind only** (`measured()`→990, `asserted()`→970, `constant()`→1000, lean_single.sio:25896-25898) — no identity | `check_knowledge_type` validates then **drops** validity/provenance before `TypeEntry` (epistemic.sio:49: "TypeEntry does not yet persist full validity/provenance metadata"); `TypeEntry` keeps only `knowledge_epsilon` (types.sio:139-156) |
| Temporal validity | `ValidUntil("2020-03-31")` parses (ast.sio:441-450; parser/types.sio:992-1136) and is stored transiently | `validity_is_expired` (epistemic.sio:432-440) **never called**; `knowledge_meta_from_ty` rebuilds every meta as always-valid/DERIVED (:497-530) |
| IR | `IrMeasure` never emitted; certificate opcodes carry span-keyed internal table ids (ir.sio:2840-2901, 3093-3114), e.g. `find_recourse_plan_id_for_span` (lower.sio:16245) | Table ids are positions, not identities |
| Codegen | — | Every epistemic opcode lowers to `MOV dst, value` (native/lower_ir.sio:281-298); below the IR, values are provably identity-free |
| Effects | `Observe` effect (check/effects.sio:107-115), explicitly "von Foerster's observer-inclusion effect"; `Unobserved<T>` type-state (types.sio:119-120) | A capability flag and a state — observer *presence*, never observer *identity* |
| Stdlib leaves | `DataSource {trust_level, source_id, citation_count}` (stdlib/web/epistemic_http.sio:4-8); `SLSAProvenance {builder_id_hash, …}` (stdlib/epistemic/slsa.sio:160-198); `Source {label, method, reliability}` (packages/epistemic-core/src/lib.sio:26-31) | Plain structs, unbound to `Knowledge<T>`; and `source_from_str` (:39-47) **ignores the label** and hardcodes method=SENSOR |
| Process layer | coord claims carry agent/lane; Pireus receipts carry producer_language/role, guardian and hardware hashes; offload log carries reviewer identity | Hash-bound and enforced — the only layer where "who produced this" is load-bearing |

## The von Foerster detail

The codebase already contains the philosophy's name: the `Observe` effect is
documented as *"von Foerster's observer-inclusion effect"* — second-order
cybernetics, the observer as part of the observed system. But it is a
capability flag: a function declares that it observes, never *who* is
observing. The observer is present as a door; the three dead sockets are the
unwired hinges.

## What this survey is not

- Not a claim that missing observer identity has caused a wrong result.
- Not a design for `Measured<T> by O`; the butterfly stays un-promoted.
- Not runtime evidence (static reading only).
- Not a criticism of the category enum — categories are a defensible first
  step; this maps where they stop.

## Next bridge (one)

A **silence witness**, in the spirit of the Pharos planted crime: a probe
program declaring two values, `Knowledge[f64, Source]` and
`Knowledge[f64, Literature]`, and a check that their `TypeEntry` after
checking is *observationally identical*. If they are indistinguishable, the
erasure is witnessed with a reproducer; if they are distinguishable, this
survey is wrong about the checker and gets a correction. Either outcome is
data. (Blocked today by the fleet-wide build-lock deadlock; the witness needs
a from-source compiler run, not the committed binary.)
