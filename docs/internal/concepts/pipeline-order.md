<!-- docs:meta
topic_id: repo.docs.internal.concepts.pipeline-order
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pipeline-order
-->

# Pipeline Order

Concept-ID: `SOUNIO-PIPELINE-ORDER`

Status: **Hypothesis** — two founder rulings of 2026-08-19. Their apparent
tension was **resolved on 2026-08-19 by claude-1 under founder delegation** (see
"Resolution of T1" below); the founder may overrule. Neither ruling is
implemented.

## The two rulings

**HLIR is always on the path.** It stops being the GPU frontend and becomes the
layer everything descends through; the backends hang below it.

```
check → HLIR → ENIR → verify → native      mandatory route
                    ↘ ir/ + e-graph        outside the trusted base, validated per run
             ↘ gpu                         route under T1-R below
```

## Resolution of T1 — the two rulings compose; the earlier diagram did not

The corpus cross-check of 2026-08-19 recorded a contradiction between
`pipeline-order.md` ("HLIR is always on the path") and `verified-lowering.md`
("ENIR becomes the only path"), and left it `INDETERMINATE`. Resolved as follows.

**The two statements are about different things.** "HLIR is always on the path"
is a claim about a **layer** — what everything descends through.
"ENIR is the only path" is a claim about a **route**, and `verified-lowering.md`
qualifies it in the same sentence: *"`ir/` and the e-graph are an optional
accelerator, never mandatory"*. An optional accelerator sits **below** the
mandatory route, not beside it.

What contradicted was this document's **diagram**, which drew
`HLIR → ir/ → native` as the CPU spine. That is the tree as measured today, not
the ruling. The diagram yields and has been replaced above.

**Measured state at resolution** (`origin/main`, 2026-08-19; instrument
`^use <dir>::`, positive control `check/` → `parser::` in 6 files):

| directory | imports |
|---|---|
| `enir/` | only itself (10). Imports neither `hlir::` nor `ir::`, and nothing imports it — an **island** |
| `ir/` | only itself (21) |
| `native/` | `ir::` (14) |
| `gpu/` | `hlir::` (1 file) |

So the live tree holds three disconnected pieces: the spine `ir/ → native/`, the
pair `hlir → gpu` with a single importer, and `enir/` connected to nothing. Both
rulings describe a tree that does not exist, which is why neither could be
falsified by the other.

### T1-R — the residue this resolution makes visible

Where **GPU** sits is *not* decided by either ruling and is not decided here.
`pipeline-order.md` hangs `gpu` directly below HLIR; `verified-lowering.md` says
ENIR is the only path. If "only" includes GPU, then `gpu/` descends through ENIR
too, and translation validation applies to PTX emission. If it does not, GPU is
a second route below HLIR and the word "only" needs narrowing to CPU lowering.

Owed to the founder. Recorded as newly visible rather than closed.

**Verification is the floor** (`SOUNIO-VERIFIED-LOWERING`): ENIR becomes the
path for content that must be trusted; `ir/` and the e-graph are an optional
accelerator, and a rewrite rule may not run until it carries translation
validation.

## Measured state, 2026-08-19 (`origin/main`)

The live CPU pipeline is `parser → check → ir → native`. HLIR is **not** off the
tree and **not** uncalled — it is the **GPU frontend**: `hlir_lower_module` is
invoked at `self-hosted/compiler/main.sio:28198`, inside the `--backend gpu`
path, and at `self-hosted/main.sio:502`. Default `souc` never reaches it.

That explains where the hypercomplex algebra lives, and it is not an accident.
`HlirTypeOctonion` (`hlir/ir.sio:135`) lowers to `<8 x float>` — one vector
register for eight components, which is what a SIMD unit wants. The algebra sits
in the GPU frontend because it was designed to descend through GPU.

**Consequence: the ruling is a reordering, not new construction.** Making HLIR
always-on does not require building it.

Cost of making a bare `Octonion` annotation sayable, counted in
`docs/audit/HLIR_DISCONNECT_COST_2026-08-19.md` (PR #1957): **two** new enum
variants (checker `TypeKind`, layout `LayTypeKind`), three if the parser also
gets its own form rather than reusing `TypeNamed`, plus **one** absent function
`TypeKind → HlirTypeKind`. The parameterised spelling `Hyper<Octonion, f64>`
needs **zero** parser and checker variants — `TyHyper` (`check/types.sio`,
tag 22) already carries `Hyper<Algebra, T>`, and `hlir_type_from_ast` already
returns `hlir_type_octonion()`.

## The unresolved tension

Putting HLIR on the path solves **rich types**. It does **not** solve
**uncertainty**, and the two problems pass through the same place.

`HlirTypeKind` has `HlirTypeKnowledge` (`hlir/ir.sio:149`) — so HLIR can *name*
an epistemic value. But variance is not part of that value: it lives in slots
bound by hand in `ir/lower.sio` (`alloc_variance_slot_for_local`,
`bind_variance_for_base_reg`, `emit_variance_add`, …), which is **below and
beside** HLIR. See `SOUNIO-VERIFIED-LOWERING` for why that representation is the
mechanism behind the FO variance defect.

So if everything descends through HLIR and epistemic content then descends
through ENIR, **HLIR must be able to carry uncertainty in order to hand it
over** — and today it cannot. Naming a `Knowledge` is not carrying its variance.

**This is recorded as an open question, not a plan.** Three shapes exist and
none is chosen:

1. HLIR carries uncertainty in the value (the representation
   `SOUNIO-VERIFIED-LOWERING` argues for), and ENIR receives it whole.
2. HLIR carries only the type, and the ENIR boundary reconstructs variance from
   provenance — which requires `SOUNIO-PROVENANCE` first.
3. Epistemic content leaves the HLIR path earlier, at the checker, and HLIR is
   always-on only for non-epistemic code — which weakens "always on the path"
   to "always on the ordinary path".

## Required Invariants

- A type is sayable in exactly one place. If `Octonion` acquires a variant in
  the checker while a second hypercomplex representation persists elsewhere,
  two enums must agree forever, and forever is when they stop agreeing.
- An unknown type name is refused, never defaulted. Today the bare name
  `"Octonion"` falls back to `hlir_type_i64()` in `hlir_type_from_ast`. Adding a
  parser variant without closing that fallback makes an octonion compile
  silently as a 64-bit integer — eight components reduced to one, no
  diagnostic. This must refuse before any variant is added.
- A downstream lowering is not evidence of a sayable type.
  `llvm/type_convert.sio:282` already lowers `HlirTypeOctonion`, and `llvm/`
  has **zero** external importers. Someone wrote the end of the road before the
  beginning; that end proves nothing about the beginning.
- Reordering is not construction, and neither is it free. HLIR handling every
  compile means HLIR meets inputs the GPU path never gave it.

## Claims Forbidden

- Do not describe HLIR as on the default path. It is the GPU frontend and
  default `souc` does not reach it.
- Do not present the two rulings as a resolved architecture. Their interaction
  is the open question above, stated as three unchosen shapes.
- Do not read `HlirTypeKnowledge` as uncertainty support. It names an epistemic
  value; variance lives in hand-bound slots in `ir/lower.sio`.
- Do not cite the `Hyper<Octonion, f64>` path as working. It fires only inside
  `hlir_lower_module`, which default `souc` never calls.
- No schedule attaches to either ruling.

## Related

- `SOUNIO-VERIFIED-LOWERING` — the other half of the ordering, and why
  uncertainty must be carried rather than bound
- `SOUNIO-EPISTEMIC-ERASURE` — what the pipeline must not silently drop
- `SOUNIO-PROVENANCE` — required first if shape 2 is chosen
