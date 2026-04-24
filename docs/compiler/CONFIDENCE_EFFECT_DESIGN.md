<!-- docs:meta
topic_id: repo.docs.compiler.confidence-effect-design
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.confidence-effect-design
-->

# Compile-time Confidence effect — design

## Status correction (2026-04-23)

**The previous version of this document (committed 55b4df38) was written as
if no compile-time confidence machinery existed.** That was wrong. A direct
audit of `self-hosted/compiler/lean_single.sio` — the file that `$SOUC`
actually delegates to — shows the core of Phase A and most of Phase B are
already shipped under the name **`Epistemic(N)`**, not `Confidence(N)`:

| Design-doc claim | Reality in lean_single.sio |
|---|---|
| Parse `Confidence(<float>)` as parameterised effect | `with Epistemic(N)` parses at lines 19991-20007; N is an integer on the same 1..1000 scale as `Knowledge<T>.confidence` (i16) |
| Per-function confidence floor | `FN_EPISTEMIC_MIN[16384]` at line 374 |
| Subsumption at call sites | `EXPR_CONF < FN_EPISTEMIC_MIN` diagnostic at lines 18370-18414 ("EpistemicComplete violation in fn X: uncertain token at pos Y (conf=A < B)") |
| Propagation through calls | `ety_conf_product`, `ESCOPE_CONF`, `EXPR_CONF` at lines 17085-17096 + ~60 call-sites |
| Cross-module via handlers/certs | `.econf` load+emit (Gen 15, lines 17318-17392) |
| Runtime case (3) — dynamic confidence | Gen 17 D4 runtime `update_conf` / `read_conf` builtins at line 10216 |
| Lean 4 proof obligation emission | Gen 17 D6 `--emit-proof-obligations` at line 376 |

The design doc was written against the modular `self-hosted/check/effects.sio`
tree (effects.sio, effects_row.sio, etc.), which is the Rust-era frontend and
is **not** the running compiler. Memory
[`feedback_lean_single_features`](../../../.home/openvscode-server/.claude/projects/-workspace-sounio/memory/feedback_lean_single_features.md)
flags this exact failure mode.

The rest of this document is rewritten against lean_single.sio ground truth.

## Why the effect exists

The runtime gates in
`stdlib/epistemic/confidence_gate.sio` and
`stdlib/darwin_pbpk/pd/pd_gate.sio` catch prior-quality violations when a
simulation runs. The compile-time gate catches them before a run is even
possible — which makes "prevent low-confidence data from contaminating
high-confidence computations" a type-level property.

## Surface (as implemented)

```sio
fn dose_from_plan(k: Knowledge<mg>) -> Dose with Epistemic(950) { ... }
```

Read as: "this function's body may not contain an expression whose
propagated confidence is below 950 (on the 1..1000 scale, i.e. 0.95)."
Without an argument, `with Epistemic` uses the global default
`GATE_THRESHOLD = 950` (line 318, lean_single.sio).

`Epistemic(N)` is the only parameterised effect label today. The parser in
`self-hosted/compiler/lean_single.sio` recognises the `(N)` suffix only for
`Epistemic`; all other effect tokens discard any parenthesised argument at
lines 7483-7496.

**Decision deferred to next session:** keep the name `Epistemic(N)`, or
introduce `Confidence(N)` as a parser alias for dissertation-code readability.
See Open Questions below.

## Typecheck rule (as implemented)

Walking the body of a function `f` with declared floor `m_f = FN_EPISTEMIC_MIN[f]`
(fallback: `GATE_THRESHOLD`), for each expression-like token `t` inside `f`:

- Compute `EXPR_CONF[t]` via `ety_conf_product` over its operands (GUM
  propagation + constructor confidence + `.econf` overrides).
- If `0 < EXPR_CONF[t] < m_f`, emit:
  ```
  error: EpistemicComplete violation in fn <name>: uncertain token at pos <p> (conf=<a> < <b>)
  ```
- At call sites, `EXPR_CONF[call] = ety_conf_product(arg_conf, FN_EFF_CONF[callee])`
  — so a caller's floor transitively rejects any callee whose returned
  confidence would fall below it.

Sources of confidence (all three of the design's tractability tiers are live):

1. **Literal / constructor**: `measured` ⇒ `FN_EFF_CONF = 990`; `asserted` ⇒ `970`
   (lines 17308-17316). Any user `Knowledge::new(v, var, conf)` propagates via
   `ety_alloc`.
2. **Struct priors**: `.econf` certificate loading (lines 17318-17392)
   overrides `FN_EFF_CONF[fi]` by function name; works across module
   boundaries without inlining.
3. **Dynamic runtime**: Gen 17 D4 `update_conf` / `read_conf` builtins at
   line 10216 allow Bayesian posterior updates that the runtime gate can
   then check at handler boundaries.

## What remains (actual gap list)

Phased, each phase independently mergeable. Sizes are estimates against
lean_single.sio (the running compiler), not the modular tree.

**Phase A′ — Float-literal surface (optional, ~40 lines in lean_single.sio)**
Today `Epistemic(950)` takes an integer. Accept `Epistemic(0.95)` and
auto-scale ×1000. Purely parser-side at lines 19995-20006. Low risk;
preserves bootstrap fixed point if the integer path is unchanged.

**Phase A″ — `Confidence(N)` alias (optional, ~15 lines)**
Add `Confidence` as a second keyword that routes to the same
`FN_EPISTEMIC_MIN` storage as `Epistemic`. Pure naming decision. The
dissertation's stdlib migration can read either way; picking one avoids
divergent prose.

**Phase B′ — Diagnostic polish (~60 lines)**
Current diagnostic at line 18388 reports `(conf=A < B)` with a token
position. For dissertation-grade error UX, extend to name the originating
`Knowledge<_>` argument and its declared confidence (e.g.
`argument 'hill.ec50_nM' carries confidence 0.400 < required 0.600`).
Requires walking `EXPR_CONF_SOURCE` back to the Knowledge constructor
site — infrastructure for this is partial at lines 18138-18178.

**Phase C — Stdlib migration (~100 lines of annotation, the dissertation-facing work)**
Annotate the PD public entry points with `with Epistemic(600)`
(or `Confidence(0.60)` if A″ lands):
- `dose_from_plan` — `stdlib/darwin_pbpk/pd/`
- `pd_endpoint_inhibition_auc`
- Any other function the dissertation cites as a "confidence gate"
Remove the runtime-gate wrapper in the hot path; keep it at module
boundaries as the bridge to dynamic-confidence case (3).

**Phase D — Test fixtures (~200 lines)**
- `tests/compile-fail/confidence_too_low.sio` — Knowledge literal at
  0.40 passed into a `Epistemic(600)` function. Current diagnostic text:
  `//@ error-pattern: EpistemicComplete violation`.
- `tests/run-pass/confidence_handler.sio` — weakened via explicit
  floor declaration on the caller.

## Bootstrap safety — do not break gen2==gen3

Any change to lean_single.sio must preserve the self-compilation fixed
point (md5=7b91e249). Safe patterns:

- Adding a branch to the effect-name parser (Phase A″) is invariant under
  self-compile — lean_single.sio itself uses no parameterised effects.
- Diagnostic text changes (Phase B′) are invariant.
- Float-literal handling (Phase A′) — verify lean_single.sio does not
  use `Epistemic(<float>)` on itself before merging.

Verify with the bootstrap chain after each phase:
```bash
./bin/souc self-hosted/compiler/lean_single.sio gen2.out
./gen2.out self-hosted/compiler/lean_single.sio gen3.out
md5sum gen2.out gen3.out  # must match
```

## Interactions with existing systems

- `Knowledge<T>`: confidence field is already `i16` (1000-scaled). The
  checker reads this directly — no new metadata.
- `gum.sio`: expanded-uncertainty ↔ confidence mapping is used by
  `ety_conf_product` already.
- Graded effects: `Epistemic(N)` is structurally a graded effect; the
  lattice is total order on `[0, 1000]`.
- `confidence_gate.sio` / `pd_gate.sio`: remain the runtime mechanism for
  case (3) and for worst-case prior-set audits the compile-time floor
  cannot express.

## What this does NOT try to do

- Not a dependent-type system — confidence is a numeric attribute, not a
  proof term.
- Not a Bayesian update system at compile time — runtime `update_conf`
  handles posterior updates.
- Not a replacement for runtime gates — Phase 3 cases cross the boundary
  at handlers, not at call sites.

## Open questions (for the implementation session)

1. **Name: `Epistemic(N)` vs `Confidence(N)`.** `Epistemic` is shipping and
   used throughout lean_single.sio and memory notes; `Confidence` reads
   more naturally in dissertation prose. Add alias (Phase A″) or retire
   the `Confidence` name from the design and update prose instead?
2. **Integer vs float surface.** `Epistemic(950)` is current; `Epistemic(0.95)`
   is more human. Accept both (Phase A′)?
3. **Diagnostic specificity.** Token position vs named argument — how much
   of `EXPR_CONF_SOURCE` is worth walking back for Phase B′?
4. **`Confidence(auto)` inference from body.** Out of scope for first pass;
   revisit once Phase C migration reveals whether explicit floors are
   ergonomic.

## References (verified, lean_single.sio line numbers)

- Line 318 — `GATE_THRESHOLD = 950` default floor
- Line 374 — `FN_EPISTEMIC_MIN[16384]` per-function floor storage
- Lines 7483-7496 — effect-list parser skipping parameterised args
- Lines 17266-17316 — Gen 16 Track B setup + Gen 12 A2 measured/asserted
- Lines 17318-17392 — Gen 15 `.econf` certificate loading
- Lines 18370-18414 — EpistemicComplete body enforcement + diagnostic
- Lines 19991-20007 — `Epistemic(N)` parameterised-effect parser
- Line 10216 — Gen 17 D4 runtime `update_conf` / `read_conf`
- `stdlib/epistemic/confidence_gate.sio` — runtime gate
- `stdlib/darwin_pbpk/pd/pd_gate.sio` — PD-specific runtime gate
- `stdlib/epistemic/knowledge.sio` — `Knowledge<T>` with i16 confidence
- `docs/compiler/EFFECT_SYSTEM_ARCHITECTURE.md` — current effect system
