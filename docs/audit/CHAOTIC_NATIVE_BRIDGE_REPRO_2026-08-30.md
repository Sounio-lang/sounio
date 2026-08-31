<!-- docs:meta
topic_id: repo.docs.audit.chaotic-native-bridge-repro-2026-08-30
authority: repo_only
audience: users
last_validated: 2026-08-30
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.chaotic-native-bridge-repro-2026-08-30
-->

# `with Chaotic` + a call kills the native-v2 bridge — two-line repro

**Date:** 2026-08-30
**Engine:** Madaros built from `origin/main` source that day
(`scripts/ci/build_modular_madaros.sh`, 101 746 924-byte ELF). Not the prebuilt
artifact.
**Why this matters:** with a current compiler, `scripts/ci/effect_archaeology_gate.sh`
has exactly **one** failure — `Chaotic chaotic_pass.sio`. It is the single thing
between that gate and being wired, and `.github/workflows/ci.yml` already names
it as the reason for leaving the gate out.

## The reproduction

```sounio
fn f() -> i64 with Chaotic { 7 }
fn main() -> i32 with Chaotic { f() as i32 }
```

    souc check    -> check: OK
    souc compile  -> rc=1, no ELF written
                     "error: native-v2 bridge compilation failed"
                     no diagnostic, no error code

It typechecks and then dies in codegen without saying why.

## Controls — every variable isolated

| variant | result |
|---|---|
| the two lines above | **rc=1** |
| identical, `Chaotic` → `Alloc` | rc=0, 8648-byte ELF, runs, exit 0 |
| identical, `Chaotic` → `Learn` (id 21, the neighbour) | rc=0 |
| identical, `Chaotic` → `NonUnitary` (id 28, whose seed bit is 22) | rc=0 |
| `f` declared `with Chaotic` but never called | rc=0 |
| `Chaotic` on `main` only, nothing called | rc=0 |

The archaeology fixtures make the same point at corpus scale:
`chaotic_pass.sio` and `alloc_pass.sio` are byte-identical apart from the effect
name, and `alloc`, `approx`, `witness`, `prob` and `temporal` all compile to an
8648-byte ELF and exit 0 while `chaotic` produces nothing.

So the trigger is **`Chaotic` specifically, and only when a function carrying it
is actually called** — not the effect id, not the neighbouring ids, not the
declaration.

## What `with Chaotic` is for

It is a deliberate semantic, not a stray name. `self-hosted/ir/lower.sio:2419`
routes it to `IR_STRATEGY_PRECISION_PRESERVING_CHAOTIC`, and
`self-hosted/ir/egraph.sio:1841` implements the point: `if ctx.chaotic { return 0 }`
— on a chaotic integration path (largest Lyapunov exponent λ > 0), float
reassociation is refused outright, because `e^{λt}` amplifies the reordering
error. `self-hosted/check/check.sio:5262` records it per call site:
`if has_effect_id(sig.effects, sig.effect_count, 22) { equiv_set_chaotic() }`.

## Suggestive, NOT established

`IR_STRATEGY_PRECISION_PRESERVING_CHAOTIC` (= 6) is referenced only under
`self-hosted/ir/` — egraph, ir, lower, opt_cleanup. It appears in neither
`self-hosted/compiler/main.sio` nor `self-hosted/native/lower_ir.sio`, while the
neighbouring `IR_STRATEGY_PRECISION_PRESERVING` appears in both.

That asymmetry is a lead, not a cause, and one comment argues against it:
`native/lower_ir.sio:659` says "IR_STRATEGY_STANDARD (0) / others: scalar SSE2,
same as PrecisionPreserving" — an unknown strategy is documented as falling back,
not failing. Someone who owns the native path should confirm or kill this.

An attempt to test whether the failure is bridge-specific — adding an import so
`main.sio:3357` declines the bridge — is **not** reported as evidence: the control
(same import with `Alloc`) also failed, so the import itself was the problem and
the experiment discriminates nothing.

## Symptom drift worth noting

`ci.yml` records the blocker as "Chaotic pass fixture dies at native ELF write
rc=12". Measured here it is `rc=1` at "native-v2 bridge compilation failed", with
no ELF produced at all. Same fixture, same conclusion for wiring purposes, but a
different stage and code — possibly a different invocation path rather than a
change in behaviour. Worth re-deriving before anyone treats the rc=12 detail as
current.
