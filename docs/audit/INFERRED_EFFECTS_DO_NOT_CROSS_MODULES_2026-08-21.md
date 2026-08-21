<!-- docs:meta
topic_id: repo.docs.audit.inferred-effects-do-not-cross-modules-2026-08-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.inferred-effects-do-not-cross-modules-2026-08-21
-->

# An inferred effect does not cross a module boundary

**Status:** open defect, reproduced with a single-variable isolation.
**Engine:** Madaros (`bin/souc`, v0.80.0 prebuilt). Not yet checked on lean_single.
**Date:** 2026-08-21.

## Summary

With `SOUNIO_EFFECT_INFER=1`, a function's effect row is widened from the rows
of the functions it calls. That widening **reaches importers only when the
callee's effect was written by hand.** When the callee's own row was itself
inferred, the effect stops at the module boundary: importing modules gain
nothing, no diagnostic is emitted, and the compiler exits 0.

The obligation is not refused. It disappears.

## Reproduction

Two trees, byte-identical except for one clause. Both checked with:

```bash
SOUNIO_EFFECT_INFER=1 SOUNIO_EFFECT_INFER_TRACE=1 ./bin/souc check main.sio
```

`deep.sio` (same in both):

```sounio
pub fn deep_panics(x: i64) -> i64 with Panic {
    if x < 0 { panic("neg") }
    x
}
```

`main.sio` (same in both):

```sounio
use mid::{mid_calls}
fn top(x: i64) -> i64 { mid_calls(x) }
fn main() -> i64 with IO {
    println("five")
    top(1)
}
```

`mid.sio` — **the only difference**:

| case | `mid.sio` | trace | rc |
|---|---|---|---|
| A | `pub fn mid_calls(x: i64) -> i64 with Panic { deep_panics(x) }` | `top += Panic (from mid_calls)`<br>`main += Panic (from top)` | 0 |
| B | `pub fn mid_calls(x: i64) -> i64 { deep_panics(x) }` | `mid_calls += Panic (from deep_panics)`<br>*(and nothing further)* | 0 |

In case B, `top` and `main` call a function that panics and neither carries
`Panic`. With inference **off**, the same shape is refused with `error[E035]`.

## Why it matters

This is self-defeating in the exact proportion that inference works. Inference
exists to remove hand-written effect clauses; every clause it removes becomes a
boundary that the next importer cannot see through. Turning inference on by
default today would therefore not "propagate more" — at module boundaries it
would propagate **less**, and silently.

It also bears on the founder's ruling of 2026-08-21 that a written effect
declaration should become a **claim refutable by the body**. Refutation needs
the inferred row to be trustworthy across the whole program, not only inside
one module.

## What is confirmed and what is not

- **Confirmed:** same-file chains propagate correctly (five levels, traced at
  every step); two-module chains propagate when the callee declares; the A/B
  pair above isolates the difference to a single clause.
- **Not established:** the mechanism. A second agent (gpt-5.6-sol, xhigh)
  predicted an ordering hazard — "a caller checked before the callee's body may
  observe the signature not yet widened" — and this is consistent with that,
  but the module boundary rather than textual order is what the A/B pair varies.
  The widening path is `fn_sig_table_widen_effects`
  (`self-hosted/check/defs.sio:1431`), reached from
  `self-hosted/check/check.sio:8085`.
- **Related and separately confirmed:** that same widener drops an effect in
  silence once a row already holds eight
  (`defs.sio:1454`, `if !seen && ... effect_count < 8`). Two independent silent
  losses on the same path.

## Reproduction files

Not checked in — they are nine lines across three files and are reproduced
verbatim above. Build them in a scratch directory and run the command shown.
