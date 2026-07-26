# Visibility widening is not semantics-preserving under filtered-fallback resolution

**Date:** 2026-07-26 · **Surface:** Madaros (`self-hosted/`) · **Status:** finding evidenced by
experiment; mitigation proposed, not yet implemented

## Summary

In a compiler whose symbol resolution consults visibility as a **filter inside a fallback
chain** — rather than resolving first and validating access afterwards — adding `pub` to a
definition is **not semantics-preserving**. It can change which definition an unrelated call
site resolves to. Where the competing definitions share a parameter type sequence the change
typechecks, and where their parameters are permuted the emitted code is silently wrong.

This was found while auditing why Madaros cannot compile its own `main.sio`, and it is the
reason a 1,276-site `pub`-widening campaign was **not** carried out. A computable detector
bounds the hazard to 2 named functions in this codebase; everything else is provably safe.

## The mechanism

Madaros resolves a free-function call through three tiers
(`self-hosted/check/defs.sio:1440-1473`, `fn_sig_table_find_prefer_module`):

| tier | rule |
|---|---|
| T1 | a free fn whose `defining_module_id` equals the caller's module — preferred |
| T2 | **any** fn whose `visibility_kind != VISIBILITY_PRIVATE` — visibility used as a *filter* |
| T3 | `fn_sig_table_find(t, name)` — first match in the merged table, private or not |

The table is merged across the whole closure (105 modules for `compiler/main.sio`), so T2 and
T3 are both "first match anywhere", differing only in the visibility filter.

**Claim.** Let `d` be a definition of name `n` with private visibility, and `widen(d)` make it
public. There exists a call site `c` such that `resolve(n, c)` differs before and after
`widen(d)`, even when `c` does not reference `d`'s module.

**Argument.** Before widening, if T1 misses and every candidate is private, T2 misses and T3
returns the first in-table match. After widening, T2 matches `d`, and T2 is consulted before
T3. Whenever `d` is not T3's pick, the resolved target changes. ∎

**Corollary (silence).** The change is loud only if the two candidates differ observably in
signature. If their parameter **type sequences** are identical the flipped call typechecks; if
their parameters are additionally **permuted**, the emitted code is wrong with no diagnostic.

## Experimental witness

Fixtures (four modules, reproduced in both directions on `madaros_fixed9.elf`):

```sounio
// good.sio      fn f(a: i64, b: i64) -> i64      private, arity 2
// bad.sio       fn f() -> i64                    private → pub, arity 0
// caller.sio    use good::*  use bad::*          pub fn go() -> i64 { f(10, 20) }
// hjmain.sio    use caller::{go}                 fn main() -> i64 { go() }
```

| state of `bad::f` | diagnostic | resolved target |
|---|---|---|
| private | `error[E175]` | `good::f`, arity 2 — correctly typed |
| **`pub`** | `error[E010]` wrong number of arguments | `bad::f`, arity 0 — call now ill-typed |
| private again | `error[E175]` | back to `good::f` |

One `pub` changed the target of a call in which the widened function never participated.

## The silent instance in this tree

```
self-hosted/compiler/main.sio:2844      fn ir_call_sret(dst, dest_reg, fn_name, fn_id,    arg0,  n)
self-hosted/ir/ir.sio:3444          pub fn ir_call_sret(dst, fn_id,    fn_name, dest_reg, first, n)
```

Identical type sequence `(i64, i64, Name, i64, i64, i64)`; positions 2 and 4 **swapped**. A
resolution flip here typechecks cleanly and transposes a destination register with a function
id in emitted IR. Today `main.sio`'s calls are protected only by T1 matching on
`defining_module_id`; that protection is one table-order perturbation away.

## Detector

`scripts/research/visibility_resolution_monotonicity.py` computes the exact set of names for
which any visibility change is potentially resolution-changing: names defined in more than one
file with unifiable parameter type sequences. Re-run with:

```bash
python3 scripts/research/visibility_resolution_monotonicity.py
```

Result on `compiler/main.sio`'s closure at commit `148179502`:

```
closure files: 105   distinct fn names: 8759   homonyms: 7
  SILENT flip risk (cross-file, identical type sequence): 3
      ir_call_sret     main.sio:2844   +   pub ir/ir.sio:3444
      ir_return_sret   main.sio:2823   +   pub ir/ir.sio:3467
      main             main.sio:28080  +   gpu/cuda_tile.sio:282   (entry points)
  LOUD flip (differing type sequences): 0
  same-file duplicates ('which definition wins'): 4
      compiler_mode_positional_arg  main.sio:1293, 1339      (parameters renamed between them)
      ir_wide_add                   ir/ir.sio:3305, 3524     (both pub, identical signatures)
      nc_emit_seta_al               codegen_x86_linux.sio:1990, 2251
      nc_emit_xor_rax_rax           codegen_x86_linux.sio:2121, 2244
```

This is what makes the campaign tractable: widening is provably resolution-safe for every name
outside that list, so the work list is 1,276 sites minus 2 reviewed exceptions rather than
1,276 individually unsafe edits.

## Mitigation

The hazard exists because visibility participates in **resolution** rather than in
**validation**. Resolve by module-qualified identity, then check access as a post-hoc
predicate. Then widening cannot change resolution *by construction*, because resolution never
consults visibility.

Concretely: drop the visibility filter from T2, leaving resolution as "prefer same module,
else first match", and let the existing visibility predicate
(`check/check.sio:6571-6592 checker_fn_sig_visible_inplace`) report the violation afterwards.
Behaviour changes only for calls that today silently skip a private homonym to reach a public
one elsewhere — bounded by the detector's list, hence enumerable rather than open-ended.

## Generality

This is not a Sounio quirk. Any bootstrapped or self-hosted compiler that resolves names over
one flat merged table with a visibility-filtered fallback tier has the property, and a flat
merged table is what is reachable early in a bootstrap chain. Languages that resolve by path
rather than by name-with-fallback (Rust) do not, because visibility there never selects
*between* candidates.

## What is not established

- Whether the two named silent-flip pairs are semantically equivalent (their bodies were not
  compared); only that a flip between them is type-invisible.
- Whether the same hazard exists for types, enums and structs. `enum_table_find`
  (`check/defs.sio:353-363`) is plain first-match with **no** prefer-module tier at all, which
  suggests the enum case is worse rather than better, but no witness was constructed.
- Whether the proposed mitigation leaves the three visibility fixtures
  (`tests/multimodule/visibility_{fn,struct,enum}_private_main.sio`) rejecting. It should, since
  they contain no homonyms, but it was not measured.

## Related

The audit that surfaced this also established that module identity was the file **basename**
(`compiler/module_loader.sio:416-451`), so 20 files across 8 basenames were one module each to
the visibility predicate, and that on such a collision the predicate reached a miscompiled
`Option<Box<StringList>>` field read and crashed with no diagnostic — an access check
simultaneously vacuous and fatal. Identity is now parent-directory-plus-basename.
