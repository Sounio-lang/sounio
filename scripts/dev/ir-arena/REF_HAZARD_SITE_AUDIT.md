# Audit: the `pkg` sites, and what auditing them turned up

Asked: audit the three `&(*ptr).array[i]` sites in `compiler/pkg/` that #1678
listed as candidates and left unchecked.

Answer up front: **all three are the broken shape, and all three are dead** —
`compiler/pkg/` is not in the compiler's module closure. But the audit falsified
the characterisation in #1678, and the corrected shape has **six live sites in
the compiler's own IR passes**.

## The three `pkg` sites

| site | shape | verdict |
|---|---|---|
| `pkg/lock.sio:217` (`lock_emit`) | `let e = &(*lock).entries[i as usize]` | broken shape, **not reachable** |
| `pkg/lock.sio:364` (`lock_build_string`) | same | broken shape, **not reachable** |
| `pkg/registry_client.sio:430` | `registry_print_package(&(*r).packages[i as usize])` | broken shape, **not reachable** |

`LockEntry` is ~296 bytes (`[i8; 64]`, `[i8; 32]`, `[i8; 64]`, `[i8; 128]`, plus
scalars) inside `[LockEntry; 64]` inside `SouniuLock`, reached through a
`&SouniuLock` parameter. That is the hazard shape exactly, and the reproducer
run confirms it rather than inferring it:

```
baseline_local=295071210553056  registry_shape=295071210553056
lock_shape_scalar=13602939470112  lock_shape_callee=295071210553056
(expect: 5 5 77 5)
```

All three forms wrong, including binding to a `let` reference and dereferencing
it in the *same* function, which is the `lock.sio` variant.

**Reachability.** The build's module closure is 113 modules; it contains
`ir/lower.sio` and `check/check.sio` and **zero** `pkg/` modules. Nothing outside
`compiler/pkg/` imports `pkg::*` (the six matches are comments in the parser
about the `pkg::mod::sym` *syntax*), `pkg/cli.sio` has no `fn main`, and no build
script compiles the subsystem. So the defect is present and latent: it would
corrupt every field of every `sounio.lock` entry the moment the subsystem is
wired up, and nothing today reads it.

#1678 acceptance criterion 2 is met for these three: shown not reachable.

## What the audit falsified in #1678

#1678 says the hazard needs the value to be **boxed**, and cites "without the
`Box` the same program prints 5 and 5". That line was measured with the wrong
compiler — `madaros --native-v2-compile`, not the seed. Under `lean_single` the
same program gives:

```
hoisted=5 inplace=288899952421056
```

Corrected matrix:

| compiler | `&local.arr[i]` (aggregate element) | `&(*box).arr[i]` |
|---|---|---|
| `madaros --native-v2-compile` | correct | segfault |
| **`lean_single` — the seed that builds the compiler** | **wrong address** | **wrong address** |

**The `Box` is irrelevant.** Taking the address of an aggregate element of an
array field is wrong on the seed whether the container is a local, a reference
parameter, or a `Box`. This matters because it changes what has to be audited:
"boxed only" would have cleared every site below.

Also falsified along the way, and worth keeping because it confounded a whole
run: **returning a large aggregate by value is broken too.** The first matrix
seeded its container from `make_seeded() -> Outer` and every one of the seven
cells came back garbage, including the control. Seed the container inline.

## The six live sites

Cross-referencing every address-of-array-element site against the 113-module
closure:

| site | shape | in closure |
|---|---|---|
| `ir/profile.sio:135` | `sprof_lookup(profile, &module.functions[fi as usize].name)` | **LIVE** |
| `ir/layout.sio:56` | same | **LIVE** |
| `ir/layout.sio:183` | same | **LIVE** |
| `ir/egraph.sio:686` | `eg_match_rule_node(graph, &rules.rules[r as usize], n)` | **LIVE** |
| `ir/egraph.sio:701` | same | **LIVE** |
| `ir/egraph.sio:752` | `eg_apply_match(graph, rules, &batch.matches[i as usize])` | **LIVE** |

Two sub-shapes, both measured broken, both with a correct hoisted control in the
same program:

```
hoisted_elem_tag=77  hoisted_field_sum=5
inplace_field_of_elem=281468932350672   inplace_elem=20909672890688
(expect: 77 5 5 77)
```

`&elem.field` (profile/layout) and `&elem` (egraph) fail alike.

### What the garbage drives

`profile.sio:135` is the one to look at first, because it does not merely print:

```sounio
let count = sprof_lookup(profile, &module.functions[fi as usize].name)
let target = sprof_promotion_target(count)
if target >= 0 {
    var func = module.functions[fi as usize]
    func.compile_strategy = target
    module.functions[fi as usize] = func
}
```

A garbage `count` selects a **compile strategy**. That is codegen input, not a
statistic. The fix is one line and the pattern is already on the next line —
`var func = module.functions[fi as usize]` is exactly the hoist needed.

`layout.sio` feeds `LayoutScore.count`, i.e. function ordering.

### Gating

- `sprof_apply_promotion` is called from `compiler/main.sio:5346` and
  `compiler/module_loader.sio:2924`, both inside the block that parses a profile
  file. **Opt-in, but a real compile path** — supply a profile and it runs.
- `eg_match_all_rules` / `eg_apply_all_matches` are private to `egraph.sio` with
  **zero external callers**; every external user goes through the separate
  `eg_small_*` API. Internally reachable at `:766`, `:1113`, `:1263`. Whether any
  of those roots is reachable from a compile is **not established here**.

## Not fixed here

The six live sites are outside what was asked and sit on shared IR passes. They
are recorded, measured, and left alone deliberately. The fix at each is the same
one line — hoist the element into a local before taking the reference.

Reproducers: `/tmp/cl2/{pkgshape,subshape}.sio`, and the committed
`repro_boxed_element_ref.sio` (whose name is now too narrow — the `Box` is not
the ingredient).

Refs #1678, #1649
