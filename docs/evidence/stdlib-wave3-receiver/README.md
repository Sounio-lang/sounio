# Concrete receiver dispatch — component qualification

The specializer previously selected and finalized generic methods by method name
alone. A call to HeapI64Vec.get therefore selected HeapVec<T>.get; the same
finalizer could remove or replace a concrete implementation. Separately, pruning
an unused template did not arm the module-frontend handoff marker, so native
compilation could reload the original body with unbound T after check succeeded.

The change derives receiver facts from nominal declarations identified by module,
declaration kind, full name and source span. It records methods by their defining
owner and receiver declaration, preserves concrete definitions, and rejects an
unknown or ambiguous receiver when a generic method homonym is relevant. Binding
facts are scoped through functions, blocks, closures, for binders and match
patterns. Alias/reference and field facts preserve nominal identity; public
reexports and qualified paths use the existing physical import map.

A constructor result is inferred only when **every** matching declaration has the
same physical nominal result. This accommodates the original private/public
heap_f64_vec_new declarations without choosing a callable by name. Any unknown
or differing return erases that fact. The ordinary checker still resolves calls,
access, argument types, effects and ownership. Generic method calls retain the
existing unique explicit singleton requirement; this adds neither implicit
generic method inference nor ambiguous generic dispatch.

Pruned templates now count as AST transformations in the existing handoff marker.
Plain programs with no specialization/pruning still report zero. The qualified
root preseed guards from a7eb6b2ae7 remain a required compiler prerequisite.

## Exact-source component receipt

Slurm **19086** completed 0:0 in 3m00s on DL380. The portable gate compiled the
actual production module without visibility changes or a mock resolver:

- Baseline specializer: e497a3b45039f2b7facc2dc6b0594e42f4c23c554ba2c8172c686bfaed7482b9.
- Changed specializer: f64ae053e17ed646a2d2faafa9118c037325db492f827fe868eb2220a56268ec.
- Common legacy heap module: 195ccc8b0e2a7758d571c84f25855bde9a55dc30275ec7c3264142994d0f9010
  (only the separately committed four Mut annotations; no generic HeapVec draft).
- Type/effect checker: qualified f3ceba1af545a006ab61c66dafde4bbc5eb63c0b1463f63d6ed2f3dd853cb332,
  26 actual modules, verdict 0.
- Legacy native compiler: 9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a.
- Before: compile 0, component native 1. After: compile 0, component native 0.

The 22 controls comprise 13 expected acceptances and 9 expected refusals:
constructor/parameter/alias/field receivers; a same-named free function; unused
generic homonyms; the existing explicit generic method case; invalid generic
inference/turbofish; lexical and pattern shadows; physical module homonyms;
named-over-glob; ambiguous/private import edges; public and fully qualified
reexports; converging/diverging constructor aliases; and the untouched original
HeapVec methods source through the actual multi-program specialization entry.

The original tests/stdlib/collections/test_heap_vec_methods.sio has SHA256
e79a029f4e8cb6ef5750b8f08330e95d3d01bf0a3714ad0f463a90c2609473d3.
Both component fixtures are byte-identical copies. ORIGINAL_HEAP failed=0 means
its actual AST was specialized without a refusal and retained both concrete
method families while dropping the unused generic family. **It is not yet a
native execution receipt for that stdlib program.**

Source-manifest SHA256 before:
e8f03c39da35305246b43a76fd48520d9aa102f9aef50ca83d666c77ad1bbc1a.
After:
647068d96f1eb80beef0d2ffc383d88f381937fcb951bc081aa7e14c80f849e0.
The input overlay is
6236587496a6570a03628430a4fc86a326cd522ae6fc2532a8440e132ca1497c.

## Preserved history and custody

19074 first demonstrated the 12-control component improvement. Its first modern
check, 19075, rejected the driver's inline Name literal syntax. 19078/19079
qualified the corrected driver and 12-control native/type pair. 19081/19082
preserve a driver lexer error caused by adjacent nested-type closers and equals.
19083 preserves the expanded native matrix's single remaining failure: the
original duplicate F64 constructors. 19084 passed type/effect checking of that
earlier candidate. None is relabeled as final acceptance.

Slurm19092 durably archived attempts receiver-dispatch01 through05, input source
overlays, native component executables, logs, manifests, verification receipts
and workers at:

/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/receiver-dispatch01/component-history.tar.gz

SHA256 a8f62a6af497c784bb7380d371c090f6ee5780ce63e5b78782af79304ffef7ce.

Unchanged baseline source is reconstructed from the previously qualified
core-preseed-raw-18900 closure (Slurm18931, raw f3ce), whose durable custody is
recorded by the owner-views-replay receipt. Reconstructed source is checked by
the full per-variant manifests. Redundant expanded source trees are excluded
from this archive; their exact overlays and manifests are retained.

## Open integration gate

Slurm **19088** rebuilds a raw CLI from the immutable qualified f3ce source plus
only this specializer and the common legacy Mut annotations, then requires the
returned-loan 51-check/51-compile/21-native matrix, seven concrete receiver native
programs, five negative programs, and the original HeapVec program before/after.
No root storage/origin/Worker/loop-region WIP is included.

This document qualifies the component and type/effect check. Full CLI/native,
owner-view facade, strict BIDI/CLDR, generic linear storage and whole-stdlib
acceptance remain separate until their own receipts complete.

## Additional lexical-scope controls

Slurm **19102** completed 0:0 in 2m55s with the identical production specializer
f64ae053 and the same baseline, common legacy heap module and compiler pins.
The expanded gate has **24 cases: 13 acceptances and 11 refusals**. The two added
negative inputs shadow an outer concrete receiver with a closure parameter or
a for-loop binder. They require the specializer to discard the outer nominal
fact and refuse the unresolved generic homonym. The original production source
and original HeapVec fixture are unchanged.

Before compile0/native1 and after typecheck0/compile0/native0 all hold in this
separate replay; the 26-module modern check reports verdict0. Its component
receipt is retained under component-19102. This tests actual lexical-fact
scoping without claiming native support for a user closure program.

Slurm19103 archived its overlay, component executables, workers, logs and
manifests in the same durable directory as:
scope-controls-19102.tar.gz

SHA256 53f3770140bfdab59feb52ed900eec98a21567eb2c350048c10cff151d20bb5d.
The earlier 22-case receipt remains unchanged. Neither component replay is
retagged as the pending source-built CLI integration result.
