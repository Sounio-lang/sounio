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

## Declared array/slice length — separate correction

The first composed view-facade witness checks normally (verdict0) but the earlier
specializer then refuses an imported method. Slurm19120 reproduces its exact
32-program source closure with the real parser, physical import map, private
function identity pass (five renames) and actual specializer. Diagnostic-only
instrumentation identifies unicode.sio char_from_utf8(bytes:&[u8]), at source
offset3495: bytes.len() had no nominal receiver and collided with HeapVec<T>.len.
The diagnostic source is separately hashed and is not production code.

Production source683d539aacd6a76e47261973f01b0fc5c927efc2b34e5c1ac84eb7b1ad8e10f4
adds a proven array-shape fact, carrying only the builtin length capability.
Declared TypeArray (including reference unwrap and alias expansion), an actual
array literal and convergent array-result facts can establish it. Only zero-arg
len uses that capability; explicit turbofish is refused. Unknown receivers,
shadows and other methods still follow the fail-closed generic path. This is
not a nominal declaration or evidence for any slice payload ABI.

Slurm **19138** completed0:0 in2m57s: the portable actual-source gate now has
**30 cases (16 acceptances, 14 refusals)**. Before compile0/native1 and after
26-module type/effect verdict0/compile0/native0 all hold. New cases cover literal,
declared slice and alias-to-array receivers, and reject a shadowed receiver,
explicit turbofish or extra length argument. The source-manifest digest after
is edc3f5e0f1b4b73f7f75e7a1d8e01b075a47ba6701c5fc186d4b76cab76b7c71.

Slurm **19139** completed0:0 in13s using the exact production683d module without
instrumentation. It replays the same 32 customer programs and explicit
GRAPH.json through load_module_file, private_fn_identity_apply and
specialize_generics_multi. Before19120 refuses; after19139 reports
FACADES_DIAGNOSTIC_REFUSED0. Source bytes, import graph, compiler and executable
are retained. This is a component reproduction, not the source-built CLI or
native execution of the composed facade witness.

## Raw6259: original native method fixed, alias lowering remains refused

Slurm **19088** rebuilt an actual raw compiler6259ed416e90a9ae555fbf2407f50a187bc45bf6323d84f9ad1085e4ce90b0c1
from the f64ae source (the earlier30-case array fix is NOT in this artifact).
The untouched original HeapVec method witness changes from check1/compile1 with
the qualified f3ce baseline to check0/compile0/native0 and exact PASS. Six other
receiver positives run at0; five negatives retain check1/compile1 and no ELF.
The C++23 returned-loan gate passes51 checks,51 compilations,21 native cases and
all30 expected compile refusals, including its wrong-output controls.

The **aggregate worker/gate remains1**. aliases_fields checks0 but native
compilation refuses a body-less i64_get at the unchanged f4 guard. No ELF is
emitted. The lowerer has lost the declared nominal alias across a reference to a
field. The original fixture is retained and remains mandatory. This source
change and these receipts do not qualify general alias lowering, generic
collections, or the whole compiler/stdlib.

Source-manifest digest for19088 is
7d41c3898e9537a43da557eb055400e98f7a28c9335a9ccb9ec38e58991bab69.
The before/after logs, aggregate failure, exact alias refusal and strict loan
receipts are in raw-19088. The complete manifest and raw binary are durable.

Slurm19150 separately rebuilds the source683d array fix and repeats the raw
matrix with literal-array and negative array controls. The alias fixture remains
expected to succeed in that raw gate; its failure is not waived. A selected
view/facade gate may report its own bounded results with this alias limit
explicitly retained, but cannot turn the aggregate raw result into acceptance.

## Additional durable custody

All following files are under the existing receiver-dispatch01 durable directory:

- raw6259-original-method-and-alias-refusal.tar.gz:
  5b87c5e192834ece3755db040d529469c04e43a0669da21c4669f6d7f8058e27.
- array30-and-real-closure32.tar.gz:
  7749bd8db96d0415025871a2f9b9e2fca272b94b84c8602936598ef88cac5fe2.
- view-api-failed-history.tar.gz:
  169ee02fc7563fffdfbbaa03404b3a2f526171b7b44bf0d24b5464151ece0d7b.
- stdlib-head1ffa-base.tar.gz:
  dceff0ea12faca83f303451d1a6a2ef02882261384f891e33417c5424181f32d.
- owner-view-overlay-v3.tar:
  7fce3530b4b8a7943f00e8ee483464c5cbe5b652c231fc0e01474d176b366f3e.

Slurm19145 and19168 verified these archives and source reconstruction inputs.
The view source base is the core branch HEAD1ffa (legacy Mut signatures, adopted
budget accessor and immutable BIDI source), without its uncommitted generic
HeapVec draft. Failed view attempts preserve an escaped-tab guard error, missing
version metadata, and the composed-facade specialization refusal. Their partial
leaf native success is not a facade gate pass. CHANGELOG.md and CITATION.cff are
now explicit source inputs of the pending view gate.
