<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-172-private-compiler.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-172-private-compiler.readme
-->

# Source-bootstrap private-field borrow and nested-store enforcement

This independent compiler fix is based on remote main7ce. It does not copy the
SIO-172 phonon implementation into the compiler PR, weaken privacy, alter the
unknown-receiver field-guessing policy, or introduce field-name exceptions.

## Concrete defect and scope

The original phonon_factor_private program from reviewed SIO-172 ae3b3ba17c
takes a mutable borrow of PhononState.values across its defining module boundary.
The modular compiler rejects it with E259. Actual hosted Stage2 in PR3111,
FullSuite121064352211/run40158000017, accepted it; the suite failed with
“expected compile failure but passed”. The harness logged Stage2 MD5 prefix
bb26fcd2. That observation was kept separate from committed seed behavior.

Source inspection locates a separate typed borrow walker. It consumes
&object.field itself, bypassing the ordinary postfix walker's existing privacy
check. The x86 local/ref, explicit-dereference and global paths, plus the ARM64
local/ref and explicit-dereference paths, resolved the actual receiver struct
but looked up field offsets without calling the established privacy helper.

The fix calls st_private_access_denied with that already resolved struct and
actual field token before offset selection. Its existing rules remain intact:
public fields, defining-module access and the struct's own impl retain their
permissions; disallowed access reports E259. No fallback candidate search is
changed.

A paired before matrix also demonstrates the same omission in six typed
nested-store handlers: value-root scalar, reference-root scalar and
reference-root array stores on x86 and ARM64. Both resolved outer and inner
fields now use the same existing helper. Existing stack cleanup is preserved
on the array-store error path. The already guarded value-root array handlers
are unchanged.

The production delta is101 added lines at17 guard sites in lean_single.sio:
five borrow sites and twelve field checks in six nested-store handlers.
Other compiler/checker/native-codegen files and the privacy helper itself
are unchanged. This establishes these concrete paths, not a universal audit
of every frontend or receiver type-inference path.

## Real source-built before and after

All heavy work uses Slurm CPU, UID1000, stack524288KiB, one CPU and4GiB job
memory. No Python/Rust/Julia reference or shared quota change is involved.

The unmodified existing selfhost_host_gate runs with forced source bootstrap
and an explicit verified bootstrap seed. It builds the host compiler from the
current source, then Stage2 and Stage3, compares executable payloads and runs
host/cross-target smokes. Host/Stage2/Stage3 artifacts and exact source identities
are retained separately for before and after.

| Receipt | Source / result |
|---|---|
| before02/job17372 | Fresh unmodified main source build; host,Stage2,Stage3 all SHA8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc |
| matrix03/job17373 | Reachable private borrow/store before cases accepted; exact original phonon program accepted and ELF retained; negative programs never executed |
| after04/job17374 | Patched source builds host,Stage2,Stage3 with identical full SHA13003a8b9b1bf4bfc64ad8d13eb0795d722888fabef5bbb2913b4f7b83bb454b; payload fixed point and smokes pass |
| qual05/job17376 | Stronger valid same-module/impl/ref/deref/nested public controls pass before and after |
| final08/job17379 | Final paired matrix with exact original phonon source/library hashes |
| final09/job17382 | Reviewed complete source manifest and EXIT status trap; focused gate passes, real old-compiler negative control fails with recorded status1 |

Fresh before Stage2 is bit-identical to the committed895 seed. This is an actual
source reproduction, not relabeling the old binary as a new build. Its full MD5
is bb26fcd2aebf6e0170b6d249db3cb149; that matches the hosted log's prefix only,
not an independently downloaded full hosted artifact hash.

The complete host gate takes34.58s before and34.95s after, with peak RSS
108564KiB and109104KiB in these individual runs. These are custody/resource
observations, not a controlled performance benchmark. The hello-host and
cross-smoke binaries retain identical before/after hashes.

## Regression matrix and scientific compatibility

The gate checks22 E259 cases per target on x86_64-linux and aarch64-linux:
shared/mutable field borrows, local/ref/explicit-dereference/chain/array paths,
global private access, both nested assignment levels, a real current-main
NumericBuffer private field, and existing1386 read/write/forge regressions.
Each refusal must exit1, report E259 and leave no nonempty output ELF.
Every forbidden access is reachable from main; no unsafe negative ELF is run.

Three positive fixtures per target cover public same-name fields, defining-module
private access, own impl mutation, shared/mutable field reborrows, explicit
dereferences, and value/ref nested public scalar/array stores. The x86 programs
run and check their outputs; ARM64 receives real ELF-header validation only,
without an ARM64 runtime claim.

A public global-field borrow is separately compile-qualified on x86. On ARM64,
the existing path types it as a value and reports E001. The private ARM64 global
counterparts already reported E259 before the fix. This pre-existing public
global limitation is recorded outside the required positive-pass count; the
probe also accepts a future valid ARM64 ELF rather than pinning broken behavior.
It is not a new known-failure baseline entry or an ignore/requires annotation.

The byte-exact original phonon negative and its frozen SIO-172 library closure
are staged OUTSIDE tracked compiler sources for before/after replay. Raw
lean_single resolves stdlib relative to its current working directory, so the
replay runs in that frozen source root. The environment variable alone does not
supply the extra library.

The new Stage2 also compiles and runs the real imported phonon positive corpus:
171 core values,30 SI parity values,13 guard values and31 statuses agree with
the unchanged independent C++23 decimal100 reference. This includes the original
145 core expectations. No scientific equation, input or numerical tolerance is
modified by this compiler fix.

The additional frozen production DP54 corpus also passes with the new Stage2
under the existing dual-engine gate, Slurm17378:304 values,7 dense nodes,
34 metadata records,22 event cases and12 invalid/reset cases, with independent
C++23 reference agreement. Here the gate column named seed is explicitly bound
to the new source-built Stage2 SHA13003a8, not the old committed895 binary.
The modular comparison compiler remains the separately verified c21 a292 pin.

## Environment failures and artifact handling

Before01/job17371 completed native source bootstrap but the canonical host gate
could not classify the output because this slim worker lacks the real file CLI.
That is an environment failure, not a compiler failure. A genuine file5.45 binary,
libmagic and its database were staged from the workspace into an isolated
tool directory, with hashes, version and dependency receipts. No fake file shim,
global package install or host-gate source change is used.

The first before02 original-phonon attempt ran from the main-only source root
and failed an unreadable import; it is explicitly invalid as a privacy witness.
Matrix03 replays the byte-exact source from the proper extra-library root and
records the actual acceptance. The initial positive matrix also exposed an
invalid simultaneous shared/mutable borrow in the fixture; the corrected
positive uses ordinary call-scoped reborrows and passes both compilers.

Compiler ELFs and unsafe negative outputs remain in immutable worker custody,
outside Git. Subsequent qualifications reuse the verified before/after compiler
paths instead of copying compiler archives. No bundled compiler gzip is
materialized, no historical packet is overwritten and no shared quota changes.

Worker root:
 /scratch/sounio-stdlib-inference-scale-20261024/SIO172private.

## Reproduction

Build the compiler with scripts/ci/selfhost_host_gate.sh and forced source
bootstrap. Then run scripts/ci/private_field_borrow_gate.sh with an absolute
SOUNIO_PRIVATE_COMPILER pointing to the freshly built Stage2 and a new absolute
SOUNIO_PRIVATE_OUTPUT. The gate hashes source/compiler and verifies them after
the run; an EXIT trap records STATUS.txt on both pass and failure. The manifest
covers the legacy fixtures and their imported module as well as the new cases.
SOUNIO_PRIVATE_EXTRA172_ROOT optionally names the frozen scientific
source packet for the exact historical phonon negative; compiler CI does not
require or import that feature branch.

Hosted full-suite/whole-source CI remains a separate result from the focused
Slurm gate. Main merge, branch integration and publication require their own
review/custody; none is implied by a local pass.
