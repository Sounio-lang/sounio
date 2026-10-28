# Typed storage authority — approved direction, not implemented

**Status: design proposal only. No stdlib API, checker admission, lowerer
integration, minter, runtime state machine or final-IR authority proof is
implemented by this document. The existing HeapVec draft is not safe or
qualified. Root owns architecture review and coordinates frontend/services.**

This records the approved direction after the coherent text replay. It does
not authorize editing a foreign lowerer lane or admit rawpointer+owner as a
safe API. Symbol spellings below are proposed contracts, not declarations that
are claimed to compile today.

## Existing foundation and exact gap

Foundation audit baseline: root commit `792a602a04715bb43b47b541e670f26998f43603`
(and its approved predecessor audit), before any implementation of this design.
The complete coherent-text native replay is recorded in
[its evidence](../evidence/stdlib-wave3-core/coherent-full-native/README.md).
This is a source/design observation, not a current global compiler gate.

The inspected compiler already has:

- compiler/trusted_origins.sio: physical dev/inode identity for the selected
  SDK's mem/typed_storage.sio, a storage module flag, and
  trusted_origin_is_storage. A source module heading confers no authority.
- ir/storage_intrinsics.sio: StorageCellDescriptor, target cell ABI version1,
  compiler-only admission flags, protected storage_ir_call and registration.
- native/codegen_x86_linux.sio: protected @storage1:* names that cannot be
  spelled as source identifiers.
- Worker capability kinds, a physical-origin raw extraction gate, source
  signature identities and final native proof as an architectural precedent.

There is no mem/typed_storage.sio facade or StorageAuthority minter in that
inspected root snapshot, and the descriptor/factory have no checker/lowerer
callers there. Their existence is not frontend or ownership acceptance.

The descriptor is target value transport, never host/C sizeof: Unit has
size0/align1; admitted word/float/handle cells have size8/align8; wide and
Knowledge payloads are blocked. Registration of a nominal name is not proof
that its hidden fields contain no borrow, closure or unsupported payload.
Unknown remains refused.

## Ownership object

Storage<T> is always linear and opaque, including Copy elements and Unit.
It owns private allocation base/aligned data address, capacity, live-slot
bitmap, initialized_count, byte accounting, descriptor/ABI identity, and a
compiler-minted managed authority. No raw address, integer owner handle or
public constructor can manufacture that authority.

The authority's managed owner value exists even for Unit and for an empty
storage. Data allocation may be empty at capacity0; a null owner is never the
empty representation. create_budget must fail if its budget cannot fund the
authority's own metadata. There is no fabricated success sentinel.

HeapVec and HeapDeque own Storage<T> and keep their logical len/head/tail
outside it. The bitmap describes physical initialized slots rather than
assuming one initialized prefix. This supports holes, wrapped Deque spans
and intermediate move states without silently destroying T.

The managed owner is the barrier target. The address of an &Storage parameter,
an mmap base, Unit owner0 or an arbitrary &O is not owner evidence.
STORAGE_PROOF_NO_BORROW and PAYLOAD_COMPLETE are element proofs; they do not
certify allocation authority or pointer provenance.

## Proposed contract surface

Contract notation (not implemented signatures):

- storage_create_budget<T>(max_bytes) -> Result<Storage<T>, StorageError>
- storage_reserve<T>(&!Storage<T>, required_capacity) -> Result<(), StorageError>
- storage_move_into<T>(&!Storage<T>, index, T) -> Result<(), (StorageError,T)>
- storage_move_out<T>(&!Storage<T>, index) -> Result<T,StorageError>
- storage_move_range<T>(&!Storage<T>, destination, source, count)
  -> Result<(),StorageError>
- storage_borrow<T>(&Storage<T>, index) -> Result<&T,StorageError>
- storage_borrow_mut<T>(&!Storage<T>, index) -> Result<&!T,StorageError>
- storage_close_empty<T>(Storage<T>) -> Result<(),Storage<T>>
- storage_dispose<T,C>(Storage<T>, &!C, consumer: fn(&!C,T)->()) -> ()
- Read-only capacity/live-count/charged-byte queries may expose scalar
  accounting. They never expose authority bits or data pointers.

create_budget rejects unproved element types during compilation. Runtime
errors describe budget/configuration/allocation/index/occupancy failures,
not an unknown-type fallback with a guessed layout.

move_into requires an uninitialized destination. A recoverable failure
returns the exact unconsumed T. move_out requires a live slot and clears its
state once. An empty/uninitialized cell must never fabricate a T.

The initial move_range contract is within one authority. All source slots
must be live; destination slots outside the source overlap must be empty.
Overlaps behave as a move, preserving order and clearing only vacated source
slots. All preconditions are checked before any move; an error leaves cells,
bitmap/count and owner unchanged. Zero-count and endpoint rules are explicit
and checked without overflowing start+count.

reserve preserves live physical slot indices. It acquires all fallible
replacement data/bitmap resources before moving values, then commits and
releases old allocations. A stable authority can own old and pending blocks
during that transaction; neither address escapes. General cross-authority
append/split is outside this first contract, rather than being inferred from
the existing one-owner raw range ABI.

HeapDeque must explicitly rebase wrapped spans when capacity changes; the
storage does not reinterpret logical head/tail. Tight-budget growth and both
range overlap directions need actual linear-owner tests.

close_empty refuses a nonempty store and returns the owner intact. dispose
moves each live T out before calling a consumer that must consume it. It has
no implicit element Drop. Recoverable consumer errors belong in the explicit
context while ownership is consumed; panic unwinding/resource cleanup is not
promised by this design.

## Budget rule proposed for review

Every per-storage metadata allocation is charged: authority/control object,
bitmap, allocation alignment overhead and pending reserve resources as well
as data. All arithmetic uses checked target/runtime sizes and range bounds.

The proposed rule is a hard peak of *owned requested bytes* during reserve,
including simultaneous old/new blocks and metadata. It may refuse growth
whose final retained footprint would fit. There is no hidden uncharged
bitmap or transaction buffer.

This is not automatically an RSS/physical-page bound. Shared allocator/arena
page rounding and bookkeeping need the canonical allocator cost model before
a stronger physical-memory claim is made. Per-storage headers cannot be
guessed with host sizeof or silently omitted. Root must confirm this peak
accounting policy before implementation; the old unqualified draft's
retained-data-only comment is not its specification.

Unit has no payload load/store. A conservative positive logical stride gives
each allocated Unit slot an actual aligned address, charged to the budget;
bitmap/count transitions still happen. The authority itself remains managed
and nonzero. Returning a sentinel or loading eight payload bytes for Unit is
not an implementation.

## Three validation layers

1. Compiler-private minter and source admission. Only the selected physical
   mem/typed_storage module and the resolved exact function/signature may use
   raw operations. Concrete T, target descriptor and value representation are
   resolved before proof flags are generated. User arguments cannot supply
   proof flags or authority bits.
2. Runtime metadata. Validate a live authority/generation, capacity/bounds,
   slot occupancy, descriptor/ABI and byte budget before touching payload.
   Keep bitmap/count and values consistent across success and every
   recoverable failure. Runtime metadata is not replaced by type names.
3. Final IR provenance. Revalidate mint origin, authority flow, concrete
   descriptor/call ABI and address/range derivation after specialization,
   cloning, inlining and other transformations. Unknown or lost evidence
   refuses emission. An approved guarded implementation must dominate the
   protected memory operation; physical origin alone is not a bounds check.

The authority needs a non-forgeable semantic kind/provenance, not just private
struct fields. Existing ordinary nominal casts largely return target_ty, so
field privacy alone is insufficient. Reject direct and recursively nested
authority production by casts, raw dereference, manual construction,
representation extraction/repacking or fabricated aggregate values. Preserve
legitimate owned movement through permitted aggregates; unknown provenance
does not become a mint.

A module-name homonym, copied file at a different physical identity, shadowed
helper, generic placeholder or bare-name layout match cannot obtain admission.
Reexports/aliases must retain the real defining identity. A user deliberately
selecting another SDK is a separate trust choice, not evidence that a same-name
module in the current SDK graph has authority.

The current effect enum has no Unsafe effect. Do not advertise a fictitious
with Unsafe facade. Raw storage remains compiler-private and origin-gated;
a future explicit unsafe language mechanism would require its own design and
gates. It is not a prerequisite that can be pretended to exist here.

## Initial element boundary

Positive targets: Unit, admitted scalar words, f32/f64 cells, and resolved
records/owned handles whose complete recursive payload proves no borrow or
captured closure. NativeBuffer and NativeChannel are required ownership
witnesses, not substitutes for arbitrary T.

Borrowed T (including nested references), wide integers/floats, Knowledge and
unknown/incomplete payloads remain refused initially. Bare function values
require actual no-capture/value-provenance proof; a function type annotation
alone does not earn it. Other protected capabilities are not ordinary words.

This does not claim unrestricted T, concurrency/Send/Sync, exception safety,
all aggregate copy semantics or a generic linear container from Word8 alone.

## Adversarial acceptance matrix

Every row is a required future gate, not an executed receipt in this document.
Numerical/bit/resource concordance uses C++23 and Sounio under Slurm.

| Boundary | Positive witness | Discriminating rejection/control |
| --- | --- | --- |
| Mint origin | Real physical SDK module and exact resolved mint signature | Same module heading in another file, shadowed helper, copied homonym and direct external raw call cannot mint |
| Authority representation | Owned Storage moves through approved container fields | Cast to/from integers/raw pointers, raw-deref manufacture and fabricated authority nested in struct/enum/tuple/array/Box carriers refuse recursively; genuine affine owned movement remains permitted |
| Final IR evidence | Unchanged mint/call/descriptor provenance survives allowed transforms | Missing/remapped authority metadata, altered body/call ABI, forged protected stub or unknown alias refuses emission; no Nop/value fallback |
| Target ABI | i8/word, f64, Unit and concrete record use canonical cells | Host sizeof assumptions, ABI mismatch, unsupported wide/Knowledge and unresolved nominal/generic payload refuse |
| Floating cell bits | Interleaved integer/f64 operations preserve -0 and NaN payload bits | A shared stub's global returns_float flag cannot retag all calls; wrong per-call marker is detected |
| Unit cell | Live slots borrow an actual address and dereference to Unit without payload load | Null/sentinel authority, fake address, skipped occupancy changes and an eight-byte Unit load fail |
| Initialization | Empty -> into -> borrow -> out -> empty | Read/out uninitialized, double out and overwrite live outside an allowed overlap fail without fabricated/lost T |
| Range movement | Both overlap directions, disjoint ranges and wrapped Deque rebasing preserve sequence/live counts | Live destination outside overlap, uninitialized source, overflow/index error and incorrect source clearing leave state unchanged or are rejected |
| Budget/transaction | C++ target cost model agrees with charged data + all metadata/pending bytes | Actual allocation failure at a reached operation, metadata allocation failure and checked overflow preserve owner/elements; no fake failure injection |
| Linear ownership | NativeBuffer/NativeChannel push/pop/replace/range/dispose conserve exact live/close counts | Reuse moved input, ignored linear Err, duplicate T, silent live close and lost element on failed reserve/push refuse |
| Returned loans | &T and affine &!T remain attached to the owning Storage and outer Vec/Deque | Grow/remove/free/move-owner while borrowed; escape or duplicate through Option/Result/record/tuple/imported facade refuse |
| Mutable linear loan | Permitted in-place operations preserve a linear T | Overwriting/moving a live linear T through &!T without consuming/returning the old owner must refuse |
| Owner after pop | Pop T, close empty Storage, then use/close popped owner once | A source-owner region/barrier dependency cannot destroy or strand the moved-out value |
| Empty close/disposal | close_empty returns live owner on refusal; explicit consumer drains cells | Nonempty free, duplicate disposer invocation, unconsumed callback T and a claimed panic-unwind guarantee are rejected/not claimed |
| Container integration | Generic HeapVec and HeapDeque use the same qualified storage authority/ABI | Legacy HeapI64Vec success, layout-only Word8 probes or public wrappers without mint/loan proofs do not qualify generic linear APIs |

For returned loans, distinguish a borrowed *result* from borrowed T: returning
Option<&T> or Result<&T,...> is required when T itself is admissible, while
storing a borrowed element remains closed. Do not replace the API with
callback-only access to hide a missing loan graph.

Resource tests must include a recovered Err value that is still usable and
closed exactly once; a popped resource surviving storage close; nonempty
close preserving the owner; both range directions; and growth with wrapped
Deque content. Copy/aggregate tests must honor Sounio's existing value-copy
semantics rather than silently aliasing a handle because its ABI is one cell.

## Qualification and handoff

Implementation ownership stays with root/core/services under their claims.
This document does not edit checker/lower or create any facade file.
The existing HeapVec source snapshot remains an explicitly unqualified
rawpointer+owner draft, preserved byte-for-byte for review (draft SHA-256
`8dd70924b3a6e6ee2171d6b5d3e830a701b2d28b6158f5f3e980712d29b23950`).

Before an API is called safe or generic, retain original before failures,
paired positive/negative checker tests, actual native/IR controls, C++23
resource/bit oracles and source/compiler/runtime/ELF/JobID receipts.
Component layout/factory results, a complete public native matrix and final
whole-stdlib acceptance are separate claims.
